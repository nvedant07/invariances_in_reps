from pytorch_lightning import utilities as pl_utils
from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning.plugins import DDPPlugin
from torchvision.models.feature_extraction import get_graph_node_names

import torch
import torch.nn as nn
from functools import partial
import argparse, os
import pathlib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import stir
import stir.model.tools.helpers as helpers
from timm.models.fx_features import GraphExtractNet

try:
    import architectures as arch
    from architectures.callbacks import LightningWrapper
    from architectures.utils import intermediate_layer_names
    from data_modules import DATA_MODULES
    import dataset_metadata as dsmd
except:
    raise ValueError('Run as a module to trigger __init__.py, ie '
                     'run as python -m human_nn_alignment.reg_free_loss')


parser = argparse.ArgumentParser(description='PyTorch Visual Explanation')
parser.add_argument('--eval_dataset', type=str, default='cifar10')
parser.add_argument('--base_dataset', type=str, default='cifar10')
parser.add_argument('--finetuning_dataset', type=str, default='cifar10')
parser.add_argument('--model', type=str, default='resnet18')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--total_imgs', type=int, default=None)
parser.add_argument('--base_checkpoint', type=str, default='')
parser.add_argument('--finetuned_checkpoint', type=str, default='')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--task', type=str, default=None)

DATA_PATH = '/NS/twitter_archive/work/vnanda/data'
ALT_DATA_PATH = '/NS/robustness_1/work/vnanda/data'
SEED = 2
NUM_NODES = 1
DEVICES = torch.cuda.device_count()
STRATEGY = DDPPlugin(find_unused_parameters=True) if DEVICES > 1 else None
BASE_PATH = pathlib.Path(__file__).parent.resolve()


def main(args=None):
    if args is None:
        args = parser.parse_args()

    dm = DATA_MODULES[args.eval_dataset](
        data_dir=DATA_PATH if 'imagenet' in args.eval_dataset else ALT_DATA_PATH,
        val_frac=0.,
        transform_train=dsmd.TEST_TRANSFORMS_DEFAULT(args.input_size),
        transform_test=dsmd.TEST_TRANSFORMS_DEFAULT(args.input_size),
        batch_size=args.batch_size)
    dm.init_remaining_attrs(args.eval_dataset)

    m1 = arch.create_model(args.model, args.base_dataset, pretrained=True,
                           checkpoint_path=args.base_checkpoint, seed=SEED, 
                           callback=partial(LightningWrapper, 
                                            dataset_name=args.base_dataset,
                                            mean=torch.tensor([0,0,0]), ## will be overridden later for stir computation
                                            std=torch.tensor([1,1,1]), ## will be overridden later for stir computation
                                            inference_kwargs={'with_latent': True},
                                            training_params_dataset=args.finetuning_dataset))
    m2 = arch.create_model(args.model, args.base_dataset, pretrained=True,
                           checkpoint_path=args.finetuned_checkpoint, seed=SEED, 
                           num_classes=dsmd.DATASET_PARAMS[args.finetuning_dataset]['num_classes'],
                           callback=partial(LightningWrapper, 
                                            dataset_name=args.finetuning_dataset,
                                            mean=torch.tensor([0,0,0]), ## will be overridden later for stir computation
                                            std=torch.tensor([1,1,1]), ## will be overridden later for stir computation
                                            inference_kwargs={'with_latent': True}),
                           loading_function_kwargs={'strict': False})

    ### TODO: make use of multiple devices, either use pytorch lightning or make use of DistributedDataParallel
    ## DataParallel leads to hanging
    # if DEVICES > 1:
    #     m1.model = nn.DataParallel(m1.model, device_ids=list(range(DEVICES))).to(f'cuda:{list(range(DEVICES))[0]}')
    #     m2.model = nn.DataParallel(m2.model, device_ids=list(range(DEVICES))).to(f'cuda:{list(range(DEVICES))[0]}')

    filtered_nodes, node_names = intermediate_layer_names(m2.model, return_all=True)
    model_info = ''
    if not os.path.exists(
        f'{pathlib.Path(__file__).parent.resolve()}/results/{args.model}-layers.txt'):
        feature_model = GraphExtractNet(m2.model, node_names)
        for X, _ in dm.test_dataloader():
            all_fts = feature_model(X)
            break
        model_info += 'All:\n'
        for n, rep in zip(node_names, all_fts):
            if isinstance(rep, tuple):
                model_info += f'{n}, '
                for x in rep:
                    model_info += f'{x.shape if hasattr(x, "shape") else x}, '
                model_info += '\n'
            else:
                model_info += f'{n}, {rep.shape if hasattr(rep, "shape") else rep}\n' 
        with open(f'{pathlib.Path(__file__).parent.resolve()}/results/{args.model}-layers.txt', 'w') as fp:
            fp.write(model_info)
    model_info += '\n\nFiltered:\n'
    for n in filtered_nodes:
        model_info += f'{n}\n'
    print (model_info)

    pl_utils.seed.seed_everything(args.seed, workers=True)

    # results = ['layer1,layer2,m1m2,m2m1,cka']
    for i in range(len(filtered_nodes)):
        for j in range(len(filtered_nodes)):
            res_filepath = f'{BASE_PATH}/results/{args.model}_eval_{args.eval_dataset}-'\
                           f'base_{args.base_dataset}-finetune_{args.finetuning_dataset}.txt' \
                           if args.task is None else \
                           f'{BASE_PATH}/results/{args.model}_eval_{args.eval_dataset}-'\
                           f'base_{args.base_dataset}-finetune_{args.finetuning_dataset}_{args.task}.txt'
            if os.path.exists(res_filepath):
                df = pd.read_csv(res_filepath, header=None, sep=',', index_col=[0,1])
                if (f'{i}({filtered_nodes[i]})',f'{j}({filtered_nodes[j]})') in df.index:
                    print (f'{i}({filtered_nodes[i]}), {j}({filtered_nodes[j]}) already done, skipping...')
                    continue
            print (f'Layer1num: {i}, Layer2num: {j}')
            stir_score = stir.STIR(m1, m2, 
                helpers.InputNormalize(dsmd.IMAGENET_INCEPTION_MEAN, dsmd.IMAGENET_INCEPTION_STD), 
                helpers.InputNormalize(dsmd.IMAGENET_INCEPTION_MEAN, dsmd.IMAGENET_INCEPTION_STD),
                (dm.test_dataloader(), args.total_imgs),
                verbose=True, 
                layer1_num=i, layer2_num=j,
                devices=list(range(DEVICES)),
                ve_kwargs={
                    'constraint': 'unconstrained',
                    'eps': 1000, # any value is fine, threat model is unconstrained, so this does not matter
                    'step_size': 0.5,
                    'iterations': 500,
                    'targeted': True,
                    'should_normalize': True,
                    'use_best': True
                })
            res = f'{i}({filtered_nodes[i]}),{j}({filtered_nodes[j]}),{stir_score.m1m2:.6f},{stir_score.m2m1:.6f},{stir_score.rsm:.6f}'

            if not os.path.exists(f'{BASE_PATH}/results'):
                os.mkdir(f'{BASE_PATH}/results')
            with open(res_filepath, 'a') as fp:
                # fp.write('\n'.join(results))
                fp.write(f'{res}\n')


if __name__=='__main__':
    main()