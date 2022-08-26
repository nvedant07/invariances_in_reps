from pytorch_lightning import utilities as pl_utils
from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning.plugins import DDPPlugin
from torchvision.models.feature_extraction import get_graph_node_names

import torch
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
    from datasets.data_modules import DATA_MODULES
    import datasets.dataset_metadata as dsmd
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

DATA_PATH = '/NS/twitter_archive/work/vnanda/data'
ALT_DATA_PATH = '/NS/robustness_1/work/vnanda/data'
SEED = 2
NUM_NODES = 1
DEVICES = 2
STRATEGY = DDPPlugin(find_unused_parameters=True) if DEVICES > 1 else None
BASE_PATH = pathlib.Path(__file__).parent.resolve()


def main(args=None):
    if args is None:
        args = parser.parse_args()

    dm = DATA_MODULES[args.eval_dataset](
        data_dir=DATA_PATH if 'imagenet' in args.eval_dataset else ALT_DATA_PATH,
        val_frac=0.,
        transform_train=dsmd.TEST_TRANSFORMS_DEFAULT(384),
        transform_test=dsmd.TEST_TRANSFORMS_DEFAULT(384),
        batch_size=args.batch_size)
    dm.init_remaining_attrs(args.eval_dataset)

    m1 = arch.create_model(args.model, args.base_dataset, pretrained=True,
                           checkpoint_path=args.base_checkpoint, seed=SEED, 
                           callback=partial(LightningWrapper, 
                                            dataset_name=args.base_dataset,
                                            mean=torch.tensor([0,0,0]),
                                            std=torch.tensor([1,1,1]),
                                            inference_kwargs={'with_latent': True}))
    m2 = arch.create_model(args.model, args.finetuning_dataset, pretrained=True,
                           checkpoint_path=args.finetuned_checkpoint, seed=SEED, 
                           callback=partial(LightningWrapper, 
                                            dataset_name=args.finetuning_dataset,
                                            mean=torch.tensor([0,0,0]),
                                            std=torch.tensor([1,1,1]),
                                            inference_kwargs={'with_latent': True}))
    _, node_names = get_graph_node_names(m2.model)
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
    filtered_nodes = []
    if m2.model.__class__.__name__ == 'VisionTransformer':
        block_number_to_layer = {}
        for n in node_names:
            if n.startswith('blocks.'):
                current_block = int(n.split('blocks.')[1].split('.')[0])
                if current_block not in block_number_to_layer:
                    block_number_to_layer[current_block] = [n]
                else:
                    block_number_to_layer[current_block].append(n)
            if n in ['fc_norm']:
                filtered_nodes.append(n)
        for block in sorted(block_number_to_layer.keys(), reverse=True):
            filtered_nodes = [block_number_to_layer[block][-1]] + filtered_nodes
    elif m2.model.__class__.__name__ == 'ResNetV2':
        for n in node_names:
            if n.endswith('.pool') or n.endswith('.add'):
                filtered_nodes.append(n)
    model_info += '\n\nFiltered:\n'
    for n in filtered_nodes:
        model_info += f'{n}\n'
    print (model_info)

    pl_utils.seed.seed_everything(args.seed, workers=True)

    # results = ['layer1,layer2,m1m2,m2m1,cka']
    for i in range(len(filtered_nodes)):
        for j in range(len(filtered_nodes)):
            if os.path.exists(f'{BASE_PATH}/results/{args.model}_eval_{args.eval_dataset}-'
                f'base_{args.base_dataset}-finetune_{args.finetuning_dataset}.txt'):
                df = pd.read_csv(f'{BASE_PATH}/results/{args.model}_eval_{args.eval_dataset}-'
                    f'base_{args.base_dataset}-finetune_{args.finetuning_dataset}.txt', 
                    header=None, sep=',', index_col=[0,1])
                if (f'{i}({filtered_nodes[i]})',f'{j}({filtered_nodes[j]})') in df.index:
                    print (f'{i}({filtered_nodes[i]}), {j}({filtered_nodes[j]}) already done, skipping...')
                    continue
            stir_score = stir.STIR(m1, m2, 
                helpers.InputNormalize(dsmd.IMAGENET_INCEPTION_MEAN, dsmd.IMAGENET_INCEPTION_STD), 
                helpers.InputNormalize(dsmd.IMAGENET_INCEPTION_MEAN, dsmd.IMAGENET_INCEPTION_STD),
                (dm.test_dataloader(), args.total_imgs),
                verbose=True, 
                layer1_num=i, layer2_num=j,
                ve_kwargs={
                    'constraint': 'unconstrained',
                    'eps': 1000, # put whatever, threat model is unconstrained, so this does not matter
                    'step_size': 0.5,
                    'iterations': 500,
                    'targeted': True,
                    'should_normalize': True,
                    'use_best': True
                })
            res = f'{i}({filtered_nodes[i]}),{j}({filtered_nodes[j]}),{stir_score.m1m2:.6f},{stir_score.m2m1:.6f},{stir_score.rsm:.6f}'

            if not os.path.exists(f'{BASE_PATH}/results'):
                os.mkdir(f'{BASE_PATH}/results')
            with open(f'{BASE_PATH}/results/{args.model}_eval_{args.eval_dataset}-base_{args.base_dataset}-'
                    f'finetune_{args.finetuning_dataset}.txt', 'a') as fp:
                # fp.write('\n'.join(results))
                fp.write(f'{res}\n')


if __name__=='__main__':
    main()