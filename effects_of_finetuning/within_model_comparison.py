from pytorch_lightning import utilities as pl_utils
from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning.plugins import DDPPlugin

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
    from architectures.utils import intermediate_layer_names
    from data_modules import DATA_MODULES
    import dataset_metadata as dsmd
except:
    raise ValueError('Run as a module to trigger __init__.py, ie '
                     'run as python -m human_nn_alignment.reg_free_loss')


parser = argparse.ArgumentParser(description='PyTorch Visual Explanation')
parser.add_argument('--eval_dataset', type=str, default='cifar10') # same as finetuned_ds
parser.add_argument('--base_dataset', type=str, default='cifar10')
parser.add_argument('--model', type=str, default='resnet18')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--total_imgs', type=int, default=None)
parser.add_argument('--checkpoint', type=str, default='')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--append_path', type=str, default='')
parser.add_argument('--image_size', type=int, default=224)

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
        transform_train=dsmd.TEST_TRANSFORMS_DEFAULT(args.image_size),
        transform_test=dsmd.TEST_TRANSFORMS_DEFAULT(args.image_size),
        batch_size=args.batch_size)
    dm.init_remaining_attrs(args.eval_dataset)

    m1 = arch.create_model(args.model, args.base_dataset, pretrained=True,
                           checkpoint_path=args.checkpoint, seed=SEED, 
                           num_classes=dsmd.DATASET_PARAMS[args.base_dataset if args.append_path == 'base' \
                                                           else args.eval_dataset]['num_classes'],
                           callback=partial(LightningWrapper, 
                                            dataset_name=args.base_dataset,
                                            mean=torch.tensor([0,0,0]), ## will be overridden later for stir computation
                                            std=torch.tensor([1,1,1]), ## will be overridden later for stir computation
                                            inference_kwargs={'with_latent': True},
                                            training_params_dataset=args.eval_dataset))
    filtered_nodes = intermediate_layer_names(m1.model)
    model_info = '\n\nFiltered:\n'
    for n in filtered_nodes:
        model_info += f'{n}\n'
    print (model_info)

    pl_utils.seed.seed_everything(args.seed, workers=True)

    for i in range(len(filtered_nodes)):
        for j in range(i + 1, len(filtered_nodes)):
            if os.path.exists(f'{BASE_PATH}/results/{args.model}_eval_{args.eval_dataset}-'
                f'base_{args.base_dataset}-{args.eval_dataset}-within-model-{args.append_path}.txt'):
                df = pd.read_csv(f'{BASE_PATH}/results/{args.model}_eval_{args.eval_dataset}-'
                    f'base_{args.base_dataset}-{args.eval_dataset}-within-model-{args.append_path}.txt', 
                    header=None, sep=',', index_col=[0,1])
                if (f'{i}({filtered_nodes[i]})',f'{j}({filtered_nodes[j]})') in df.index:
                    print (f'{i}({filtered_nodes[i]}), {j}({filtered_nodes[j]}) already done, skipping...')
                    continue
            stir_score = stir.STIR(m1, m1, 
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
            with open(f'{BASE_PATH}/results/{args.model}_eval_{args.eval_dataset}-base_'
                      f'{args.base_dataset}-{args.eval_dataset}-within-model-{args.append_path}.txt', 'a') as fp:
                fp.write(f'{res}\n')


if __name__=='__main__':
    main()