from pytorch_lightning import utilities as pl_utils
from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.callbacks import BasePredictionWriter
from torchvision.transforms import transforms
from pytorch_lightning.core.lightning import LightningModule
import torch
import pathlib, argparse
try:
    from training import LitProgressBar
    import architectures as arch
    from attack.callbacks import AdvCallback
    from architectures.inverted_rep_callback import InvertedRepWrapper
    from datasets.data_modules import DATA_MODULES
    from datasets.dataset_metadata import DATASET_PARAMS
    from human_nn_alignment.save_inverted_reps import save_tensor_images, get_classes_names
    from human_nn_alignment.transforms import compose, jitter, pad, random_scale, random_rotate
    from human_nn_alignment.fft_image import fft_image
    from human_nn_alignment.utils import initialize_seed, LOSSES_MAPPING, ADDITIONAL_DATAMODULES
except:
    raise ValueError('Run as a module to trigger __init__.py, ie '
                     'run as python -m human_nn_alignment.reg_free_loss')
from functools import partial

parser = argparse.ArgumentParser(description='PyTorch Visual Explanation')
parser.add_argument('--source_dataset', type=str, default=None)
parser.add_argument('--dataset', type=str, default='cifar10')
parser.add_argument('--model', type=str, default='resnet18')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--checkpoint_path', type=str, default='')
parser.add_argument('--append_path', type=str, default='')
parser.add_argument('--trans_robust', type=bool, default=False)
parser.add_argument('--fft', type=bool, default=False)
parser.add_argument('--step_size', type=float, default=1.)
parser.add_argument('--seed_type', type=str, default='super-noise')
parser.add_argument('--iters', type=int, default=None)


DATA_PATH = '/NS/twitter_archive/work/vnanda/data'

def main(args=None):
    if args is None:
        args = parser.parse_args()

    dm = DATA_MODULES(args.dataset)(
        data_dir=DATA_PATH,
        val_frac=0.,
        subset=100,
        transform_train=DATASET_PARAMS[args.source_dataset]['transform_train'],
        transform_test=DATASET_PARAMS[args.source_dataset]['transform_test'],
        batch_size=args.batch_size)
    dm.init_remaining_attrs(args.source_dataset)

    