from pytorch_lightning import utilities as pl_utils
from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning.plugins import DDPPlugin
import torch
import pathlib, argparse
from functools import partial
try:
    from training import LitProgressBar, NicerModelCheckpointing
    import training.finetuning as ft
    import architectures as arch
    from architectures.callbacks import LightningWrapper, LinearEvalWrapper
    from attack.callbacks import AdvCallback
    from datasets.data_modules import DATA_MODULES
    import datasets.dataset_metadata as dsmd
    from partially_inverted_reps.partial_loss import PartialInversionLoss, PartialInversionRegularizedLoss
    from partially_inverted_reps import DATA_PATH_IMAGENET, DATA_PATH
except:
    raise ValueError('Run as a module to trigger __init__.py, ie '
                     'run as python -m human_nn_alignment.reg_free_loss')


parser = argparse.ArgumentParser(description='PyTorch Visual Explanation')
parser.add_argument('--source_dataset', type=str, default=None)
parser.add_argument('--finetuning_dataset', type=str, default='cifar10')
parser.add_argument('--finetune_mode', type=str, default='linear')
parser.add_argument('--model', type=str, default='resnet18')
parser.add_argument('--batch_size', type=int, default=None)
parser.add_argument('--checkpoint_path', type=str, default='')
parser.add_argument('--append_path', type=str, default='')
parser.add_argument('--epochs', type=int, default=None)
parser.add_argument('--optimizer', type=str, default='sgd')
parser.add_argument('--step_lr', type=int, default=None)
### specific to partial representation
parser.add_argument('--mode', type=str, default='random')
parser.add_argument('--fraction', type=float, default=0.5)
parser.add_argument('--seed', type=int, default=2)

SEED = 2
NUM_NODES = 1
DEVICES = 2
STRATEGY = DDPPlugin(find_unused_parameters=False) if DEVICES > 1 else None
BASE_DIR = pathlib.Path(__file__).parent.resolve()

def lightningmodule_callback(args):
    if args.finetune_mode == 'linear':
        return LinearEvalWrapper
    else:
        return LightningWrapper


def main(args=None):
    if args is None:
        args = parser.parse_args()

    dm = DATA_MODULES[args.finetuning_dataset](
        data_dir=DATA_PATH_IMAGENET if 'imagenet' in args.finetuning_dataset else DATA_PATH,
        transform_train=dsmd.TRAIN_TRANSFORMS_TRANSFER_DEFAULT(224),
        transform_test=dsmd.TEST_TRANSFORMS_DEFAULT(224),
        batch_size=args.batch_size)
    dm.init_remaining_attrs(args.source_dataset)

    m1 = arch.create_model(args.model, args.source_dataset, pretrained=True,
                           checkpoint_path=args.checkpoint_path, seed=SEED, 
                           callback=partial(lightningmodule_callback(args), 
                                            dataset_name=args.source_dataset, 
                                            optimizer=args.optimizer,
                                            step_lr=args.step_lr)) ## assign mean and std from source dataset
    new_layer = ft.setup_model_for_finetuning(
        m1.model, 
        dsmd.DATASET_PARAMS[args.finetuning_dataset]['num_classes'],
        args.mode, args.fraction, args.seed)
    m1.__setattr__('on_save_checkpoint', 
        lambda checkpoint: checkpoint.update([['neuron_indices', new_layer.neuron_indices]]))

    pl_utils.seed.seed_everything(args.seed, workers=True)
    
    checkpointer = NicerModelCheckpointing(
        dirpath=f'{BASE_DIR}/'
                'checkpoints/'
                f'{args.model}-base-{args.source_dataset}-ft-{args.finetuning_dataset}/'
                f'frac-{args.fraction:.5f}-mode-{args.mode}-seed-{args.seed}-lr-{m1.lr}-bs-{dm.batch_size}/'
                f'{args.append_path}', 
        filename='{epoch}', 
        every_n_epochs=20, 
        save_top_k=1, 
        save_last=False,
        verbose=True,
        mode='max', 
        monitor='val_acc1',
        save_partial=ft.get_param_names(m1.model, args.finetune_mode))

    trainer = Trainer(accelerator='gpu', 
                      devices=DEVICES,
                      num_nodes=NUM_NODES,
                      strategy=STRATEGY, 
                      log_every_n_steps=1,
                      auto_select_gpus=True, 
                      deterministic=True,
                      max_epochs=args.epochs,
                      check_val_every_n_epoch=1,
                      num_sanity_val_steps=0,
                      sync_batchnorm=True,
                      callbacks=[
                        LitProgressBar(['loss', 
                                        'running_train_acc', 
                                        'running_val_acc']), 
                        checkpointer, 
                        ft.FinetuningCallback(args.finetune_mode)])
    trainer.fit(m1, datamodule=dm)


if __name__=='__main__':
    main()