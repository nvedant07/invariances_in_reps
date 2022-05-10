import sys
sys.path.append('../deep-learning-base')
from pytorch_lightning import utilities as pl_utils
from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.callbacks import BasePredictionWriter
from torchvision.transforms import transforms
from pytorch_lightning.core.lightning import LightningModule
import torch
try:
    from training import NicerModelCheckpointing, LitProgressBar
    import architectures as arch
    from attack.callbacks import AdvCallback
    from attack.attack_module import Attacker
    from attack.losses import LPNormLossSingleModel
    from architectures.callbacks import LightningWrapper, AdvAttackWrapper, LinearEvalWrapper
    from architectures.inverted_rep_callback import InvertedRepWrapper
    from architectures.inference import inference_with_features
    from datasets.data_modules import DATA_MODULES
    from datasets.dataset_metadata import DATASET_PARAMS
    from self_supervised.simclr_datamodule import simclr_dm
    from self_supervised.simclr_callback import SimCLRWrapper
except:
    raise ValueError('Run as a module to trigger __init__.py, ie '
                     'run as python -m human_nn_alignment.reg_free_loss')
from functools import partial


if __name__=='__main__':
    # dataset = 'cifar10'
    dataset = 'imagenet'

    model = 'vgg16'
    # model = 'resnet18'
    # model = 'densenet121'
    # model = 'resnet50'
    # model = 'convit_base'

    # checkpoint_path = '/NS/robustness_2/work/vnanda/adv-trades/checkpoints'\
    #                   '/resnet18/beta_1.0_eps_1.000/checkpoint.pt.best'
    # checkpoint_path = './tests/weights/resnet18_cifar10.pt'
    # checkpoint_path = './tests/weights/vgg16_cifar10.pt'
    # checkpoint_path = './tests/weights/resnet18_l2eps3_imagenet.pt'
    checkpoint_path = ''

    pretrained = True
    seed = 2
    devices = 1
    num_nodes = 1
    strategy = DDPPlugin(find_unused_parameters=True) if devices > 1 else None
    max_epochs = 100
    # max_epochs = DATASET_PARAMS[dataset]['epochs']

    imagenet_path = '/NS/twitter_archive/work/vnanda/data'
    data_path = '/NS/twitter_archive2/work/vnanda/data'

    dm = DATA_MODULES[dataset](
        data_dir=imagenet_path if dataset == 'imagenet' else data_path,
        val_frac=0.,
        subset=100,
        batch_size=32)
    adv_callback = AdvCallback(constraint_train='unconstrained',
                               constraint_test='unconstrained',
                               constraint_val='unconstrained',
                               eps_train=100.,
                               step_size=0.01,
                               iterations_train=1,
                               iterations_val=5000,
                               iterations_test=5000,
                               random_start_train=False,
                               random_restarts_train=0,
                               return_image=True,
                               targeted=True,
                               use_best=True,
                               do_tqdm=True,
                               custom_loss=LPNormLossSingleModel(lpnorm_type=2))
    init_seed = torch.randn(3,dm.input_size,dm.input_size)
    m1 = arch.create_model(model, dataset, pretrained=pretrained,
                           checkpoint_path=checkpoint_path, seed=seed, 
                           callback=partial(InvertedRepWrapper, 
                                         seed=init_seed,
                                         dataset_name=dataset))
    pl_utils.seed.seed_everything(seed, workers=True)

    trainer = Trainer(accelerator='gpu', devices=devices,
                      num_nodes=num_nodes,
                      strategy=strategy, 
                      log_every_n_steps=1,
                      auto_select_gpus=True, deterministic=True,
                      max_epochs=1,
                      check_val_every_n_epoch=1,
                      num_sanity_val_steps=0,
                      callbacks=[LitProgressBar(['loss', 'running_acc_clean', 'running_acc_adv']), 
                                adv_callback])

    out = trainer.predict(m1, dataloaders=[dm.val_dataloader()])
    if trainer.is_global_zero:
        ## do things on the main process
        og, ir = out
        print (og.shape)
        print (ir.shape)

