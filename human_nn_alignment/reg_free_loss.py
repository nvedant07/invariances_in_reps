from pytorch_lightning import utilities as pl_utils
from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.callbacks import BasePredictionWriter
from torchvision.transforms import transforms
from pytorch_lightning.core.lightning import LightningModule
import torch
import pathlib, argparse
try:
    from training import NicerModelCheckpointing, LitProgressBar
    import architectures as arch
    from attack.callbacks import AdvCallback
    from attack.attack_module import Attacker
    from attack.losses import LOSSES_MAPPING
    from architectures.callbacks import LightningWrapper, AdvAttackWrapper, LinearEvalWrapper
    from architectures.inverted_rep_callback import InvertedRepWrapper
    from architectures.inference import inference_with_features
    from datasets.data_modules import DATA_MODULES
    from datasets.dataset_metadata import DATASET_PARAMS
    from self_supervised.simclr_datamodule import simclr_dm
    from self_supervised.simclr_callback import SimCLRWrapper
    from human_nn_alignment.save_inverted_reps import save_tensor_images, get_classes_names
    from human_nn_alignment.transforms import compose, jitter, pad, random_scale, random_rotate
    from human_nn_alignment.fft_image import rfft2d_freqs, fft_image
except:
    raise ValueError('Run as a module to trigger __init__.py, ie '
                     'run as python -m human_nn_alignment.reg_free_loss')
from functools import partial

parser = argparse.ArgumentParser(description='PyTorch Visual Explanation')
parser.add_argument('--dataset', type=str, default='cifar10')
parser.add_argument('--model', type=str, default='resnet18')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--checkpoint_path', type=str, default='')
parser.add_argument('--append_path', type=str, default='')
parser.add_argument('--inversion_loss', type=str, default='reg_free')
parser.add_argument('--trans_robust', type=bool, default=False)
parser.add_argument('--fft', type=bool, default=False)
parser.add_argument('--step_size', type=float, default=1.)
parser.add_argument('--seed_type', type=str, default='super-noise')
parser.add_argument('--iters', type=int, default=None)
args = parser.parse_args()


TRANSFORMS = {'cifar10': [jitter(8),
                          random_scale([1 + (i - 5) / 50.0 for i in range(11)]),
                          random_rotate(list(range(-10, 11)) + 5 * [0]),
                          jitter(4)],
              'imagenet': [pad(12, mode="constant", constant_value=0.5),
                           jitter(8),
                           random_scale([1 + (i - 5) / 50.0 for i in range(11)]),
                           random_rotate(list(range(-10, 11)) + 5 * [0]),
                           jitter(4)]}

def initialize_seed(input_size, seed, fft):
    ## if fft then seed must be initialzed in the fourier domain
    ## the descent will happen in fourier domain
    shape = (3,input_size,input_size) if not fft else \
        (3,*rfft2d_freqs(input_size, input_size).shape,2)
    if seed == 'super-noise':
        return torch.randn(*shape)
    if seed == 'white':
        return torch.ones(*shape)
    if seed == 'black':
        return torch.zeros(*shape)
    if seed == 'light-noise':
        return torch.randn(*shape) * 0.01

if __name__=='__main__':
    dataset = args.dataset
    model = args.model
    
    # checkpoint_path = '/NS/robustness_2/work/vnanda/adv-trades/checkpoints'\
    #                   '/resnet18/beta_1.0_eps_1.000/checkpoint.pt.best'
    # checkpoint_path = './tests/weights/resnet18_cifar10.pt'
    # checkpoint_path = './tests/weights/vgg16_cifar10.pt'
    # checkpoint_path = './tests/weights/resnet18_l2eps3_imagenet.pt'
    # checkpoint_path = '/NS/robustness_1/work/vnanda/CKA-Centered-Kernel-Alignment/'\
    #     'checkpoints/cifar10/resnet18/nonrobust/checkpoint_rand_seed_2.pt.best'
    checkpoint_path = args.checkpoint_path

    # inversion_loss = 'reg_free'
    inversion_loss = args.inversion_loss
    # inversion_loss = 'freq_rob'
    # inversion_loss = 'freq_rob_fft'

    append_path = args.append_path

    pretrained = True
    seed = 2
    devices = 1
    num_nodes = 1
    strategy = DDPPlugin(find_unused_parameters=True) if devices > 1 else None

    data_path = '/NS/twitter_archive/work/vnanda/data' if dataset == 'imagenet' \
        else '/NS/twitter_archive2/work/vnanda/data'

    dm = DATA_MODULES[dataset](
        data_dir=data_path,
        val_frac=0.,
        subset=100,
        batch_size=args.batch_size)
    
    init_seed = initialize_seed(dm.input_size, args.seed_type, args.fft)
    m1 = arch.create_model(model, dataset, pretrained=pretrained,
                           checkpoint_path=checkpoint_path, seed=seed, 
                           callback=partial(InvertedRepWrapper, 
                                         seed=init_seed,
                                         dataset_name=dataset))

    custom_loss = LOSSES_MAPPING[inversion_loss]
    custom_loss._set_normalizer(m1.normalizer)
    if args.fft:
        custom_loss._set_fft(
            fft_image(
                (dm.batch_size, 3, dm.input_size, dm.input_size)
                )
            )
        print (hasattr(custom_loss, 'fft_transform'))
    if args.trans_robust:
        # using standard transforms from lucid 
        # (https://github.com/tensorflow/lucid/blob/master/lucid/optvis/transform.py)
        custom_loss._set_transforms(
            compose(TRANSFORMS[dataset]))
    adv_callback = AdvCallback(constraint_train='unconstrained',
                               constraint_test='unconstrained',
                               constraint_val='unconstrained',
                               eps_train=100.,
                               step_size=args.step_size,
                               iterations_train=1,
                               iterations_val=5000 if args.iters is None else args.iters,
                               iterations_test=5000 if args.iters is None else args.iters,
                               random_start_train=False,
                               random_restarts_train=0,
                               return_image=True,
                               targeted=True,
                               use_best=True,
                               do_tqdm=True,
                               should_normalize=False, # normalizer is implemented in losses
                               custom_loss=custom_loss)

    pl_utils.seed.seed_everything(seed, workers=True)

    trainer = Trainer(accelerator='gpu', devices=devices,
                      num_nodes=num_nodes,
                      strategy=strategy, 
                      log_every_n_steps=1,
                      auto_select_gpus=True, 
                      deterministic=not args.trans_robust,
                      max_epochs=1,
                      check_val_every_n_epoch=1,
                      num_sanity_val_steps=0,
                      callbacks=[LitProgressBar(['loss']), 
                                adv_callback])

    out = trainer.predict(m1, dataloaders=[dm.val_dataloader()])
    if trainer.is_global_zero:
        ## do things on the main process
        og, ir, labels = out

        if args.fft:
            init_seed = custom_loss.fft_transform(init_seed.unsqueeze(0)).squeeze()
            ir = custom_loss.fft_transform(ir)

        path = f'{pathlib.Path(__file__).parent.resolve()}/results/generated_images/{dataset}/'\
            f'{dataset}_{model}_{inversion_loss}'
        if args.trans_robust:
            path = f'{path}_transforms_{args.trans_robust}'
        if args.fft:
            path = f'{path}_fft'
        if args.iters:
            path = f'{path}_{args.iters}'
        if append_path:
            path += f'_{append_path}'
        
        save_tensor_images(path, torch.arange(len(og)), args.seed_type, 
            ir, init_seed, og, labels, get_classes_names(dataset, data_path))

