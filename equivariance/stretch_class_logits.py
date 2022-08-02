### For two NNs, stretches class logits by some amount, 
### inverts representations and tests it on another model

from pytorch_lightning import utilities as pl_utils
from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning.plugins import DDPPlugin
from torch.utils.data import DataLoader

import torch
from functools import partial
import glob, argparse
import pathlib
import numpy as np

try:
    from training import LitProgressBar
    import architectures as arch
    from attack.callbacks import AdvCallback
    from architectures.callbacks import LightningWrapper
    from .stretched_inversion_callback import StretchedInvertedRepWrapper
    from .utils import wrap_into_dataloader, DatasetFromImagesPaths
    from datasets.data_modules import DATA_MODULES
    from datasets.dataset_metadata import DATASET_PARAMS
    from human_nn_alignment.utils import initialize_seed, LOSSES_MAPPING
    from human_nn_alignment.save_inverted_reps import \
        save_tensor_images, get_classes_names, save_tensor_reps
    from equivariance.callbacks import VectorCallback
except:
    raise ValueError('Run as a module to trigger __init__.py, ie '
                     'run as python -m human_nn_alignment.reg_free_loss')


parser = argparse.ArgumentParser(description='PyTorch Visual Explanation')
parser.add_argument('--dataset', type=str, default='cifar10')
parser.add_argument('--model1', type=str, default='resnet18')
parser.add_argument('--model2', type=str, default='resnet50')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--inversion_loss', type=str, default='reg_free')
parser.add_argument('--checkpoint1_path', type=str, default='')
parser.add_argument('--checkpoint2_path', type=str, default='')
parser.add_argument('--step_size', type=float, default=1.)
parser.add_argument('--seed_type', type=str, default='super-noise')
parser.add_argument('--iters', type=int, default=None)
parser.add_argument('--vector_type', type=str, default='', choices=['class_logit', 'random'])
parser.add_argument('--vector_class_idx', type=int, default=None)
parser.add_argument('--mean', type=float, default=None)
parser.add_argument('--std', type=float, default=None)
parser.add_argument('--append_path', type=str, default=None)
parser.add_argument('--total_imgs', type=int, default=5000)


def construct_vector_params(args):
    if args.vector_type == 'class_logit':
        assert args.vector_class_idx is not None, \
            'Must pass vector_class_idx for vector_type = class_logit'
        return {'index': args.vector_class_idx}
    else:
        assert args.mean is not None and args.std is not None, \
            'Must pass mean and std for vector_type = random'
        return {'mean': args.mean, 'std': args.std}


def main(args=None):
    if args is None:
        args = parser.parse_args()

    pretrained = True
    seed = 2
    devices = 2
    num_nodes = 1
    strategy = DDPPlugin(find_unused_parameters=True) if devices > 1 else None
    data_path = '/NS/twitter_archive/work/vnanda/data'

    dm = DATA_MODULES[args.dataset](
        data_dir=data_path,
        val_frac=0.,
        subset=args.total_imgs,
        transform_train=DATASET_PARAMS[args.dataset]['transform_test'],
        transform_test=DATASET_PARAMS[args.dataset]['transform_test'],
        batch_size=args.batch_size)
    dm.init_remaining_attrs(args.dataset)

    init_seed = initialize_seed(dm.input_size, args.seed_type, False)
    m1 = arch.create_model(args.model1, args.dataset, pretrained=pretrained,
                           checkpoint_path=args.checkpoint1_path, seed=seed, 
                           callback=partial(StretchedInvertedRepWrapper, 
                                            seed=init_seed,
                                            dataset_name=args.dataset))
    m2 = arch.create_model(args.model2, args.dataset, pretrained=pretrained,
                           checkpoint_path=args.checkpoint2_path, seed=seed, 
                           callback=partial(LightningWrapper, 
                                            inference_kwargs={'with_latent': True},
                                            dataset_name=args.dataset))

    custom_loss = LOSSES_MAPPING[args.inversion_loss]
    custom_loss._set_normalizer(m1.normalizer)
    adv_callback = AdvCallback(constraint_train='unconstrained',
                               constraint_test='unconstrained',
                               constraint_val='unconstrained',
                               eps_train=100., # does not matter since threat model is unconstrained
                               step_size=args.step_size,
                               iterations_train=1,
                               iterations_val=5000 if args.iters is None else args.iters,
                               iterations_test=5000 if args.iters is None else args.iters,
                               random_start_train=False, random_restarts_train=0,
                               return_image=True, targeted=True,
                               use_best=True, do_tqdm=True,
                               should_normalize=False, # normalizer is implemented in losses
                               custom_loss=custom_loss)

    for stretch_factor in np.linspace(0., 100., 40):
        pl_utils.seed.seed_everything(seed, workers=True)
        vector_callback = VectorCallback(args.vector_type, 
            construct_vector_params(args), 
            stretch_factor)
        trainer = Trainer(accelerator='gpu', devices=devices,
                        num_nodes=num_nodes, strategy=strategy, 
                        log_every_n_steps=1, auto_select_gpus=True, 
                        deterministic=True, max_epochs=1,
                        check_val_every_n_epoch=1, num_sanity_val_steps=0,
                        callbacks=[LitProgressBar(['loss']), 
                                    adv_callback, 
                                    vector_callback])
        out_m1 = trainer.predict(m1, dataloaders=[dm.val_dataloader()])
        # CAUTION!!! order will be different for DDP strategy vs single GPU inference
        og, ir, labels = out_m1
        # dl_og = wrap_into_dataloader(og, labels, batch_size=args.batch_size)
        # dl_ir = wrap_into_dataloader(ir, labels, batch_size=args.batch_size)
        # out_m2 = trainer.predict(m2, dataloaders=[dl_og, dl_ir])
        # # CAUTION:!!!! If DDP is used then og_put will not have same order as og!!
        # (og_out, og_latent, og_y), (ir_out, ir_latent, ir_y) = out_m2

        classes_names = get_classes_names(args.dataset, data_path)
        path = f'{pathlib.Path(__file__).parent.resolve()}/results/generated_images/{args.dataset}/'\
            f'{args.dataset}_{args.model1}_{args.inversion_loss}_stretch_{stretch_factor:.2f}_{classes_names[args.vector_class_idx]}'
        if args.append_path:
            path += f'_{args.append_path}'
        if trainer.is_global_zero:
            save_tensor_images(path, torch.arange(len(og)), args.seed_type, 
                ir, init_seed, og, labels, classes_names)
            # save_tensor_reps(path, args.model2, args.seed_type, ir_latent, og_latent, 'latent')
            # save_tensor_reps(path, args.model2, args.seed_type, ir_out, og_out, 'out')

if __name__=='__main__':
    main()