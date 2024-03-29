from pytorch_lightning import utilities as pl_utils
from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning.plugins import DDPPlugin
from typing import Optional, Union
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
    from partially_inverted_reps.partial_loss import PartialInversionLoss, PartialInversionRegularizedLoss
    from architectures.inference import inference_with_features
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
parser.add_argument('--step_size', type=float, default=1.)
parser.add_argument('--seed_type', type=str, default='super-noise')
parser.add_argument('--iters', type=int, default=None)
### specific to partial inversion
parser.add_argument('--type', type=str, default='reg_free')
parser.add_argument('--mode', type=str, default='random')
parser.add_argument('--fraction', type=float, default=0.5)
parser.add_argument('--seed', type=int, default=2)


class FeatureAttackWrapper(InvertedRepWrapper):

    def __init__(self, model: torch.nn.Module, seed: Union[torch.Tensor, str], 
                       layer: Optional[int] = None, *args, **kwargs) -> None:
        super().__init__(model, seed, layer, *args, **kwargs)

    def construct_seed(self, x):
        if isinstance(self.seed, str):
            ## add a small noise to original images to start the optimization
            return x + torch.randn(*x.shape, device=x.device) * 1e-10
        else:
            assert x.shape == self.seed.shape
            return self.seed

    def forward(self, x, **kwargs):
        seed_images = self.construct_seed(x)
        target_rep = inference_with_features(self.model, 
            self.normalizer(x), **self.inference_kwargs)[1].detach()
        seed_rep = inference_with_features(self.model, 
            self.normalizer(seed_images), **self.inference_kwargs)[1].detach()
        print ('Target Rep and Seed Rep norm: '\
               f'{torch.mean(torch.linalg.norm(target_rep - seed_rep, axis=1))}')
        ir, _ = self.attacker(seed_images, target_rep, **kwargs)
        return x, ir


DATA_PATH = '/NS/twitter_archive/work/vnanda/data'
SEED = 2
NUM_NODES = 1
DEVICES = 1
STRATEGY = DDPPlugin(find_unused_parameters=True) if DEVICES > 1 else None

def main(args=None):
    if args is None:
        args = parser.parse_args()

    dm = DATA_MODULES[args.dataset](
        data_dir=DATA_PATH,
        val_frac=0.,
        subset=100,
        transform_train=DATASET_PARAMS[args.source_dataset]['transform_train'],
        transform_test=DATASET_PARAMS[args.source_dataset]['transform_test'],
        batch_size=args.batch_size)
    dm.init_remaining_attrs(args.source_dataset)

    m1 = arch.create_model(args.model, args.dataset, pretrained=True,
                           checkpoint_path=args.checkpoint_path, seed=SEED, 
                           callback=partial(FeatureAttackWrapper, 
                                            seed=args.seed_type,
                                            dataset_name=args.source_dataset))

    ## TODO: make layer an actual number
    loss_obj = PartialInversionLoss if args.type == 'reg_free' else \
        PartialInversionRegularizedLoss
    custom_loss = loss_obj(lpnorm_type=2, layer=-1, 
        fraction=args.fraction, mode=args.mode, seed=args.seed)
    custom_loss._set_normalizer(m1.normalizer)

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
                               use_best=True,
                               do_tqdm=True,
                               should_normalize=False, # normalizer is implemented in losses
                               custom_loss=custom_loss)

    pl_utils.seed.seed_everything(args.seed, workers=True)

    trainer = Trainer(accelerator='gpu', 
                      devices=DEVICES,
                      num_nodes=NUM_NODES,
                      strategy=STRATEGY, 
                      log_every_n_steps=1,
                      auto_select_gpus=True, 
                      deterministic=True,
                      max_epochs=1,
                      check_val_every_n_epoch=1,
                      num_sanity_val_steps=0,
                      callbacks=[LitProgressBar(['loss']), 
                                adv_callback])

    out = trainer.predict(m1, dataloaders=[dm.val_dataloader()])
    if trainer.is_global_zero:
        ## do things on the main process
        og, ir, labels = out
        path = f'{pathlib.Path(__file__).parent.resolve()}/results/generated_images/{args.source_dataset}/'\
            f'{args.dataset}_{args.model}_{args.mode}_{args.fraction:.5f}_seed_{args.seed}_featureattack'
        if args.append_path:
            path += f'_{args.append_path}'
        save_tensor_images(path, torch.arange(len(og)), args.seed_type, 
            ir, None, og, labels, get_classes_names(args.dataset, DATA_PATH))

if __name__=='__main__':
    main()