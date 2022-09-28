from pytorch_lightning import utilities as pl_utils
from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning.plugins import DDPPlugin
import torch, glob
import itertools
import torch.nn as nn
import pathlib, argparse
from functools import partial
from collections import OrderedDict
import numpy as np
try:
    import plot_helper as plt_hp
    import output as out
    from training import LitProgressBar, NicerModelCheckpointing
    import training.finetuning as ft
    import architectures as arch
    from architectures.callbacks import AdvAttackWrapper
    from attack.callbacks import AdvCallback
    from datasets.data_modules import DATA_MODULES
    import datasets.dataset_metadata as dsmd
    from partially_inverted_reps.partial_loss import PartialInversionLoss, PartialInversionRegularizedLoss
    from partially_inverted_reps import DATA_PATH_IMAGENET, DATA_PATH, SERVER_PROJECT_PATH
    from training.partial_inference_layer import EnsembleHead, HardEnsembleLoss
except:
    raise ValueError('Run as a module to trigger __init__.py, ie '
                     'run as python -m partially_inverted_reps.ensemble')


parser = argparse.ArgumentParser(description='PyTorch Visual Explanation')
parser.add_argument('--base_dataset', type=str, default='cifar10')
parser.add_argument('--finetuning_dataset', type=str, default='cifar10')
parser.add_argument('--model', type=str, default='resnet18')
parser.add_argument('--batch_size', type=int, default=500)
parser.add_argument('--checkpoint_path', type=str, default='')
parser.add_argument('--append_path', type=str, 
    default='nonrob', choices=['nonrob', 'robustl2eps3'])
parser.add_argument('--ensemble_type', type=str, 
    default='soft', choices=['soft', 'hard'])
### specific to partial representation
parser.add_argument('--mode', type=str, default='random')


SEED = 2
NUM_NODES = 1
DEVICES = torch.cuda.device_count()
STRATEGY = DDPPlugin(find_unused_parameters=False) if DEVICES > 1 else None
BASE_DIR = pathlib.Path(__file__).parent.resolve()
MODEL_LOSSES = {
    'hard': HardEnsembleLoss(nn.CrossEntropyLoss(reduction='none')),
    'soft': None
}


def inference(pred, ensemble_type):
    if ensemble_type == 'soft':
        return torch.argmax(pred, 1)
    else:
        pred = torch.argmax(pred, -1)
        return torch.mode(pred, -1).values


def main(args=None):
    if args is None:
        args = parser.parse_args()

    dm = DATA_MODULES[args.finetuning_dataset](
        data_dir=DATA_PATH_IMAGENET if 'imagenet' in args.finetuning_dataset else DATA_PATH,
        transform_train=dsmd.TRAIN_TRANSFORMS_TRANSFER_DEFAULT(224),
        transform_test=dsmd.TEST_TRANSFORMS_DEFAULT(224),
        batch_size=args.batch_size)
    dm.init_remaining_attrs(args.base_dataset)

    m1 = arch.create_model(args.model, args.base_dataset, pretrained=True,
                           checkpoint_path=args.checkpoint_path, seed=SEED, 
                           callback=partial(AdvAttackWrapper,
                                            dataset_name=args.base_dataset,
                                            return_adv_samples=True,
                                            loss=MODEL_LOSSES[args.ensemble_type]
                                            )) ## assign mean and std from source dataset
    PARTIAL_FRACTIONS = sorted(
        list(set(
            [float(x.split('/frac-')[1].split('-')[0]) for x in \
                glob.glob(f'{BASE_DIR}/checkpoints/{args.model}-base-'
                          f'{args.base_dataset}-ft-{args.finetuning_dataset}/'
                          f'*-bs-256-{args.append_path}')]
            )))
    frac_to_layers = OrderedDict()
    for frac in PARTIAL_FRACTIONS:
        SEEDS = [int(x.split('seed-')[1].split('-')[0]) for x in glob.glob(
                          f'{BASE_DIR}/checkpoints/{args.model}-base-'
                          f'{args.base_dataset}-ft-{args.finetuning_dataset}/'
                          f'frac-{frac:.5f}-*-bs-256-{args.append_path}')]
        for seed in SEEDS:
            FINETUNED_CHECKPOINT = glob.glob(
                f'{BASE_DIR}/checkpoints/{args.model}-base-'
                f'{args.base_dataset}-ft-{args.finetuning_dataset}/'
                f'frac-{frac:.5f}-mode-{args.mode}-seed-{seed}-ftmode-'
                f'linear-lr-0.1-bs-256-{args.append_path}/*-topk=1.ckpt')
            if len(FINETUNED_CHECKPOINT) == 0:
                continue
            FINETUNED_CHECKPOINT = FINETUNED_CHECKPOINT[0]
            state_dict = torch.load(FINETUNED_CHECKPOINT)
            new_layer = ft.setup_model_for_finetuning(
                m1.model, 
                dsmd.DATASET_PARAMS[args.finetuning_dataset]['num_classes'],
                args.mode, frac, seed, inplace=False)
            new_layer.load_state_dict({'.'.join(k.split('.')[-2:]):v \
                                        for k,v in state_dict['state_dict'].items()}, strict=True)
            if hasattr(new_layer, 'neuron_indices') and 'neuron_indices' in state_dict:
                assert torch.all(new_layer.neuron_indices == state_dict['neuron_indices'])
            frac_to_layers[frac] = frac_to_layers[frac] + [new_layer] \
                if frac in frac_to_layers else [new_layer]

    pl_utils.seed.seed_everything(SEED, workers=True)
    adv_callback = AdvCallback(constraint_train='2',
                           eps_train=3.,
                           step_size=1.,
                           iterations_train=10,
                           iterations_test=50,
                           random_start_train=False,
                           random_restarts_train=0,
                           return_image=True, 
                           do_tqdm=False,
                           custom_loss=MODEL_LOSSES[args.ensemble_type])
    trainer = Trainer(accelerator='gpu',
                      devices=DEVICES,
                      num_nodes=NUM_NODES,
                      strategy=STRATEGY,
                      log_every_n_steps=1,
                      auto_select_gpus=True,
                      deterministic=True,
                      check_val_every_n_epoch=1,
                      num_sanity_val_steps=0,
                      sync_batchnorm=True,
                      callbacks=[
                        LitProgressBar(['loss',
                                        'running_acc_clean',
                                        'running_acc_adv']),
                        adv_callback])

    frac_to_clean_acc, frac_to_rob_acc = [], []
    name, _ = list(m1.model.named_modules())[-1]
    for frac in PARTIAL_FRACTIONS:
        print (frac)
        m1.model.__setattr__(name, EnsembleHead(frac_to_layers.get(frac), args.ensemble_type))
        predictions = trainer.predict(m1, dataloaders=[dm.test_dataloader()])
        clean_pred = inference(predictions[0][1], args.ensemble_type)
        adv_pred = inference(predictions[1][1], args.ensemble_type)
        y = predictions[2]
        frac_to_clean_acc.append(torch.sum(clean_pred == y)/len(y))
        frac_to_rob_acc.append(torch.sum(adv_pred == y)/len(y))
    
    if trainer.is_global_zero:
        plt_str = '<<TableOfContents()>>\n\n'\
            f'== Ensemble Predictions ==\n\n=== {args.finetuning_dataset} ===\n\n'
        baseline_clean, baseline_rob = frac_to_clean_acc[-1], frac_to_rob_acc[-1]
        x_vals = [PARTIAL_FRACTIONS[idx] for idx in range(len(frac_to_clean_acc[:-1]))]
        plt_str += '{}\n\n'.format(plt_hp.get_wiki_link(plt_hp.line_plot(
            [frac_to_clean_acc[:-1], frac_to_rob_acc[:-1]], 
            'Fraction of neurons', 'Ensemble Accuracy', args.finetuning_dataset, 
            subfolder=args.base_dataset, 
            filename=f'{args.model}_{args.finetuning_dataset}_'
                     f'{args.append_path}_{args.ensemble_type}', 
            extension='png', x_vals=x_vals, 
            legend_vals=['Clean Acc', 'Robust Acc ' + r'($\ell_2 \epsilon = 3$)', 
                'Clean (Full Layer)', 'Rob (Full Layer)'], 
            vertical_line=None, horizontal_lines=[baseline_clean, baseline_rob], 
            colors=[plt_hp.COLORS[0], plt_hp.COLORS[1], plt_hp.COLORS[0], plt_hp.COLORS[1]], 
            linestyles=['-', '-', ':', ':'], y_lims=(0.,1.), root_dir='./partially_inverted_reps', 
            paper_friendly_plots=False, plot_inside=False, legend_location='best', 
            savefig=True, figsize=(10,6), marker=[True, True, False, False], 
            results_subfolder_name='ensemble_analysis', grid_spacing=None, 
            legend_ncol=None), SERVER_PROJECT_PATH, size=700))
        with open(f'partially_inverted_reps/results/ensemble_analysis/{args.base_dataset}/'
                  f'wiki_results_{args.finetuning_dataset}-{args.append_path}-{args.ensemble_type}.txt', 
                  'w') as fp:
            fp.write(plt_str)
        out.upload_results(['partially_inverted_reps/{}/{}/{}'.format(
            plt_hp.RESULTS_FOLDER_NAME, 
            'ensemble_analysis', args.base_dataset)], 
            f'partially_inverted_reps/{plt_hp.RESULTS_FOLDER_NAME}', 
            SERVER_PROJECT_PATH, '.png')


if __name__ == '__main__':
    main()

