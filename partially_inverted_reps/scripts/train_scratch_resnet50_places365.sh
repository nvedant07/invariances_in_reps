#!/bin/bash
#
#SBATCH -p a100
#SBATCH --gres=gpu:2
#SBATCH -c 16                   # Number of cores
#SBATCH -N 1                    # Ensure that all cores are on one machine
#SBATCH --ntasks-per-node=1     
#SBATCH -t 4-00:00              # Maximum run-time in D-HH:MM
#SBATCH --mem=100GB               # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o /NS/robustness_2/work/vnanda/invariances_in_reps/deep-learning-base/checkpoints/sbatch_logs/%x_%j.out      # File to which STDOUT will be written
#SBATCH -e /NS/robustness_2/work/vnanda/invariances_in_reps/deep-learning-base/checkpoints/sbatch_logs/%x_%j.err      # File to which STDERR will be written


srun --jobid $SLURM_JOBID bash -c 'python -m partially_inverted_reps.finetuning \
--source_dataset places365 \
--finetuning_dataset places365 \
--finetune_mode full \
--base_dir partially_inverted_reps \
--save_every 0 \
--model resnet50 \
--pretrained False \
--batch_size 512 \
--append_path scratch \
--epochs 100 \
--optimizer sgd \
--lr 0.001 \
--step_lr 500 \
--warmup_steps 100 \
--gradient_clipping 1.0'
