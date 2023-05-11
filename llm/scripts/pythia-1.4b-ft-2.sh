#!/bin/bash
#
#SBATCH --partition=a100        # Use GPU partition "a100"
#SBATCH --gres=gpu:2            # set 2 GPUs per job
#SBATCH --nodelist=sws-8a100-04
#SBATCH -c 16                   # Number of cores
#SBATCH -N 1                    # Ensure that all cores are on one machine
#SBATCH --ntasks-per-node=1     # crucial - only 1 task per dist per node!
#SBATCH -t 2-00:00              # Maximum run-time in D-HH:MM
#SBATCH --mem=100GB               # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o /NS/llm-1/work/vnanda/llm_finetuning/pythia-1.4b/%x_%j.out      # File to which STDOUT will be written
#SBATCH -e /NS/llm-1/work/vnanda/llm_finetuning/pythia-1.4b/%x_%j.err      # File to which STDERR will be written

export GPUS_PER_NODE=2
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=9901

srun --jobid $SLURM_JOBID bash -c 'python -m torch.distributed.run \
 --nproc_per_node $GPUS_PER_NODE --nnodes $SLURM_NNODES --node_rank $SLURM_PROCID \
 --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
 finetuning_instructions.py \
 --input-model /NS/llm-1/nobackup/vnanda/llm_base_models/pythia-1.4b \
 --epochs 50 \
 --local-output-dir /NS/llm-1/nobackup/vnanda/llm_finetuning/pythia-1.4b-finetuning \
 --partial-dataset qa_only \
 --finetuning-ds databricks/databricks-dolly-15k \
 --wandb-project-name pythia-1.4b-dolly-finetuning \
 --ds-cache-dir /NS/twitter-9/work/vnanda/invariances_in_reps/llm/data \
 --logging-steps 10 \
 --per-device-train-batch-size 16 \
 --per-device-eval-batch-size 16 \
 --gradient-accumulation-steps 1 \
 --save-steps 1000 \
 --save-total-limit 20 \
 --eval-steps 50 \
 --warmup-steps 50 \
 --test-size 200 \
 --lr 5e-6 \
 --deepspeed dolly/config/ds_z3_bf16_config.json'
