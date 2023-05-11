deepspeed --num_gpus=2 --num_nodes=1 finetuning_instructions.py \
--input-model /NS/llm-1/nobackup/vnanda/llm_base_models/pythia-12b \
--epochs 50 \
--local-output-dir /NS/llm-1/work/vnanda/llm_finetuning/pythia-12b-finetuning \
--partial-dataset qa_only \
--finetuning-ds databricks/databricks-dolly-15k \
--wandb-project-name pythia12b-dolly-finetuning \
--ds-cache-dir /NS/twitter-9/work/vnanda/invariances_in_reps/llm/data \
--logging-steps 10 \
--per-device-train-batch-size 1 \
--per-device-eval-batch-size 1 \
--gradient-accumulation-steps 8 \
--gradient-checkpointing \
--save-steps 200 \
--save-total-limit 20 \
--eval-steps 50 \
--warmup-steps 50 \
--test-size 200 \
--lr 5e-6 \
--deepspeed dolly/config/ds_z3_bf16_config.json


deepspeed --num_gpus=2 --num_nodes=1 finetuning_instructions.py \
--input-model /NS/twitter-9/work/vnanda/invariances_in_reps/llm/checkpoints/gpt-j-6b \
--epochs 50 \
--local-output-dir /NS/twitter-9/work/vnanda/invariances_in_reps/llm/checkpoints/gpt-j-6b-finetuning \
--partial-dataset qa_only \
--finetuning-ds databricks/databricks-dolly-15k \
--wandb-project-name gpt-j-6b-dolly-finetuning \
--ds-cache-dir /NS/twitter-9/work/vnanda/invariances_in_reps/llm/data \
--logging-steps 10 \
--per-device-train-batch-size 1 \
--per-device-eval-batch-size 1 \
--save-steps 200 \
--save-total-limit 20 \
--eval-steps 50 \
--warmup-steps 50 \
--test-size 200 \
--lr 5e-6 \
--deepspeed dolly/config/ds_z3_bf16_config.json




