## Full Dolly Finetuning
deepspeed --num_gpus=2 --num_nodes=1 finetuning_instructions.py \
--input-model /NS/llm-1/nobackup/vnanda/llm_base_models/pythia-2.8b \
--epochs 50 \
--local-output-dir /NS/llm-1/nobackup/vnanda/llm_finetuning/pythia-2.8b-finetuning \
--finetuning-ds databricks/databricks-dolly-15k \
--wandb-project-name pythia-2.8b-dolly-finetuning \
--ds-cache-dir /NS/twitter-9/work/vnanda/invariances_in_reps/llm/data \
--logging-steps 10 \
--per-device-train-batch-size 2 \
--per-device-eval-batch-size 2 \
--gradient-accumulation-steps 8 \
--save-steps 1000 \
--save-total-limit 20 \
--eval-steps 50 \
--warmup-steps 50 \
--test-size 200 \
--lr 5e-6 \
--deepspeed dolly/config/ds_z3_bf16_config.json

deepspeed --num_gpus=2 --num_nodes=1 finetuning_instructions.py \
--input-model /NS/llm-1/nobackup/vnanda/llm_base_models/pythia-2.8b \
--epochs 50 \
--local-output-dir /NS/llm-1/nobackup/vnanda/llm_finetuning/pythia-2.8b-finetuning \
--finetuning-ds databricks/databricks-dolly-15k \
--wandb-project-name pythia-2.8b-dolly-finetuning \
--ds-cache-dir /NS/twitter-9/work/vnanda/invariances_in_reps/llm/data \
--logging-steps 10 \
--per-device-train-batch-size 4 \
--per-device-eval-batch-size 4 \
--save-steps 1000 \
--save-total-limit 20 \
--eval-steps 50 \
--warmup-steps 50 \
--test-size 200 \
--lr 5e-6 \
--deepspeed dolly/config/ds_z3_bf16_config.json \
--wandb-run-name robust-river-9






deepspeed --num_gpus=2 --num_nodes=1 finetuning_instructions.py \
--input-model /NS/llm-1/nobackup/vnanda/llm_base_models/pythia-2.8b \
--epochs 50 \
--local-output-dir /NS/llm-1/nobackup/vnanda/llm_finetuning/pythia-2.8b-finetuning \
--partial-dataset qa_only \
--finetuning-ds databricks/databricks-dolly-15k \
--wandb-project-name pythia-2.8b-dolly-finetuning \
--ds-cache-dir /NS/twitter-9/work/vnanda/invariances_in_reps/llm/data \
--logging-steps 10 \
--per-device-train-batch-size 3 \
--per-device-eval-batch-size 3 \
--gradient-accumulation-steps 8 \
--save-steps 1000 \
--save-total-limit 20 \
--eval-steps 50 \
--warmup-steps 50 \
--test-size 200 \
--lr 5e-6 \
--deepspeed dolly/config/ds_z3_bf16_config.json


deepspeed --num_gpus=2 --num_nodes=1 finetuning_instructions.py \
--input-model /NS/llm-1/nobackup/vnanda/llm_base_models/pythia-2.8b \
--epochs 50 \
--local-output-dir /NS/llm-1/nobackup/vnanda/llm_finetuning/pythia-2.8b-finetuning \
--partial-dataset qa_only \
--finetuning-ds databricks/databricks-dolly-15k \
--wandb-project-name pythia-2.8b-dolly-finetuning \
--ds-cache-dir /NS/twitter-9/work/vnanda/invariances_in_reps/llm/data \
--logging-steps 10 \
--per-device-train-batch-size 4 \
--per-device-eval-batch-size 4 \
--save-steps 1000 \
--save-total-limit 20 \
--eval-steps 50 \
--warmup-steps 50 \
--test-size 200 \
--lr 5e-6 \
--deepspeed dolly/config/ds_z3_bf16_config.json






deepspeed --num_gpus=2 --num_nodes=1 finetuning_instructions.py \
--input-model /NS/llm-1/nobackup/vnanda/llm_base_models/pythia-2.8b \
--epochs 50 \
--local-output-dir /NS/llm-1/nobackup/vnanda/llm_finetuning/pythia-2.8b-finetuning \
--partial-dataset context_only \
--finetuning-ds databricks/databricks-dolly-15k \
--wandb-project-name pythia-2.8b-dolly-finetuning \
--ds-cache-dir /NS/twitter-9/work/vnanda/invariances_in_reps/llm/data \
--logging-steps 10 \
--per-device-train-batch-size 3 \
--per-device-eval-batch-size 3 \
--gradient-accumulation-steps 8 \
--save-steps 1000 \
--save-total-limit 20 \
--eval-steps 50 \
--warmup-steps 50 \
--test-size 200 \
--lr 5e-6 \
--deepspeed dolly/config/ds_z3_bf16_config.json

deepspeed --num_gpus=2 --num_nodes=1 finetuning_instructions.py \
--input-model /NS/llm-1/nobackup/vnanda/llm_base_models/pythia-2.8b \
--epochs 50 \
--local-output-dir /NS/llm-1/nobackup/vnanda/llm_finetuning/pythia-2.8b-finetuning \
--partial-dataset context_only \
--finetuning-ds databricks/databricks-dolly-15k \
--wandb-project-name pythia-2.8b-dolly-finetuning \
--ds-cache-dir /NS/twitter-9/work/vnanda/invariances_in_reps/llm/data \
--logging-steps 10 \
--per-device-train-batch-size 4 \
--per-device-eval-batch-size 4 \
--save-steps 1000 \
--save-total-limit 20 \
--eval-steps 50 \
--warmup-steps 50 \
--test-size 200 \
--lr 5e-6 \
--deepspeed dolly/config/ds_z3_bf16_config.json






deepspeed --num_gpus=2 --num_nodes=1 finetuning_instructions.py \
--input-model /NS/llm-1/nobackup/vnanda/llm_base_models/pythia-2.8b \
--epochs 50 \
--local-output-dir /NS/llm-1/nobackup/vnanda/llm_finetuning/pythia-2.8b-finetuning \
--finetuning-ds databricks/databricks-dolly-15k \
--partial-dataset creative_writing \
--wandb-project-name pythia-2.8b-dolly-finetuning \
--ds-cache-dir /NS/twitter-9/work/vnanda/invariances_in_reps/llm/data \
--logging-steps 10 \
--per-device-train-batch-size 3 \
--per-device-eval-batch-size 3 \
--gradient-accumulation-steps 8 \
--save-steps 1000 \
--save-total-limit 20 \
--eval-steps 50 \
--warmup-steps 50 \
--test-size 200 \
--lr 5e-6 \
--deepspeed dolly/config/ds_z3_bf16_config.json

deepspeed --num_gpus=2 --num_nodes=1 finetuning_instructions.py \
--input-model /NS/llm-1/nobackup/vnanda/llm_base_models/pythia-2.8b \
--epochs 50 \
--local-output-dir /NS/llm-1/nobackup/vnanda/llm_finetuning/pythia-2.8b-finetuning \
--finetuning-ds databricks/databricks-dolly-15k \
--partial-dataset creative_writing \
--wandb-project-name pythia-2.8b-dolly-finetuning \
--ds-cache-dir /NS/twitter-9/work/vnanda/invariances_in_reps/llm/data \
--logging-steps 10 \
--per-device-train-batch-size 4 \
--per-device-eval-batch-size 4 \
--save-steps 1000 \
--save-total-limit 20 \
--eval-steps 50 \
--warmup-steps 50 \
--test-size 200 \
--lr 5e-6 \
--deepspeed dolly/config/ds_z3_bf16_config.json





deepspeed --num_gpus=2 --num_nodes=1 finetuning_instructions.py \
--input-model /NS/llm-1/nobackup/vnanda/llm_base_models/pythia-2.8b \
--epochs 50 \
--local-output-dir /NS/llm-1/nobackup/vnanda/llm_finetuning/pythia-2.8b-finetuning \
--finetuning-ds databricks/databricks-dolly-15k \
--partial-dataset summarization \
--wandb-project-name pythia-2.8b-dolly-finetuning \
--ds-cache-dir /NS/twitter-9/work/vnanda/invariances_in_reps/llm/data \
--logging-steps 10 \
--per-device-train-batch-size 3 \
--per-device-eval-batch-size 3 \
--gradient-accumulation-steps 8 \
--save-steps 1000 \
--save-total-limit 20 \
--eval-steps 50 \
--warmup-steps 50 \
--test-size 200 \
--lr 5e-6 \
--deepspeed dolly/config/ds_z3_bf16_config.json

deepspeed --num_gpus=2 --num_nodes=1 finetuning_instructions.py \
--input-model /NS/llm-1/nobackup/vnanda/llm_base_models/pythia-2.8b \
--epochs 50 \
--local-output-dir /NS/llm-1/nobackup/vnanda/llm_finetuning/pythia-2.8b-finetuning \
--finetuning-ds databricks/databricks-dolly-15k \
--partial-dataset summarization \
--wandb-project-name pythia-2.8b-dolly-finetuning \
--ds-cache-dir /NS/twitter-9/work/vnanda/invariances_in_reps/llm/data \
--logging-steps 10 \
--per-device-train-batch-size 4 \
--per-device-eval-batch-size 4 \
--save-steps 1000 \
--save-total-limit 20 \
--eval-steps 50 \
--warmup-steps 50 \
--test-size 200 \
--lr 5e-6 \
--deepspeed dolly/config/ds_z3_bf16_config.json
