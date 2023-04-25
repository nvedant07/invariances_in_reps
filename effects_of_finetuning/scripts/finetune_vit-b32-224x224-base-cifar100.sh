# base model is trained on cifar100
python -m partially_inverted_reps.finetuning \
--source_dataset cifar100 \
--finetuning_dataset cifar10 \
--finetune_mode full \
--base_dir effects_of_finetuning \
--save_every 0 \
--model vit_base_patch32_224 \
--batch_size 256 \
--use_timm_for_cifar True \
--pretrained True \
--checkpoint_path /NS/robustness_2/work/vnanda/invariances_in_reps/effects_of_finetuning/checkpoints/vit_base_patch32_224-base-cifar100-ft-cifar100/ftmode-full-lr-0.001-steplr-500.0-bs-128-scratch-warmup-880/epoch=92-topk=1.ckpt \
--append_path lr-0.001-steplr-500.0-bs-128-scratch-warmup-880 \
--epochs 100 \
--optimizer sgd \
--lr 0.001 \
--step_lr 200 \
--warmup_steps 100 \
--gradient_clipping 1.0
