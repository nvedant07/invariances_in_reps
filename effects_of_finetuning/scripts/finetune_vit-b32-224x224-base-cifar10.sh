# base model is trained on cifar10
python -m partially_inverted_reps.finetuning \
--source_dataset cifar10 \
--finetuning_dataset cifar100 \
--finetune_mode full \
--base_dir effects_of_finetuning \
--save_every 0 \
--model vit_base_patch32_224 \
--batch_size 256 \
--use_timm_for_cifar True \
--pretrained True \
--checkpoint_path /NS/robustness_2/work/vnanda/invariances_in_reps/effects_of_finetuning/checkpoints/vit_base_patch32_224-base-cifar10-ft-cifar10/ftmode-full-lr-0.001-steplr-500.0-bs-256-scratch-warmup-440/epoch=83-topk=1.ckpt \
--append_path lr-0.001-steplr-500.0-bs-256-scratch-warmup-440 \
--epochs 100 \
--optimizer sgd \
--lr 0.001 \
--step_lr 200 \
--warmup_steps 100 \
--gradient_clipping 1.0
