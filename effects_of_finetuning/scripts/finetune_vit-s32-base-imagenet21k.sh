CUDA_VISIBLE_DEVICES=6,7 python -m partially_inverted_reps.finetuning \
--source_dataset imagenet21k \
--finetuning_dataset cifar10 \
--finetune_mode full \
--base_dir effects_of_finetuning \
--save_every 0 \
--model vit_small_patch32_224 \
--batch_size 256 \
--checkpoint_path /NS/robustness_2/work/vnanda/invariances_in_reps/deep-learning-base/checkpoints/imagenet21k/S_32-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz \
--append_path aug_light1-wd_0.03-do_0.0-sd_0.0 \
--epochs 100 \
--optimizer sgd \
--lr 0.001 \
--step_lr 20 \
--warmup_steps 100 \
--gradient_clipping 1.0

CUDA_VISIBLE_DEVICES=4,5 python -m partially_inverted_reps.finetuning \
--source_dataset imagenet21k \
--finetuning_dataset cifar100 \
--finetune_mode full \
--base_dir effects_of_finetuning \
--save_every 0 \
--model vit_small_patch32_224 \
--batch_size 512 \
--checkpoint_path /NS/robustness_2/work/vnanda/invariances_in_reps/deep-learning-base/checkpoints/imagenet21k/S_32-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz \
--append_path aug_light1-wd_0.03-do_0.0-sd_0.0 \
--epochs 100 \
--lr 0.001 \
--step_lr 20 \
--warmup_steps 100 \
--gradient_clipping 1.0

