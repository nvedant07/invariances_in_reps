python -m partially_inverted_reps.finetuning \
--source_dataset imagenet \
--finetuning_dataset cifar10 \
--finetune_mode full \
--base_dir effects_of_finetuning \
--save_every 0 \
--model vit_base_patch32_224 \
--batch_size 256 \
--checkpoint_path /NS/robustness_2/work/vnanda/invariances_in_reps/deep-learning-base/checkpoints/imagenet1k/B_32-i1k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz \
--append_path aug_light1-wd_0.03-do_0.0-sd_0.0 \
--epochs 100 \
--optimizer sgd \
--lr 0.001 \
--step_lr 20 \
--warmup_steps 100 \
--gradient_clipping 1.0


python -m partially_inverted_reps.finetuning \
--source_dataset imagenet \
--finetuning_dataset cifar10 \
--finetune_mode linear \
--base_dir effects_of_finetuning \
--save_every 0 \
--model vit_base_patch32_224 \
--batch_size 256 \
--checkpoint_path /NS/robustness_2/work/vnanda/invariances_in_reps/deep-learning-base/checkpoints/imagenet1k/B_32-i1k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz \
--append_path aug_light1-wd_0.03-do_0.0-sd_0.0 \
--epochs 100 \
--optimizer sgd \
--lr 0.001 \
--step_lr 20


python -m partially_inverted_reps.finetuning \
--source_dataset imagenet \
--finetuning_dataset cifar100 \
--finetune_mode full \
--base_dir effects_of_finetuning \
--save_every 0 \
--model vit_base_patch32_224 \
--batch_size 256 \
--checkpoint_path /NS/robustness_2/work/vnanda/invariances_in_reps/deep-learning-base/checkpoints/imagenet1k/B_32-i1k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz \
--append_path aug_light1-wd_0.03-do_0.0-sd_0.0 \
--epochs 100 \
--optimizer sgd \
--lr 0.001 \
--step_lr 20 \
--warmup_steps 100 \
--gradient_clipping 1.0


python -m partially_inverted_reps.finetuning \
--source_dataset imagenet \
--finetuning_dataset cifar100 \
--finetune_mode linear \
--base_dir effects_of_finetuning \
--save_every 0 \
--model vit_base_patch32_224 \
--batch_size 256 \
--checkpoint_path /NS/robustness_2/work/vnanda/invariances_in_reps/deep-learning-base/checkpoints/imagenet1k/B_32-i1k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz \
--append_path aug_light1-wd_0.03-do_0.0-sd_0.0 \
--epochs 100 \
--optimizer sgd \
--lr 0.001 \
--step_lr 20
