python -m partially_inverted_reps.finetuning \
--source_dataset imagenet21k-miil \
--finetuning_dataset cifar10 \
--finetune_mode full \
--base_dir effects_of_finetuning \
--save_every 0 \
--model resnet50 \
--batch_size 256 \
--checkpoint_path /NS/robustness_2/work/vnanda/invariances_in_reps/deep-learning-base/checkpoints/imagenet21k-miil/resnet50_miil_21k.pth \
--append_path miil-weights \
--epochs 100 \
--optimizer sgd \
--lr 0.001 \
--step_lr 20 \
--warmup_steps 100 \
--gradient_clipping 1.0


python -m partially_inverted_reps.finetuning \
--source_dataset imagenet21k-miil \
--finetuning_dataset cifar10 \
--finetune_mode linear \
--base_dir effects_of_finetuning \
--save_every 0 \
--model resnet50 \
--batch_size 256 \
--checkpoint_path /NS/robustness_2/work/vnanda/invariances_in_reps/deep-learning-base/checkpoints/imagenet21k-miil/resnet50_miil_21k.pth \
--append_path miil-weights \
--epochs 100 \
--optimizer sgd \
--lr 0.001 \
--step_lr 20


python -m partially_inverted_reps.finetuning \
--source_dataset imagenet21k-miil \
--finetuning_dataset cifar100 \
--finetune_mode full \
--base_dir effects_of_finetuning \
--save_every 0 \
--model resnet50 \
--batch_size 256 \
--checkpoint_path /NS/robustness_2/work/vnanda/invariances_in_reps/deep-learning-base/checkpoints/imagenet21k-miil/resnet50_miil_21k.pth \
--append_path miil-weights \
--epochs 100 \
--optimizer sgd \
--lr 0.001 \
--step_lr 20 \
--warmup_steps 100 \
--gradient_clipping 1.0


python -m partially_inverted_reps.finetuning \
--source_dataset imagenet21k-miil \
--finetuning_dataset cifar100 \
--finetune_mode linear \
--base_dir effects_of_finetuning \
--save_every 0 \
--model resnet50 \
--batch_size 256 \
--checkpoint_path /NS/robustness_2/work/vnanda/invariances_in_reps/deep-learning-base/checkpoints/imagenet21k-miil/resnet50_miil_21k.pth \
--append_path miil-weights \
--epochs 100 \
--optimizer sgd \
--lr 0.001 \
--step_lr 20

