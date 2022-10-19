python -m partially_inverted_reps.finetuning \
--source_dataset imagenet \
--finetuning_dataset cifar10 \
--finetune_mode full \
--base_dir effects_of_finetuning \
--save_every 0 \
--model resnet50 \
--batch_size 256 \
--append_path timm-weights \
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
--model resnet50 \
--batch_size 256 \
--append_path timm-weights \
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
--model resnet50 \
--batch_size 256 \
--append_path timm-weights \
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
--model resnet50 \
--batch_size 256 \
--append_path timm-weights \
--epochs 100 \
--optimizer sgd \
--lr 0.001 \
--step_lr 20

