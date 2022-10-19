python -m effects_of_finetuning.off_the_shelf_vits \
--eval_dataset cifar100 \
--base_dataset imagenet \
--finetuning_dataset cifar100 \
--model resnet50 \
--batch_size 50 \
--total_imgs 500 \
--finetuned_checkpoint /NS/robustness_2/work/vnanda/invariances_in_reps/effects_of_finetuning/checkpoints/resnet50-base-imagenet-ft-cifar100/ftmode-full-lr-0.001-bs-256-timm-weights-warmup-100/epoch=90-topk=1.ckpt

python -m effects_of_finetuning.off_the_shelf_vits \
--eval_dataset cifar10 \
--base_dataset imagenet \
--finetuning_dataset cifar10 \
--model resnet50 \
--batch_size 50 \
--total_imgs 500 \
--finetuned_checkpoint /NS/robustness_2/work/vnanda/invariances_in_reps/effects_of_finetuning/checkpoints/resnet50-base-imagenet-ft-cifar10/ftmode-full-lr-0.001-bs-256-timm-weights-warmup-100/epoch=83-topk=1.ckpt
