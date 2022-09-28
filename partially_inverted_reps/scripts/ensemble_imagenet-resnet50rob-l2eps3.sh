python -m partially_inverted_reps.ensemble \
--base_dataset imagenet \
--finetuning_dataset cifar10 \
--model resnet50 \
--batch_size 100 \
--checkpoint_path /NS/robustness_2/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/resnet-50-l2-eps3.ckpt \
--append_path robustl2eps3 \
--ensemble_type hard

python -m partially_inverted_reps.ensemble \
--base_dataset imagenet \
--finetuning_dataset cifar10 \
--model resnet50 \
--batch_size 100 \
--checkpoint_path /NS/robustness_2/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/resnet-50-l2-eps3.ckpt \
--append_path robustl2eps3 \
--ensemble_type soft

python -m partially_inverted_reps.ensemble \
--base_dataset imagenet \
--finetuning_dataset cifar100 \
--model resnet50 \
--batch_size 100 \
--checkpoint_path /NS/robustness_2/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/resnet-50-l2-eps3.ckpt \
--append_path robustl2eps3 \
--ensemble_type hard

python -m partially_inverted_reps.ensemble \
--base_dataset imagenet \
--finetuning_dataset cifar100 \
--model resnet50 \
--batch_size 100 \
--checkpoint_path /NS/robustness_2/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/resnet-50-l2-eps3.ckpt \
--append_path robustl2eps3 \
--ensemble_type soft

python -m partially_inverted_reps.ensemble \
--base_dataset imagenet \
--finetuning_dataset flowers \
--model resnet50 \
--batch_size 100 \
--checkpoint_path /NS/robustness_2/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/resnet-50-l2-eps3.ckpt \
--append_path robustl2eps3 \
--ensemble_type hard

python -m partially_inverted_reps.ensemble \
--base_dataset imagenet \
--finetuning_dataset flowers \
--model resnet50 \
--batch_size 100 \
--checkpoint_path /NS/robustness_2/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/resnet-50-l2-eps3.ckpt \
--append_path robustl2eps3 \
--ensemble_type soft

python -m partially_inverted_reps.ensemble \
--base_dataset imagenet \
--finetuning_dataset oxford-iiit-pets \
--model resnet50 \
--batch_size 100 \
--checkpoint_path /NS/robustness_2/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/resnet-50-l2-eps3.ckpt \
--append_path robustl2eps3 \
--ensemble_type hard

python -m partially_inverted_reps.ensemble \
--base_dataset imagenet \
--finetuning_dataset oxford-iiit-pets \
--model resnet50 \
--batch_size 100 \
--checkpoint_path /NS/robustness_2/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/resnet-50-l2-eps3.ckpt \
--append_path robustl2eps3 \
--ensemble_type soft