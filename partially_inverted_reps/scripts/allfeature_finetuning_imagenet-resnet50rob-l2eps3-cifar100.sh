for seed in {1..5}
do
python -m partially_inverted_reps.finetuning_all_features \
--source_dataset imagenet \
--finetuning_dataset cifar100 \
--finetune_mode linear \
--model resnet50 \
--batch_size 256 \
--checkpoint_path /NS/robustness_3/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/resnet-50-l2-eps3.ckpt \
--append_path robustl2eps3 \
--epochs 50 \
--step_lr 10 \
--save_every 0 \
--mode random \
--fraction 0.000001 \
--seed $seed

python -m partially_inverted_reps.finetuning_all_features \
--source_dataset imagenet \
--finetuning_dataset cifar100 \
--finetune_mode linear \
--model resnet50 \
--batch_size 256 \
--checkpoint_path /NS/robustness_3/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/resnet-50-l2-eps3.ckpt \
--append_path robustl2eps3 \
--epochs 50 \
--step_lr 10 \
--save_every 0 \
--mode random \
--fraction 0.00001 \
--seed $seed

python -m partially_inverted_reps.finetuning_all_features \
--source_dataset imagenet \
--finetuning_dataset cifar100 \
--finetune_mode linear \
--model resnet50 \
--batch_size 256 \
--checkpoint_path /NS/robustness_3/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/resnet-50-l2-eps3.ckpt \
--append_path robustl2eps3 \
--epochs 50 \
--step_lr 10 \
--save_every 0 \
--mode random \
--fraction 0.0001 \
--seed $seed


python -m partially_inverted_reps.finetuning_all_features \
--source_dataset imagenet \
--finetuning_dataset cifar100 \
--finetune_mode linear \
--model resnet50 \
--batch_size 256 \
--checkpoint_path /NS/robustness_3/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/resnet-50-l2-eps3.ckpt \
--append_path robustl2eps3 \
--epochs 50 \
--step_lr 10 \
--save_every 0 \
--mode random \
--fraction 0.0005 \
--seed $seed

python -m partially_inverted_reps.finetuning_all_features \
--source_dataset imagenet \
--finetuning_dataset cifar100 \
--finetune_mode linear \
--model resnet50 \
--batch_size 256 \
--checkpoint_path /NS/robustness_3/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/resnet-50-l2-eps3.ckpt \
--append_path robustl2eps3 \
--epochs 50 \
--step_lr 10 \
--save_every 0 \
--mode random \
--fraction 0.005 \
--seed $seed

python -m partially_inverted_reps.finetuning_all_features \
--source_dataset imagenet \
--finetuning_dataset cifar100 \
--finetune_mode linear \
--model resnet50 \
--batch_size 256 \
--checkpoint_path /NS/robustness_3/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/resnet-50-l2-eps3.ckpt \
--append_path robustl2eps3 \
--epochs 50 \
--step_lr 10 \
--save_every 0 \
--mode random \
--fraction 0.05 \
--seed $seed

python -m partially_inverted_reps.finetuning_all_features \
--source_dataset imagenet \
--finetuning_dataset cifar100 \
--finetune_mode linear \
--model resnet50 \
--batch_size 256 \
--checkpoint_path /NS/robustness_3/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/resnet-50-l2-eps3.ckpt \
--append_path robustl2eps3 \
--epochs 50 \
--step_lr 10 \
--save_every 0 \
--mode random \
--fraction 0.1 \
--seed $seed

python -m partially_inverted_reps.finetuning_all_features \
--source_dataset imagenet \
--finetuning_dataset cifar100 \
--finetune_mode linear \
--model resnet50 \
--batch_size 256 \
--checkpoint_path /NS/robustness_3/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/resnet-50-l2-eps3.ckpt \
--append_path robustl2eps3 \
--epochs 50 \
--step_lr 10 \
--save_every 0 \
--mode random \
--fraction 0.2 \
--seed $seed
done

