
for seed in {1..5}
do
python -m partially_inverted_reps.finetuning \
--source_dataset imagenet \
--finetuning_dataset imagenet \
--finetune_mode linear \
--model resnet18 \
--batch_size 512 \
--checkpoint_path /NS/robustness_3/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/resnet-18-l2-eps3.ckpt \
--append_path robustl2eps3 \
--epochs 25 \
--mode random \
--step_lr 5 \
--fraction 0.005 \
--seed $seed

python -m partially_inverted_reps.finetuning \
--source_dataset imagenet \
--finetuning_dataset imagenet \
--finetune_mode linear \
--model resnet18 \
--batch_size 512 \
--checkpoint_path /NS/robustness_3/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/resnet-18-l2-eps3.ckpt \
--append_path robustl2eps3 \
--epochs 25 \
--mode random \
--step_lr 5 \
--fraction 0.01 \
--seed $seed

python -m partially_inverted_reps.finetuning \
--source_dataset imagenet \
--finetuning_dataset imagenet \
--finetune_mode linear \
--model resnet18 \
--batch_size 512 \
--checkpoint_path /NS/robustness_3/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/resnet-18-l2-eps3.ckpt \
--append_path robustl2eps3 \
--epochs 25 \
--mode random \
--step_lr 5 \
--fraction 0.05 \
--seed $seed

python -m partially_inverted_reps.finetuning \
--source_dataset imagenet \
--finetuning_dataset imagenet \
--finetune_mode linear \
--model resnet18 \
--batch_size 512 \
--checkpoint_path /NS/robustness_3/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/resnet-18-l2-eps3.ckpt \
--append_path robustl2eps3 \
--epochs 25 \
--mode random \
--step_lr 5 \
--fraction 0.1 \
--seed $seed

python -m partially_inverted_reps.finetuning \
--source_dataset imagenet \
--finetuning_dataset imagenet \
--finetune_mode linear \
--model resnet18 \
--batch_size 512 \
--checkpoint_path /NS/robustness_3/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/resnet-18-l2-eps3.ckpt \
--append_path robustl2eps3 \
--epochs 25 \
--mode random \
--step_lr 5 \
--fraction 0.2 \
--seed $seed

python -m partially_inverted_reps.finetuning \
--source_dataset imagenet \
--finetuning_dataset imagenet \
--finetune_mode linear \
--model resnet18 \
--batch_size 512 \
--checkpoint_path /NS/robustness_3/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/resnet-18-l2-eps3.ckpt \
--append_path robustl2eps3 \
--epochs 25 \
--mode random \
--step_lr 5 \
--fraction 0.3 \
--seed $seed

python -m partially_inverted_reps.finetuning \
--source_dataset imagenet \
--finetuning_dataset imagenet \
--finetune_mode linear \
--model resnet18 \
--batch_size 512 \
--checkpoint_path /NS/robustness_3/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/resnet-18-l2-eps3.ckpt \
--append_path robustl2eps3 \
--epochs 25 \
--mode random \
--step_lr 5 \
--fraction 0.5 \
--seed $seed

python -m partially_inverted_reps.finetuning \
--source_dataset imagenet \
--finetuning_dataset imagenet \
--finetune_mode linear \
--model resnet18 \
--batch_size 512 \
--checkpoint_path /NS/robustness_3/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/resnet-18-l2-eps3.ckpt \
--append_path robustl2eps3 \
--epochs 25 \
--mode random \
--step_lr 5 \
--fraction 0.8 \
--seed $seed

python -m partially_inverted_reps.finetuning \
--source_dataset imagenet \
--finetuning_dataset imagenet \
--finetune_mode linear \
--model resnet18 \
--batch_size 512 \
--checkpoint_path /NS/robustness_3/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/resnet-18-l2-eps3.ckpt \
--append_path robustl2eps3 \
--epochs 25 \
--mode random \
--step_lr 5 \
--fraction 0.9 \
--seed $seed
done