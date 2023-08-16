for seed in {1,}
do
python -m partially_inverted_reps.finetuning \
--source_dataset imagenet \
--finetuning_dataset cifar10 \
--finetune_mode linear \
--model resnet50 \
--batch_size 256 \
--append_path nonrob \
--epochs 50 \
--mode pca-least \
--pretrained True \
--step_lr 10 \
--fraction 0.0005 \
--seed $seed

python -m partially_inverted_reps.finetuning \
--source_dataset imagenet \
--finetuning_dataset cifar10 \
--finetune_mode linear \
--model resnet50 \
--batch_size 256 \
--append_path nonrob \
--epochs 50 \
--mode pca-least \
--pretrained True \
--step_lr 10 \
--fraction 0.001 \
--seed $seed

python -m partially_inverted_reps.finetuning \
--source_dataset imagenet \
--finetuning_dataset cifar10 \
--finetune_mode linear \
--model resnet50 \
--batch_size 256 \
--append_path nonrob \
--epochs 50 \
--mode pca-least \
--pretrained True \
--step_lr 10 \
--fraction 0.005 \
--seed $seed

python -m partially_inverted_reps.finetuning \
--source_dataset imagenet \
--finetuning_dataset cifar10 \
--finetune_mode linear \
--model resnet50 \
--batch_size 256 \
--append_path nonrob \
--epochs 50 \
--mode pca-least \
--pretrained True \
--step_lr 10 \
--fraction 0.01 \
--seed $seed

python -m partially_inverted_reps.finetuning \
--source_dataset imagenet \
--finetuning_dataset cifar10 \
--finetune_mode linear \
--model resnet50 \
--batch_size 256 \
--append_path nonrob \
--epochs 50 \
--mode pca-least \
--pretrained True \
--step_lr 10 \
--fraction 0.05 \
--seed $seed

python -m partially_inverted_reps.finetuning \
--source_dataset imagenet \
--finetuning_dataset cifar10 \
--finetune_mode linear \
--model resnet50 \
--batch_size 256 \
--append_path nonrob \
--epochs 50 \
--mode pca-least \
--pretrained True \
--step_lr 10 \
--fraction 0.1 \
--seed $seed

python -m partially_inverted_reps.finetuning \
--source_dataset imagenet \
--finetuning_dataset cifar10 \
--finetune_mode linear \
--model resnet50 \
--batch_size 256 \
--append_path nonrob \
--epochs 50 \
--mode pca-least \
--pretrained True \
--step_lr 10 \
--fraction 0.2 \
--seed $seed

python -m partially_inverted_reps.finetuning \
--source_dataset imagenet \
--finetuning_dataset cifar10 \
--finetune_mode linear \
--model resnet50 \
--batch_size 256 \
--append_path nonrob \
--epochs 50 \
--mode pca-least \
--pretrained True \
--step_lr 10 \
--fraction 0.3 \
--seed $seed

python -m partially_inverted_reps.finetuning \
--source_dataset imagenet \
--finetuning_dataset cifar10 \
--finetune_mode linear \
--model resnet50 \
--batch_size 256 \
--append_path nonrob \
--epochs 50 \
--mode pca-least \
--pretrained True \
--step_lr 10 \
--fraction 0.5 \
--seed $seed

python -m partially_inverted_reps.finetuning \
--source_dataset imagenet \
--finetuning_dataset cifar10 \
--finetune_mode linear \
--model resnet50 \
--batch_size 256 \
--append_path nonrob \
--epochs 50 \
--mode pca-least \
--pretrained True \
--step_lr 10 \
--fraction 0.8 \
--seed $seed

python -m partially_inverted_reps.finetuning \
--source_dataset imagenet \
--finetuning_dataset cifar10 \
--finetune_mode linear \
--model resnet50 \
--batch_size 256 \
--append_path nonrob \
--epochs 50 \
--mode pca-least \
--pretrained True \
--step_lr 10 \
--fraction 0.9 \
--seed $seed
done

