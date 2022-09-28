for seed in {1..5}
do
python -m partially_inverted_reps.finetuning_all_features \
--source_dataset imagenet \
--finetuning_dataset oxford-iiit-pets \
--finetune_mode linear \
--model resnet50 \
--batch_size 256 \
--append_path nonrob \
--epochs 50 \
--step_lr 10 \
--save_every 0 \
--mode random \
--fraction 0.000001 \
--seed $seed

python -m partially_inverted_reps.finetuning_all_features \
--source_dataset imagenet \
--finetuning_dataset oxford-iiit-pets \
--finetune_mode linear \
--model resnet50 \
--batch_size 256 \
--append_path nonrob \
--epochs 50 \
--step_lr 10 \
--save_every 0 \
--mode random \
--fraction 0.00001 \
--seed $seed

python -m partially_inverted_reps.finetuning_all_features \
--source_dataset imagenet \
--finetuning_dataset oxford-iiit-pets \
--finetune_mode linear \
--model resnet50 \
--batch_size 256 \
--append_path nonrob \
--epochs 50 \
--step_lr 10 \
--save_every 0 \
--mode random \
--fraction 0.0001 \
--seed $seed


python -m partially_inverted_reps.finetuning_all_features \
--source_dataset imagenet \
--finetuning_dataset oxford-iiit-pets \
--finetune_mode linear \
--model resnet50 \
--batch_size 256 \
--append_path nonrob \
--epochs 50 \
--step_lr 10 \
--save_every 0 \
--mode random \
--fraction 0.0005 \
--seed $seed

python -m partially_inverted_reps.finetuning_all_features \
--source_dataset imagenet \
--finetuning_dataset oxford-iiit-pets \
--finetune_mode linear \
--model resnet50 \
--batch_size 256 \
--append_path nonrob \
--epochs 50 \
--step_lr 10 \
--save_every 0 \
--mode random \
--fraction 0.005 \
--seed $seed

python -m partially_inverted_reps.finetuning_all_features \
--source_dataset imagenet \
--finetuning_dataset oxford-iiit-pets \
--finetune_mode linear \
--model resnet50 \
--batch_size 256 \
--append_path nonrob \
--epochs 50 \
--step_lr 10 \
--save_every 0 \
--mode random \
--fraction 0.05 \
--seed $seed

python -m partially_inverted_reps.finetuning_all_features \
--source_dataset imagenet \
--finetuning_dataset oxford-iiit-pets \
--finetune_mode linear \
--model resnet50 \
--batch_size 256 \
--append_path nonrob \
--epochs 50 \
--step_lr 10 \
--save_every 0 \
--mode random \
--fraction 0.1 \
--seed $seed

python -m partially_inverted_reps.finetuning_all_features \
--source_dataset imagenet \
--finetuning_dataset oxford-iiit-pets \
--finetune_mode linear \
--model resnet50 \
--batch_size 256 \
--append_path nonrob \
--epochs 50 \
--step_lr 10 \
--save_every 0 \
--mode random \
--fraction 0.2 \
--seed $seed
done

