python -m partially_inverted_reps.ensemble \
--base_dataset imagenet \
--finetuning_dataset cifar10 \
--model resnet50 \
--batch_size 100 \
--append_path nonrob \
--ensemble_type soft

python -m partially_inverted_reps.ensemble \
--base_dataset imagenet \
--finetuning_dataset cifar10 \
--model resnet50 \
--batch_size 100 \
--append_path nonrob \
--ensemble_type hard

python -m partially_inverted_reps.ensemble \
--base_dataset imagenet \
--finetuning_dataset cifar100 \
--model resnet50 \
--batch_size 100 \
--append_path nonrob \
--ensemble_type soft

python -m partially_inverted_reps.ensemble \
--base_dataset imagenet \
--finetuning_dataset cifar100 \
--model resnet50 \
--batch_size 100 \
--append_path nonrob \
--ensemble_type hard

python -m partially_inverted_reps.ensemble \
--base_dataset imagenet \
--finetuning_dataset flowers \
--model resnet50 \
--batch_size 50 \
--append_path nonrob \
--ensemble_type soft

python -m partially_inverted_reps.ensemble \
--base_dataset imagenet \
--finetuning_dataset flowers \
--model resnet50 \
--batch_size 50 \
--append_path nonrob \
--ensemble_type hard

python -m partially_inverted_reps.ensemble \
--base_dataset imagenet \
--finetuning_dataset oxford-iiit-pets \
--model resnet50 \
--batch_size 50 \
--append_path nonrob \
--ensemble_type soft

python -m partially_inverted_reps.ensemble \
--base_dataset imagenet \
--finetuning_dataset oxford-iiit-pets \
--model resnet50 \
--batch_size 50 \
--append_path nonrob \
--ensemble_type hard