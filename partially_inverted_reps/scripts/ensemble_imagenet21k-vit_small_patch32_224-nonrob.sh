python -m partially_inverted_reps.ensemble \
--base_dataset imagenet21k \
--finetuning_dataset cifar10 \
--model vit_small_patch32_224 \
--batch_size 100 \
--append_path nonrob \
--ensemble_type soft

python -m partially_inverted_reps.ensemble \
--base_dataset imagenet21k \
--finetuning_dataset cifar10 \
--model vit_small_patch32_224 \
--batch_size 100 \
--append_path nonrob \
--ensemble_type hard

python -m partially_inverted_reps.ensemble \
--base_dataset imagenet21k \
--finetuning_dataset cifar100 \
--model vit_small_patch32_224 \
--batch_size 100 \
--append_path nonrob \
--ensemble_type soft

python -m partially_inverted_reps.ensemble \
--base_dataset imagenet21k \
--finetuning_dataset cifar100 \
--model vit_small_patch32_224 \
--batch_size 100 \
--append_path nonrob \
--ensemble_type hard

python -m partially_inverted_reps.ensemble \
--base_dataset imagenet21k \
--finetuning_dataset flowers \
--model vit_small_patch32_224 \
--batch_size 50 \
--append_path nonrob \
--ensemble_type soft

python -m partially_inverted_reps.ensemble \
--base_dataset imagenet21k \
--finetuning_dataset flowers \
--model vit_small_patch32_224 \
--batch_size 50 \
--append_path nonrob \
--ensemble_type hard

python -m partially_inverted_reps.ensemble \
--base_dataset imagenet21k \
--finetuning_dataset oxford-iiit-pets \
--model vit_small_patch32_224 \
--batch_size 50 \
--append_path nonrob \
--ensemble_type soft

python -m partially_inverted_reps.ensemble \
--base_dataset imagenet21k \
--finetuning_dataset oxford-iiit-pets \
--model vit_small_patch32_224 \
--batch_size 50 \
--append_path nonrob \
--ensemble_type hard