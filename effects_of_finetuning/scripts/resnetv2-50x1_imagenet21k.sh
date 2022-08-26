python -m effects_of_finetuning.off_the_shelf_vits \
--eval_dataset cifar100 \
--base_dataset imagenet21k \
--finetuning_dataset cifar100 \
--model resnetv2_50x1_bitm \
--batch_size 10 \
--total_imgs 100 \
--base_checkpoint /NS/robustness_2/work/vnanda/invariances_in_reps/deep-learning-base/checkpoints/imagenet21k/BiT-M-R50x1.npz \
--finetuned_checkpoint /NS/robustness_2/work/vnanda/invariances_in_reps/deep-learning-base/checkpoints/imagenet21k/BiT-M-R50x1-run0-cifar100.npz

python -m effects_of_finetuning.off_the_shelf_vits \
--eval_dataset imagenet \
--base_dataset imagenet21k \
--finetuning_dataset imagenet \
--model resnetv2_50x1_bitm \
--batch_size 10 \
--total_imgs 100 \
--base_checkpoint /NS/robustness_2/work/vnanda/invariances_in_reps/deep-learning-base/checkpoints/imagenet21k/BiT-M-R50x1.npz \
--finetuned_checkpoint /NS/robustness_2/work/vnanda/invariances_in_reps/deep-learning-base/checkpoints/imagenet21k/BiT-M-R50x1-ILSVRC2012.npz

python -m effects_of_finetuning.off_the_shelf_vits \
--eval_dataset oxford-iiit-pets \
--base_dataset imagenet21k \
--finetuning_dataset oxford-iiit-pets \
--model resnetv2_50x1_bitm \
--batch_size 10 \
--total_imgs 100 \
--base_checkpoint /NS/robustness_2/work/vnanda/invariances_in_reps/deep-learning-base/checkpoints/imagenet21k/BiT-M-R50x1.npz \
--finetuned_checkpoint /NS/robustness_2/work/vnanda/invariances_in_reps/deep-learning-base/checkpoints/imagenet21k/BiT-M-R50x1-run0-oxford_iiit_pet.npz