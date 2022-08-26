python -m effects_of_finetuning.off_the_shelf_vits \
--eval_dataset cifar100 \
--base_dataset imagenet21k \
--finetuning_dataset cifar100 \
--model vit_base_patch16_384 \
--batch_size 10 \
--total_imgs 100 \
--base_checkpoint /NS/robustness_2/work/vnanda/invariances_in_reps/deep-learning-base/checkpoints/imagenet21k/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz \
--finetuned_checkpoint /NS/robustness_2/work/vnanda/invariances_in_reps/deep-learning-base/checkpoints/imagenet21k/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--cifar100-steps_10k-lr_0.001-res_384.npz

python -m effects_of_finetuning.off_the_shelf_vits \
--eval_dataset imagenet \
--base_dataset imagenet21k \
--finetuning_dataset imagenet \
--model vit_base_patch16_384 \
--batch_size 10 \
--total_imgs 100 \
--base_checkpoint /NS/robustness_2/work/vnanda/invariances_in_reps/deep-learning-base/checkpoints/imagenet21k/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz \
--finetuned_checkpoint /NS/robustness_2/work/vnanda/invariances_in_reps/deep-learning-base/checkpoints/imagenet21k/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_384.npz

python -m effects_of_finetuning.off_the_shelf_vits \
--eval_dataset oxford-iiit-pets \
--base_dataset imagenet21k \
--finetuning_dataset oxford-iiit-pets \
--model vit_base_patch16_384 \
--batch_size 10 \
--total_imgs 100 \
--base_checkpoint /NS/robustness_2/work/vnanda/invariances_in_reps/deep-learning-base/checkpoints/imagenet21k/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz \
--finetuned_checkpoint /NS/robustness_2/work/vnanda/invariances_in_reps/deep-learning-base/checkpoints/imagenet21k/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--oxford_iiit_pet-steps_0k-lr_0.001-res_384.npz
