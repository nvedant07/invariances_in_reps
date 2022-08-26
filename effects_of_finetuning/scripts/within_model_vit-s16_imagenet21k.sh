# base
python -m effects_of_finetuning.within_model_comparison \
--eval_dataset cifar100 \
--base_dataset imagenet21k \
--model vit_small_patch16_384 \
--batch_size 25 \
--total_imgs 100 \
--checkpoint /NS/robustness_2/work/vnanda/invariances_in_reps/deep-learning-base/checkpoints/imagenet21k/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz \
--append_path base

# finetuned
python -m effects_of_finetuning.within_model_comparison \
--eval_dataset cifar100 \
--base_dataset imagenet21k \
--model vit_small_patch16_384 \
--batch_size 25 \
--total_imgs 100 \
--checkpoint /NS/robustness_2/work/vnanda/invariances_in_reps/deep-learning-base/checkpoints/imagenet21k/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--cifar100-steps_10k-lr_0.001-res_384.npz \
--append_path finetuned

# base
python -m effects_of_finetuning.within_model_comparison \
--eval_dataset imagenet \
--base_dataset imagenet21k \
--model vit_small_patch16_384 \
--batch_size 25 \
--total_imgs 100 \
--checkpoint /NS/robustness_2/work/vnanda/invariances_in_reps/deep-learning-base/checkpoints/imagenet21k/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz \
--append_path base

# finetuned
python -m effects_of_finetuning.within_model_comparison \
--eval_dataset imagenet \
--base_dataset imagenet21k \
--model vit_small_patch16_384 \
--batch_size 25 \
--total_imgs 100 \
--checkpoint /NS/robustness_2/work/vnanda/invariances_in_reps/deep-learning-base/checkpoints/imagenet21k/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_384.npz \
--append_path finetuned

# base
python -m effects_of_finetuning.within_model_comparison \
--eval_dataset oxford-iiit-pets \
--base_dataset imagenet21k \
--model vit_small_patch16_384 \
--batch_size 25 \
--total_imgs 100 \
--checkpoint /NS/robustness_2/work/vnanda/invariances_in_reps/deep-learning-base/checkpoints/imagenet21k/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz \
--append_path base

# finetuned
python -m effects_of_finetuning.within_model_comparison \
--eval_dataset oxford-iiit-pets \
--base_dataset imagenet21k \
--model vit_small_patch16_384 \
--batch_size 25 \
--total_imgs 100 \
--checkpoint /NS/robustness_2/work/vnanda/invariances_in_reps/deep-learning-base/checkpoints/imagenet21k/S_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--oxford_iiit_pet-steps_0k-lr_0.001-res_384.npz \
--append_path finetuned