# base
python -m effects_of_finetuning.within_model_comparison \
--eval_dataset cifar100 \
--base_dataset imagenet \
--model vit_base_patch32_224 \
--batch_size 250 \
--total_imgs 500 \
--checkpoint /NS/robustness_2/work/vnanda/invariances_in_reps/deep-learning-base/checkpoints/imagenet1k/B_32-i1k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz \
--append_path base

# base
python -m effects_of_finetuning.within_model_comparison \
--eval_dataset cifar10 \
--base_dataset imagenet \
--model vit_base_patch32_224 \
--batch_size 250 \
--total_imgs 500 \
--checkpoint /NS/robustness_2/work/vnanda/invariances_in_reps/deep-learning-base/checkpoints/imagenet1k/B_32-i1k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz \
--append_path base
