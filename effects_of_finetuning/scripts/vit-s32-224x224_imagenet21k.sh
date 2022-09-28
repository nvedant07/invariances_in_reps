python -m effects_of_finetuning.off_the_shelf_vits \
--eval_dataset cifar100 \
--base_dataset imagenet21k \
--finetuning_dataset cifar100 \
--model vit_small_patch32_224 \
--batch_size 100 \
--total_imgs 500 \
--input_size 224 \
--base_checkpoint /NS/robustness_2/work/vnanda/invariances_in_reps/deep-learning-base/checkpoints/imagenet21k/S_32-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz \
--finetuned_checkpoint /NS/robustness_2/work/vnanda/invariances_in_reps/effects_of_finetuning/checkpoints/vit_small_patch32_224-base-imagenet21k-ft-cifar100/ftmode-full-lr-0.001-bs-512-aug_light1-wd_0.03-do_0.0-sd_0.0-warmup-100/epoch=88-topk=1.ckpt


python -m effects_of_finetuning.off_the_shelf_vits \
--eval_dataset cifar10 \
--base_dataset imagenet21k \
--finetuning_dataset cifar10 \
--model vit_small_patch32_224 \
--batch_size 100 \
--total_imgs 500 \
--input_size 224 \
--base_checkpoint /NS/robustness_2/work/vnanda/invariances_in_reps/deep-learning-base/checkpoints/imagenet21k/S_32-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz \
--finetuned_checkpoint /NS/robustness_2/work/vnanda/invariances_in_reps/effects_of_finetuning/checkpoints/vit_small_patch32_224-base-imagenet21k-ft-cifar10/ftmode-full-lr-0.001-bs-256-aug_light1-wd_0.03-do_0.0-sd_0.0-warmup-100/epoch=86-topk=1.ckpt


