python -m effects_of_finetuning.off_the_shelf_vits \
--eval_dataset cifar100 \
--base_dataset imagenet21k \
--finetuning_dataset cifar100 \
--model vit_base_patch32_224 \
--batch_size 250 \
--total_imgs 500 \
--base_checkpoint /NS/robustness_2/work/vnanda/invariances_in_reps/deep-learning-base/checkpoints/imagenet21k/B_32-i21k-300ep-lr_0.001-aug_light1-wd_0.1-do_0.0-sd_0.0.npz \
--finetuned_checkpoint /NS/robustness_2/work/vnanda/invariances_in_reps/effects_of_finetuning/checkpoints/vit_base_patch32_224-base-imagenet21k-ft-cifar100/ftmode-full-lr-0.001-bs-256-aug_light1-wd_0.1-do_0.0-sd_0.0-warmup-100/epoch=98-topk=1.ckpt

python -m effects_of_finetuning.off_the_shelf_vits \
--eval_dataset cifar10 \
--base_dataset imagenet21k \
--finetuning_dataset cifar10 \
--model vit_base_patch32_224 \
--batch_size 250 \
--total_imgs 500 \
--base_checkpoint /NS/robustness_2/work/vnanda/invariances_in_reps/deep-learning-base/checkpoints/imagenet21k/B_32-i21k-300ep-lr_0.001-aug_light1-wd_0.1-do_0.0-sd_0.0.npz \
--finetuned_checkpoint /NS/robustness_2/work/vnanda/invariances_in_reps/effects_of_finetuning/checkpoints/vit_base_patch32_224-base-imagenet-ft-cifar10/ftmode-full-lr-0.001-bs-256-aug_light1-wd_0.03-do_0.0-sd_0.0-warmup-100/epoch=83-topk=1.ckpt
