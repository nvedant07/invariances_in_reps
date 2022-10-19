# base
python -m effects_of_finetuning.within_model_comparison \
--eval_dataset cifar100 \
--base_dataset imagenet21k \
--model vit_small_patch16_224 \
--batch_size 100 \
--total_imgs 500 \
--checkpoint /NS/robustness_2/work/vnanda/invariances_in_reps/deep-learning-base/checkpoints/imagenet21k/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz \
--append_path base \
--image_size 224

# finetuned
python -m effects_of_finetuning.within_model_comparison \
--eval_dataset cifar100 \
--base_dataset imagenet21k \
--model vit_small_patch16_224 \
--batch_size 100 \
--total_imgs 500 \
--checkpoint /NS/robustness_2/work/vnanda/invariances_in_reps/effects_of_finetuning/checkpoints/vit_small_patch16_224-base-imagenet21k-ft-cifar100/ftmode-full-lr-0.001-bs-256-aug_light1-wd_0.03-do_0.0-sd_0.0-warmup-100/epoch=70-topk=1.ckpt \
--append_path finetuned \
--image_size 224

# base
CUDA_VISIBLE_DEVICES=1 python -m effects_of_finetuning.within_model_comparison \
--eval_dataset cifar10 \
--base_dataset imagenet21k \
--model vit_small_patch16_224 \
--batch_size 100 \
--total_imgs 500 \
--checkpoint /NS/robustness_2/work/vnanda/invariances_in_reps/deep-learning-base/checkpoints/imagenet21k/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz \
--append_path base \
--image_size 224

# finetuned
CUDA_VISIBLE_DEVICES=1 python -m effects_of_finetuning.within_model_comparison \
--eval_dataset cifar10 \
--base_dataset imagenet21k \
--model vit_small_patch16_224 \
--batch_size 100 \
--total_imgs 500 \
--checkpoint /NS/robustness_2/work/vnanda/invariances_in_reps/effects_of_finetuning/checkpoints/vit_small_patch16_224-base-imagenet21k-ft-cifar10/ftmode-full-lr-0.001-bs-256-aug_light1-wd_0.03-do_0.0-sd_0.0-warmup-100/epoch=83-topk=1.ckpt \
--append_path finetuned \
--image_size 224
