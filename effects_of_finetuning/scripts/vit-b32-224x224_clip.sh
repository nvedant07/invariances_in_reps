python -m effects_of_finetuning.off_the_shelf_vits \
--eval_dataset cifar100 \
--base_dataset clip \
--finetuning_dataset cifar100 \
--model vit_base_patch32_224 \
--batch_size 100 \
--total_imgs 500 \
--base_checkpoint /NS/robustness_2/work/vnanda/invariances_in_reps/deep-learning-base/checkpoints/clip/ViT-B-32.pt \
--finetuned_checkpoint /NS/robustness_2/work/vnanda/invariances_in_reps/effects_of_finetuning/checkpoints/vit_base_patch32_224-base-clip-ft-cifar100/ftmode-full-lr-0.001-bs-256-clip-warmup-100/epoch=88-topk=1.ckpt

python -m effects_of_finetuning.off_the_shelf_vits \
--eval_dataset cifar10 \
--base_dataset clip \
--finetuning_dataset cifar10 \
--model vit_base_patch32_224 \
--batch_size 100 \
--total_imgs 500 \
--base_checkpoint /NS/robustness_2/work/vnanda/invariances_in_reps/deep-learning-base/checkpoints/clip/ViT-B-32.pt \
--finetuned_checkpoint /NS/robustness_2/work/vnanda/invariances_in_reps/effects_of_finetuning/checkpoints/vit_base_patch32_224-base-clip-ft-cifar10/ftmode-full-lr-0.001-bs-256-clip-warmup-100/epoch=93-topk=1.ckpt
