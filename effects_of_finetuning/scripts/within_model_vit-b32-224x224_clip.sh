# base
python -m effects_of_finetuning.within_model_comparison \
--eval_dataset cifar100 \
--base_dataset clip \
--model vit_base_patch32_224 \
--batch_size 100 \
--total_imgs 500 \
--checkpoint /NS/robustness_2/work/vnanda/invariances_in_reps/deep-learning-base/checkpoints/clip/ViT-B-32.pt \
--append_path base

# base
python -m effects_of_finetuning.within_model_comparison \
--eval_dataset cifar10 \
--base_dataset clip \
--model vit_base_patch32_224 \
--batch_size 100 \
--total_imgs 500 \
--checkpoint /NS/robustness_2/work/vnanda/invariances_in_reps/deep-learning-base/checkpoints/clip/ViT-B-32.pt \
--append_path base
