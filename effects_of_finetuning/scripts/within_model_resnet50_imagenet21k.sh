# base
python -m effects_of_finetuning.within_model_comparison \
--eval_dataset cifar100 \
--base_dataset imagenet21k-miil \
--model resnet50 \
--batch_size 25 \
--total_imgs 500 \
--checkpoint /NS/robustness_2/work/vnanda/invariances_in_reps/deep-learning-base/checkpoints/imagenet21k-miil/resnet50_miil_21k.pth \
--append_path base

# base
python -m effects_of_finetuning.within_model_comparison \
--eval_dataset cifar10 \
--base_dataset imagenet21k-miil \
--model resnet50 \
--batch_size 25 \
--total_imgs 500 \
--checkpoint /NS/robustness_2/work/vnanda/invariances_in_reps/deep-learning-base/checkpoints/imagenet21k-miil/resnet50_miil_21k.pth \
--append_path base
