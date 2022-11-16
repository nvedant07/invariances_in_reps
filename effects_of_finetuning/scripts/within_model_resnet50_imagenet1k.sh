# base
python -m effects_of_finetuning.within_model_comparison \
--eval_dataset cifar100 \
--base_dataset imagenet \
--model resnet50 \
--batch_size 50 \
--total_imgs 500 \
--append_path base

# base
python -m effects_of_finetuning.within_model_comparison \
--eval_dataset cifar10 \
--base_dataset imagenet \
--model resnet50 \
--batch_size 50 \
--total_imgs 500 \
--append_path base
