python -m human_nn_alignment.reg_free_loss \
--dataset imagenet \
--model resnet50 \
--append_path nonrob \
--inversion_loss adv_alex_finetuned_seed

python -m human_nn_alignment.reg_free_loss \
--dataset imagenet \
--model resnet18 \
--append_path nonrob \
--inversion_loss adv_alex_finetuned_seed

python -m human_nn_alignment.reg_free_loss \
--dataset imagenet \
--model vgg16_bn \
--append_path nonrob \
--inversion_loss adv_alex_finetuned_seed

python -m human_nn_alignment.reg_free_loss \
--dataset imagenet \
--model wide_resnet50_2 \
--append_path nonrob \
--inversion_loss adv_alex_finetuned_seed

python -m human_nn_alignment.reg_free_loss \
--dataset imagenet \
--model densenet161 \
--append_path nonrob \
--inversion_loss adv_alex_finetuned_seed