## OG

python -m human_nn_alignment.reg_free_loss \
--dataset imagenet \
--model resnet50 \
--append_path nonrob \
--inversion_loss adv_alex_imagenet \
--seed_type light-noise

python -m human_nn_alignment.reg_free_loss \
--dataset imagenet \
--model resnet18 \
--append_path nonrob \
--inversion_loss adv_alex_imagenet \
--seed_type light-noise

python -m human_nn_alignment.reg_free_loss \
--dataset imagenet \
--model vgg16_bn \
--append_path nonrob \
--inversion_loss adv_alex_imagenet \
--seed_type light-noise


## finetuned

python -m human_nn_alignment.reg_free_loss \
--dataset imagenet \
--model resnet50 \
--append_path nonrob \
--inversion_loss adv_alex_finetuned \
--seed_type light-noise

python -m human_nn_alignment.reg_free_loss \
--dataset imagenet \
--model resnet18 \
--append_path nonrob \
--inversion_loss adv_alex_finetuned \
--seed_type light-noise

python -m human_nn_alignment.reg_free_loss \
--dataset imagenet \
--model vgg16_bn \
--append_path nonrob \
--inversion_loss adv_alex_finetuned \
--seed_type light-noise