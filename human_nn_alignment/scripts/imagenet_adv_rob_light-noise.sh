## OG

python -m human_nn_alignment.reg_free_loss \
--dataset imagenet \
--model resnet50 \
--append_path rob_l2eps3 \
--inversion_loss adv_alex_imagenet \
--checkpoint_path /NS/robustness_2/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/resnet-50-l2-eps3.ckpt \
--seed_type light-noise

python -m human_nn_alignment.reg_free_loss \
--dataset imagenet \
--model resnet18 \
--append_path rob_l2eps3 \
--inversion_loss adv_alex_imagenet \
--checkpoint_path /NS/robustness_2/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/resnet-18-l2-eps3.ckpt \
--seed_type light-noise

python -m human_nn_alignment.reg_free_loss \
--dataset imagenet \
--model vgg16_bn \
--append_path rob_l2eps3 \
--inversion_loss adv_alex_imagenet \
--checkpoint_path /NS/robustness_2/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/vgg16_bn_l2_eps3.ckpt \
--seed_type light-noise


## finetuned

python -m human_nn_alignment.reg_free_loss \
--dataset imagenet \
--model resnet50 \
--append_path rob_l2eps3 \
--inversion_loss adv_alex_finetuned \
--checkpoint_path /NS/robustness_2/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/resnet-50-l2-eps3.ckpt \
--seed_type light-noise

python -m human_nn_alignment.reg_free_loss \
--dataset imagenet \
--model resnet18 \
--append_path rob_l2eps3 \
--checkpoint_path /NS/robustness_2/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/resnet-18-l2-eps3.ckpt \
--inversion_loss adv_alex_finetuned \
--seed_type light-noise

python -m human_nn_alignment.reg_free_loss \
--dataset imagenet \
--model vgg16_bn \
--append_path rob_l2eps3 \
--inversion_loss adv_alex_finetuned \
--checkpoint_path /NS/robustness_2/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/vgg16_bn_l2_eps3.ckpt \
--seed_type light-noise