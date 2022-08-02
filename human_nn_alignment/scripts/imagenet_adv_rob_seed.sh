# python -m human_nn_alignment.reg_free_loss \
# --dataset imagenet \
# --model resnet50 \
# --append_path rob_l2eps3 \
# --checkpoint_path /NS/robustness_2/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/resnet-50-l2-eps3.ckpt \
# --inversion_loss adv_alex_finetuned_seed

python -m human_nn_alignment.reg_free_loss \
--dataset imagenet \
--model resnet18 \
--append_path rob_l2eps3 \
--checkpoint_path /NS/robustness_2/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/resnet-18-l2-eps3.ckpt \
--inversion_loss adv_alex_finetuned_seed

python -m human_nn_alignment.reg_free_loss \
--dataset imagenet \
--model vgg16_bn \
--append_path rob_l2eps3 \
--checkpoint_path /NS/robustness_2/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/vgg16_bn_l2_eps3.ckpt \
--inversion_loss adv_alex_finetuned_seed

python -m human_nn_alignment.reg_free_loss \
--dataset imagenet \
--model wide_resnet50_2 \
--append_path rob_l2eps3 \
--checkpoint_path /NS/robustness_2/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/wideresnet-50-2-l2-eps3.ckpt \
--inversion_loss adv_alex_finetuned_seed

python -m human_nn_alignment.reg_free_loss \
--dataset imagenet \
--model densenet161 \
--append_path rob_l2eps3 \
--checkpoint_path /NS/robustness_2/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/densenet161-l2-eps3.ckpt \
--inversion_loss adv_alex_finetuned_seed