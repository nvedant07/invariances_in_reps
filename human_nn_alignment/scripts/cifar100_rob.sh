python -m human_nn_alignment.reg_free_loss \
--dataset cifar100 \
--checkpoint_path /NS/robustness_2/work/vnanda/adv-robustness/logs/robust_cifar100/r18/eps1/adv_training/checkpoint.pt.best \
--model resnet18 \
--append_path rob_l2eps1 \
--inversion_loss reg_free

python -m human_nn_alignment.reg_free_loss \
--dataset cifar100 \
--checkpoint_path /NS/robustness_2/work/vnanda/adv-robustness/logs/robust_cifar100/vgg16/eps1/adv_training/checkpoint.pt.best \
--model vgg16 \
--append_path rob_l2eps1 \
--inversion_loss reg_free

python -m human_nn_alignment.reg_free_loss \
--dataset cifar100 \
--checkpoint_path /NS/robustness_2/work/vnanda/adv-robustness/logs/robust_cifar100/inceptionv3/eps1/adv_training/checkpoint.pt.best \
--model inceptionv3 \
--append_path rob_l2eps1 \
--inversion_loss reg_free

# python -m human_nn_alignment.reg_free_loss \
# --dataset cifar100 \
# --checkpoint_path /NS/robustness_2/work/vnanda/adv-robustness/logs/robust_cifar100/densenet121/eps1/adv_training/checkpoint.pt.best \
# --model densenet121 \
# --append_path rob_l2eps1 \
# --inversion_loss reg_free