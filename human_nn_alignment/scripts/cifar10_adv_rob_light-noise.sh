python -m human_nn_alignment.reg_free_loss \
--dataset cifar10 \
--checkpoint_path /NS/robustness_1/work/vnanda/CKA-Centered-Kernel-Alignment/checkpoints/cifar10/resnet18/robust/l2/eps1/checkpoint_rand_seed_2.pt.best \
--model resnet18 \
--append_path rob_l2eps1 \
--inversion_loss adv_alex_finetuned \
--seed_type light-noise

python -m human_nn_alignment.reg_free_loss \
--dataset cifar10 \
--checkpoint_path /NS/robustness_1/work/vnanda/CKA-Centered-Kernel-Alignment/checkpoints/cifar10/vgg16/robust/l2/eps1/checkpoint_rand_seed_2.pt.best \
--model vgg16 \
--append_path rob_l2eps1 \
--inversion_loss adv_alex_finetuned \
--seed_type light-noise

python -m human_nn_alignment.reg_free_loss \
--dataset cifar10 \
--checkpoint_path /NS/robustness_1/work/vnanda/CKA-Centered-Kernel-Alignment/checkpoints/cifar10/densenet121/robust/l2/eps1/checkpoint_rand_seed_2.pt.best \
--model densenet121 \
--append_path rob_l2eps1 \
--inversion_loss adv_alex_finetuned \
--seed_type light-noise

python -m human_nn_alignment.reg_free_loss \
--dataset cifar10 \
--checkpoint_path /NS/twitter_archive2/work/vnanda/adv-robustness/logs/robust/inceptionv3/eps1/checkpoint.pt.best \
--model inceptionv3 \
--append_path rob_l2eps1 \
--inversion_loss adv_alex_finetuned \
--seed_type light-noise
