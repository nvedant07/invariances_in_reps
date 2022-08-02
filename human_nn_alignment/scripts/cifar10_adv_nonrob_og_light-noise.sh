python -m human_nn_alignment.reg_free_loss \
--dataset cifar10 \
--checkpoint_path /NS/robustness_1/work/vnanda/CKA-Centered-Kernel-Alignment/checkpoints/cifar10/resnet18/nonrobust/checkpoint_rand_seed_2.pt.best \
--model resnet18 \
--append_path nonrob_rand_seed_2 \
--inversion_loss adv_alex_imagenet \
--seed_type light-noise

python -m human_nn_alignment.reg_free_loss \
--dataset cifar10 \
--checkpoint_path /NS/robustness_1/work/vnanda/CKA-Centered-Kernel-Alignment/checkpoints/cifar10/vgg16/nonrobust/checkpoint_rand_seed_2.pt.best \
--model vgg16 \
--append_path nonrob_rand_seed_2 \
--inversion_loss adv_alex_imagenet \
--seed_type light-noise

python -m human_nn_alignment.reg_free_loss \
--dataset cifar10 \
--checkpoint_path /NS/robustness_1/work/vnanda/CKA-Centered-Kernel-Alignment/checkpoints/cifar10/densenet121/nonrobust/checkpoint_rand_seed_2.pt.best \
--model densenet121 \
--append_path nonrob_rand_seed_2 \
--inversion_loss adv_alex_imagenet \
--seed_type light-noise

python -m human_nn_alignment.reg_free_loss \
--dataset cifar10 \
--checkpoint_path /NS/twitter_archive2/work/vnanda/adv-robustness/logs/nonrobust/inceptionv3/checkpoint.pt.best \
--model inceptionv3 \
--append_path nonrob_rand_seed_2 \
--inversion_loss adv_alex_imagenet \
--seed_type light-noise
