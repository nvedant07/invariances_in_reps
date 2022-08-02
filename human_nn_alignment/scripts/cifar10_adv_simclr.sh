# python -m human_nn_alignment.reg_free_loss \
# --dataset cifar10 \
# --model resnet18 \
# --append_path simclr_all \
# --checkpoint_path /NS/robustness_2/work/vnanda/deep_learning_base/checkpoints/cifar10/resnet18/simclr_bs_1024/lars_excludebn_True/epoch=999_rand_seed_1.ckpt \
# --inversion_loss adv_alex_imagenet

# python -m human_nn_alignment.reg_free_loss \
# --dataset cifar10 \
# --model resnet18 \
# --append_path simclr_adv \
# --checkpoint_path /NS/robustness_2/work/vnanda/deep_learning_base/checkpoints/cifar10/resnet18/simclr_bs_1024_adv_True/lars_excludebn_True/epoch=999_rand_seed_1.ckpt \
# --inversion_loss adv_alex_imagenet

# python -m human_nn_alignment.reg_free_loss \
# --dataset cifar10 \
# --model resnet18 \
# --append_path simclr_nocolor \
# --checkpoint_path /NS/robustness_2/work/ayan/deep_learning_base/checkpoints/cifar10/resnet18/simclr_bs_1024/lars_excludebn_True_augmentations_reduced/epoch=999_rand_seed_1.ckpt \
# --inversion_loss adv_alex_imagenet


python -m human_nn_alignment.reg_free_loss \
--dataset cifar10 \
--model resnet18 \
--append_path simclr_all \
--checkpoint_path /NS/robustness_2/work/vnanda/deep_learning_base/checkpoints/cifar10/resnet18/simclr_bs_1024/lars_excludebn_True/epoch=999_rand_seed_1.ckpt \
--inversion_loss adv_alex_imagenet \
--seed_type light-noise

python -m human_nn_alignment.reg_free_loss \
--dataset cifar10 \
--model resnet18 \
--append_path simclr_adv \
--checkpoint_path /NS/robustness_2/work/vnanda/deep_learning_base/checkpoints/cifar10/resnet18/simclr_bs_1024_adv_True/lars_excludebn_True/epoch=999_rand_seed_1.ckpt \
--inversion_loss adv_alex_imagenet \
--seed_type light-noise

python -m human_nn_alignment.reg_free_loss \
--dataset cifar10 \
--model resnet18 \
--append_path simclr_nocolor \
--checkpoint_path /NS/robustness_2/work/ayan/deep_learning_base/checkpoints/cifar10/resnet18/simclr_bs_1024/lars_excludebn_True_augmentations_reduced/epoch=999_rand_seed_1.ckpt \
--inversion_loss adv_alex_imagenet \
--seed_type light-noise