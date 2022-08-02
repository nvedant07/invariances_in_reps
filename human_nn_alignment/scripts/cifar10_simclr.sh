python -m human_nn_alignment.simclr_all_epochs \
--dataset cifar10 \
--model resnet18 \
--append_path simclr_all \
--inversion_loss reg_free \
--checkpoint_dir /NS/robustness_2/work/vnanda/deep_learning_base/checkpoints/cifar10/resnet18/simclr_bs_1024/lars_excludebn_True

python -m human_nn_alignment.simclr_all_epochs \
--dataset cifar10 \
--model resnet18 \
--append_path simclr_adv \
--inversion_loss reg_free \
--checkpoint_dir /NS/robustness_2/work/vnanda/deep_learning_base/checkpoints/cifar10/resnet18/simclr_bs_1024_adv_True/lars_excludebn_True

python -m human_nn_alignment.simclr_all_epochs \
--dataset cifar10 \
--model resnet18 \
--append_path simclr_nocolor \
--inversion_loss reg_free \
--checkpoint_dir /NS/robustness_2/work/ayan/deep_learning_base/checkpoints/cifar10/resnet18/simclr_bs_1024/lars_excludebn_True_augmentations_reduced