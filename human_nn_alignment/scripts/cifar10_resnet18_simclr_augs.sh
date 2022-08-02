# ## reg-free

# python -m human_nn_alignment.reg_free_loss \
# --dataset cifar10 \
# --checkpoint_path /NS/robustness_2/work/vnanda/deep_learning_base/checkpoints/cifar10/resnet18/supervised_nonrob_simclr_augs/checkpoint.pt.best \
# --model resnet18 \
# --append_path nonrob_simclraug \
# --inversion_loss reg_free

# ## transforms

# python -m human_nn_alignment.reg_free_loss \
# --dataset cifar10 \
# --checkpoint_path /NS/robustness_2/work/vnanda/deep_learning_base/checkpoints/cifar10/resnet18/supervised_nonrob_simclr_augs/checkpoint.pt.best \
# --model resnet18 \
# --append_path nonrob_simclraug \
# --inversion_loss reg_free \
# --trans_robust True

# ## reg-free + trans robust + fft

# python -m human_nn_alignment.reg_free_loss \
# --dataset cifar10 \
# --checkpoint_path /NS/robustness_2/work/vnanda/deep_learning_base/checkpoints/cifar10/resnet18/supervised_nonrob_simclr_augs/checkpoint.pt.best \
# --model resnet18 \
# --append_path nonrob_simclraug \
# --inversion_loss reg_free \
# --trans_robust True \
# --fft True

### adv

# python -m human_nn_alignment.reg_free_loss \
# --dataset cifar10 \
# --checkpoint_path /NS/robustness_2/work/vnanda/deep_learning_base/checkpoints/cifar10/resnet18/supervised_nonrob_simclr_augs/checkpoint.pt.best \
# --model resnet18 \
# --append_path nonrob_simclraug \
# --inversion_loss adv_alex_finetuned

# python -m human_nn_alignment.reg_free_loss \
# --dataset cifar10 \
# --checkpoint_path /NS/robustness_2/work/vnanda/deep_learning_base/checkpoints/cifar10/resnet18/supervised_nonrob_simclr_augs/checkpoint.pt.best \
# --model resnet18 \
# --append_path nonrob_simclraug \
# --inversion_loss adv_alex_imagenet

python -m human_nn_alignment.reg_free_loss \
--dataset cifar10 \
--checkpoint_path /NS/robustness_2/work/vnanda/deep_learning_base/checkpoints/cifar10/resnet18/supervised_nonrob_simclr_augs/checkpoint.pt.best \
--model resnet18 \
--append_path nonrob_simclraug \
--inversion_loss adv_alex_finetuned \
--seed_type light-noise

python -m human_nn_alignment.reg_free_loss \
--dataset cifar10 \
--checkpoint_path /NS/robustness_2/work/vnanda/deep_learning_base/checkpoints/cifar10/resnet18/supervised_nonrob_simclr_augs/checkpoint.pt.best \
--model resnet18 \
--append_path nonrob_simclraug \
--inversion_loss adv_alex_imagenet \
--seed_type light-noise