## reg free

python -m human_nn_alignment.reg_free_loss \
--dataset cifar10 \
--checkpoint_path /NS/robustness_1/work/vnanda/CKA-Centered-Kernel-Alignment/checkpoints/cifar10/resnet18/robust/l2/eps1/checkpoint_rand_seed_2.pt.best \
--model resnet18 \
--append_path robl2eps1_rand_seed_2 \
--inversion_loss reg_free

## transforms

python -m human_nn_alignment.reg_free_loss \
--dataset cifar10 \
--checkpoint_path /NS/robustness_1/work/vnanda/CKA-Centered-Kernel-Alignment/checkpoints/cifar10/resnet18/robust/l2/eps1/checkpoint_rand_seed_2.pt.best \
--model resnet18 \
--append_path robl2eps1_rand_seed_2 \
--inversion_loss reg_free \
--trans_robust True

## Freq based

python -m human_nn_alignment.reg_free_loss \
--dataset cifar10 \
--checkpoint_path /NS/robustness_1/work/vnanda/CKA-Centered-Kernel-Alignment/checkpoints/cifar10/resnet18/robust/l2/eps1/checkpoint_rand_seed_2.pt.best \
--model resnet18 \
--append_path robl2eps1_rand_seed_2 \
--inversion_loss freq

### fft

python -m human_nn_alignment.reg_free_loss \
--dataset cifar10 \
--checkpoint_path /NS/robustness_1/work/vnanda/CKA-Centered-Kernel-Alignment/checkpoints/cifar10/resnet18/robust/l2/eps1/checkpoint_rand_seed_2.pt.best \
--model resnet18 \
--append_path robl2eps1_rand_seed_2 \
--inversion_loss reg_free \
--fft True

### transforms + freq

python -m human_nn_alignment.reg_free_loss \
--dataset cifar10 \
--checkpoint_path /NS/robustness_1/work/vnanda/CKA-Centered-Kernel-Alignment/checkpoints/cifar10/resnet18/robust/l2/eps1/checkpoint_rand_seed_2.pt.best \
--model resnet18 \
--append_path robl2eps1_rand_seed_2 \
--inversion_loss freq \
--trans_robust True


### fft + transforms

python -m human_nn_alignment.reg_free_loss \
--dataset cifar10 \
--checkpoint_path /NS/robustness_1/work/vnanda/CKA-Centered-Kernel-Alignment/checkpoints/cifar10/resnet18/robust/l2/eps1/checkpoint_rand_seed_2.pt.best \
--model resnet18 \
--append_path robl2eps1_rand_seed_2 \
--inversion_loss reg_free \
--trans_robust True \
--fft True

### fft + transforms + freq

python -m human_nn_alignment.reg_free_loss \
--dataset cifar10 \
--checkpoint_path /NS/robustness_1/work/vnanda/CKA-Centered-Kernel-Alignment/checkpoints/cifar10/resnet18/robust/l2/eps1/checkpoint_rand_seed_2.pt.best \
--model resnet18 \
--append_path robl2eps1_rand_seed_2 \
--inversion_loss freq \
--trans_robust True \
--fft True
