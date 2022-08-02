## reg free

python -m human_nn_alignment.reg_free_loss \
--dataset cifar10 \
--checkpoint_path /NS/robustness_1/work/vnanda/CKA-Centered-Kernel-Alignment/checkpoints/cifar10/vgg16/nonrobust/checkpoint_rand_seed_2.pt.best \
--model vgg16 \
--append_path nonrob_rand_seed_2 \
--inversion_loss reg_free

## transforms

python -m human_nn_alignment.reg_free_loss \
--dataset cifar10 \
--checkpoint_path /NS/robustness_1/work/vnanda/CKA-Centered-Kernel-Alignment/checkpoints/cifar10/vgg16/nonrobust/checkpoint_rand_seed_2.pt.best \
--model vgg16 \
--append_path nonrob_rand_seed_2 \
--inversion_loss reg_free \
--trans_robust True

## Freq based

python -m human_nn_alignment.reg_free_loss \
--dataset cifar10 \
--checkpoint_path /NS/robustness_1/work/vnanda/CKA-Centered-Kernel-Alignment/checkpoints/cifar10/vgg16/nonrobust/checkpoint_rand_seed_2.pt.best \
--model vgg16 \
--append_path nonrob_rand_seed_2 \
--inversion_loss freq

### fft

python -m human_nn_alignment.reg_free_loss \
--dataset cifar10 \
--checkpoint_path /NS/robustness_1/work/vnanda/CKA-Centered-Kernel-Alignment/checkpoints/cifar10/vgg16/nonrobust/checkpoint_rand_seed_2.pt.best \
--model vgg16 \
--append_path nonrob_rand_seed_2 \
--inversion_loss reg_free \
--fft True

### transforms + freq

python -m human_nn_alignment.reg_free_loss \
--dataset cifar10 \
--checkpoint_path /NS/robustness_1/work/vnanda/CKA-Centered-Kernel-Alignment/checkpoints/cifar10/vgg16/nonrobust/checkpoint_rand_seed_2.pt.best \
--model vgg16 \
--append_path nonrob_rand_seed_2 \
--inversion_loss freq \
--trans_robust True


### fft + transforms

python -m human_nn_alignment.reg_free_loss \
--dataset cifar10 \
--checkpoint_path /NS/robustness_1/work/vnanda/CKA-Centered-Kernel-Alignment/checkpoints/cifar10/vgg16/nonrobust/checkpoint_rand_seed_2.pt.best \
--model vgg16 \
--append_path nonrob_rand_seed_2 \
--inversion_loss reg_free \
--trans_robust True \
--fft True

### fft + transforms + freq

python -m human_nn_alignment.reg_free_loss \
--dataset cifar10 \
--checkpoint_path /NS/robustness_1/work/vnanda/CKA-Centered-Kernel-Alignment/checkpoints/cifar10/vgg16/nonrobust/checkpoint_rand_seed_2.pt.best \
--model vgg16 \
--append_path nonrob_rand_seed_2 \
--inversion_loss freq \
--trans_robust True \
--fft True
