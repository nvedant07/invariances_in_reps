## reg free

python -m human_nn_alignment.reg_free_loss \
--dataset cifar10 \
--checkpoint_path /NS/twitter_archive2/work/vnanda/adv-robustness/logs/nonrobust/inceptionv3/checkpoint.pt.best \
--model inceptionv3 \
--append_path nonrob_rand_seed_2 \
--inversion_loss reg_free

## transforms

python -m human_nn_alignment.reg_free_loss \
--dataset cifar10 \
--checkpoint_path /NS/twitter_archive2/work/vnanda/adv-robustness/logs/nonrobust/inceptionv3/checkpoint.pt.best \
--model inceptionv3 \
--append_path nonrob_rand_seed_2 \
--inversion_loss reg_free \
--trans_robust True

## Freq based

python -m human_nn_alignment.reg_free_loss \
--dataset cifar10 \
--checkpoint_path /NS/twitter_archive2/work/vnanda/adv-robustness/logs/nonrobust/inceptionv3/checkpoint.pt.best \
--model inceptionv3 \
--append_path nonrob_rand_seed_2 \
--inversion_loss freq

### fft

python -m human_nn_alignment.reg_free_loss \
--dataset cifar10 \
--checkpoint_path /NS/twitter_archive2/work/vnanda/adv-robustness/logs/nonrobust/inceptionv3/checkpoint.pt.best \
--model inceptionv3 \
--append_path nonrob_rand_seed_2 \
--inversion_loss reg_free \
--fft True

### transforms + freq

python -m human_nn_alignment.reg_free_loss \
--dataset cifar10 \
--checkpoint_path /NS/twitter_archive2/work/vnanda/adv-robustness/logs/nonrobust/inceptionv3/checkpoint.pt.best \
--model inceptionv3 \
--append_path nonrob_rand_seed_2 \
--inversion_loss freq \
--trans_robust True


### fft + transforms

python -m human_nn_alignment.reg_free_loss \
--dataset cifar10 \
--checkpoint_path /NS/twitter_archive2/work/vnanda/adv-robustness/logs/nonrobust/inceptionv3/checkpoint.pt.best \
--model inceptionv3 \
--append_path nonrob_rand_seed_2 \
--inversion_loss reg_free \
--trans_robust True \
--fft True

### fft + transforms + freq

python -m human_nn_alignment.reg_free_loss \
--dataset cifar10 \
--checkpoint_path /NS/twitter_archive2/work/vnanda/adv-robustness/logs/nonrobust/inceptionv3/checkpoint.pt.best \
--model inceptionv3 \
--append_path nonrob_rand_seed_2 \
--inversion_loss freq \
--trans_robust True \
--fft True
