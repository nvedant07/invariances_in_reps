## reg-free

python -m human_nn_alignment.reg_free_loss \
--dataset cifar10 \
--checkpoint_path /NS/twitter_archive2/work/vnanda/adv-robustness/logs/robust/inceptionv3/eps1/adv_training_no_data_aug/checkpoint.pt.best \
--model inceptionv3 \
--append_path rob_noaug \
--inversion_loss reg_free

## transforms

python -m human_nn_alignment.reg_free_loss \
--dataset cifar10 \
--checkpoint_path /NS/twitter_archive2/work/vnanda/adv-robustness/logs/robust/inceptionv3/eps1/adv_training_no_data_aug/checkpoint.pt.best \
--model inceptionv3 \
--append_path rob_noaug \
--inversion_loss reg_free \
--trans_robust True

## reg-free + trans robust + fft

python -m human_nn_alignment.reg_free_loss \
--dataset cifar10 \
--checkpoint_path /NS/twitter_archive2/work/vnanda/adv-robustness/logs/robust/inceptionv3/eps1/adv_training_no_data_aug/checkpoint.pt.best \
--model inceptionv3 \
--append_path rob_noaug \
--inversion_loss reg_free \
--trans_robust True \
--fft True

