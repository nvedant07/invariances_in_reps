## Reg Free
python -m human_nn_alignment.reg_free_loss \
--dataset imagenet \
--model resnet18 \
--append_path rob_l2eps3 \
--checkpoint_path /NS/robustness_2/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/resnet-18-l2-eps3.ckpt \
--inversion_loss reg_free

## Freq based reg
python -m human_nn_alignment.reg_free_loss \
--dataset imagenet \
--model resnet18 \
--append_path rob_l2eps3 \
--checkpoint_path /NS/robustness_2/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/resnet-18-l2-eps3.ckpt \
--inversion_loss freq

python -m human_nn_alignment.reg_free_loss \
--dataset imagenet \
--model resnet18 \
--append_path rob_l2eps3 \
--checkpoint_path /NS/robustness_2/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/resnet-18-l2-eps3.ckpt \
--inversion_loss freq_blur

python -m human_nn_alignment.reg_free_loss \
--dataset imagenet \
--model resnet18 \
--append_path rob_l2eps3 \
--checkpoint_path /NS/robustness_2/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/resnet-18-l2-eps3.ckpt \
--inversion_loss freq_lp

python -m human_nn_alignment.reg_free_loss \
--dataset imagenet \
--model resnet18 \
--append_path rob_l2eps3 \
--checkpoint_path /NS/robustness_2/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/resnet-18-l2-eps3.ckpt \
--inversion_loss freq_tv

## Transformation robustness based reg
python -m human_nn_alignment.reg_free_loss \
--dataset imagenet \
--model resnet18 \
--append_path rob_l2eps3 \
--checkpoint_path /NS/robustness_2/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/resnet-18-l2-eps3.ckpt \
--inversion_loss reg_free \
--trans_robust True

## freq + transformation robustness

python -m human_nn_alignment.reg_free_loss \
--dataset imagenet \
--model resnet18 \
--append_path rob_l2eps3 \
--checkpoint_path /NS/robustness_2/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/resnet-18-l2-eps3.ckpt \
--inversion_loss freq_tv \
--trans_robust True

python -m human_nn_alignment.reg_free_loss \
--dataset imagenet \
--model resnet18 \
--append_path rob_l2eps3 \
--checkpoint_path /NS/robustness_2/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/resnet-18-l2-eps3.ckpt \
--inversion_loss freq \
--trans_robust True

python -m human_nn_alignment.reg_free_loss \
--dataset imagenet \
--model resnet18 \
--append_path rob_l2eps3 \
--checkpoint_path /NS/robustness_2/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/resnet-18-l2-eps3.ckpt \
--inversion_loss freq_blur \
--trans_robust True

python -m human_nn_alignment.reg_free_loss \
--dataset imagenet \
--model resnet18 \
--append_path rob_l2eps3 \
--checkpoint_path /NS/robustness_2/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/resnet-18-l2-eps3.ckpt \
--inversion_loss freq_lp \
--trans_robust True
