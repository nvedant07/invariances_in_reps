## Reg Free
python -m human_nn_alignment.reg_free_loss \
--dataset imagenet \
--model resnet18 \
--append_path rob_l2eps3 \
--seed_type light-noise \
--inversion_loss reg_free \
--checkpoint_path /NS/robustness_2/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/resnet-18-l2-eps3.ckpt

## robustness to transforms
python -m human_nn_alignment.reg_free_loss \
--dataset imagenet \
--model resnet18 \
--append_path rob_l2eps3 \
--seed_type light-noise \
--inversion_loss reg_free \
--checkpoint_path /NS/robustness_2/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/resnet-18-l2-eps3.ckpt \
--trans_robust True

## fft precondition
python -m human_nn_alignment.reg_free_loss \
--dataset imagenet \
--model resnet18 \
--append_path rob_l2eps3 \
--seed_type light-noise \
--inversion_loss reg_free \
--checkpoint_path /NS/robustness_2/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/resnet-18-l2-eps3.ckpt \
--fft True

## Freq based reg
python -m human_nn_alignment.reg_free_loss \
--dataset imagenet \
--model resnet18 \
--append_path rob_l2eps3 \
--seed_type light-noise \
--inversion_loss freq \
--checkpoint_path /NS/robustness_2/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/resnet-18-l2-eps3.ckpt

## Freq + robustness to transforms
python -m human_nn_alignment.reg_free_loss \
--dataset imagenet \
--model resnet18 \
--append_path rob_l2eps3 \
--seed_type light-noise \
--inversion_loss freq \
--checkpoint_path /NS/robustness_2/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/resnet-18-l2-eps3.ckpt \
--trans_robust True

## fft precondition + robustness to transforms
python -m human_nn_alignment.reg_free_loss \
--dataset imagenet \
--model resnet18 \
--append_path rob_l2eps3 \
--seed_type light-noise \
--inversion_loss reg_free \
--checkpoint_path /NS/robustness_2/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/resnet-18-l2-eps3.ckpt \
--trans_robust True \
--fft True

## Freq + robustness to transforms + fft precondition

python -m human_nn_alignment.reg_free_loss \
--dataset imagenet \
--model resnet18 \
--append_path rob_l2eps3 \
--seed_type light-noise \
--inversion_loss freq \
--checkpoint_path /NS/robustness_2/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/resnet-18-l2-eps3.ckpt \
--trans_robust True \
--fft True
