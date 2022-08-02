#### beta = 0.1
## reg-free

python -m human_nn_alignment.reg_free_loss \
--dataset cifar10 \
--checkpoint_path /NS/robustness_2/work/vnanda/adv-trades/checkpoints/resnet18/beta_0.1_eps_1.000/checkpoint.pt.best \
--model resnet18 \
--append_path tradesbeta0.1eps1 \
--inversion_loss reg_free

## transforms

python -m human_nn_alignment.reg_free_loss \
--dataset cifar10 \
--checkpoint_path /NS/robustness_2/work/vnanda/adv-trades/checkpoints/resnet18/beta_0.1_eps_1.000/checkpoint.pt.best \
--model resnet18 \
--append_path tradesbeta0.1eps1 \
--inversion_loss reg_free \
--trans_robust True

## reg-free + trans robust + fft

python -m human_nn_alignment.reg_free_loss \
--dataset cifar10 \
--checkpoint_path /NS/robustness_2/work/vnanda/adv-trades/checkpoints/resnet18/beta_0.1_eps_1.000/checkpoint.pt.best \
--model resnet18 \
--append_path tradesbeta0.1eps1 \
--inversion_loss reg_free \
--trans_robust True \
--fft True

#### beta = 1
## reg-free

python -m human_nn_alignment.reg_free_loss \
--dataset cifar10 \
--checkpoint_path /NS/robustness_2/work/vnanda/adv-trades/checkpoints/resnet18/beta_1.0_eps_1.000/checkpoint.pt.best \
--model resnet18 \
--append_path tradesbeta1eps1 \
--inversion_loss reg_free

## transforms

python -m human_nn_alignment.reg_free_loss \
--dataset cifar10 \
--checkpoint_path /NS/robustness_2/work/vnanda/adv-trades/checkpoints/resnet18/beta_1.0_eps_1.000/checkpoint.pt.best \
--model resnet18 \
--append_path tradesbeta1eps1 \
--inversion_loss reg_free \
--trans_robust True

## reg-free + trans robust + fft

python -m human_nn_alignment.reg_free_loss \
--dataset cifar10 \
--checkpoint_path /NS/robustness_2/work/vnanda/adv-trades/checkpoints/resnet18/beta_1.0_eps_1.000/checkpoint.pt.best \
--model resnet18 \
--append_path tradesbeta1eps1 \
--inversion_loss reg_free \
--trans_robust True \
--fft True

#### beta = 6
## reg-free

python -m human_nn_alignment.reg_free_loss \
--dataset cifar10 \
--checkpoint_path /NS/robustness_2/work/vnanda/adv-trades/checkpoints/resnet18/beta_6.0_eps_1.000/checkpoint.pt.best \
--model resnet18 \
--append_path tradesbeta6eps1 \
--inversion_loss reg_free

## transforms

python -m human_nn_alignment.reg_free_loss \
--dataset cifar10 \
--checkpoint_path /NS/robustness_2/work/vnanda/adv-trades/checkpoints/resnet18/beta_6.0_eps_1.000/checkpoint.pt.best \
--model resnet18 \
--append_path tradesbeta6eps1 \
--inversion_loss reg_free \
--trans_robust True

## reg-free + trans robust + fft

python -m human_nn_alignment.reg_free_loss \
--dataset cifar10 \
--checkpoint_path /NS/robustness_2/work/vnanda/adv-trades/checkpoints/resnet18/beta_6.0_eps_1.000/checkpoint.pt.best \
--model resnet18 \
--append_path tradesbeta6eps1 \
--inversion_loss reg_free \
--trans_robust True \
--fft True

#### beta = 10
## reg-free

python -m human_nn_alignment.reg_free_loss \
--dataset cifar10 \
--checkpoint_path /NS/robustness_2/work/vnanda/adv-trades/checkpoints/resnet18/beta_10.0_eps_1.000/checkpoint.pt.best \
--model resnet18 \
--append_path tradesbeta10eps1 \
--inversion_loss reg_free

## transforms

python -m human_nn_alignment.reg_free_loss \
--dataset cifar10 \
--checkpoint_path /NS/robustness_2/work/vnanda/adv-trades/checkpoints/resnet18/beta_10.0_eps_1.000/checkpoint.pt.best \
--model resnet18 \
--append_path tradesbeta10eps1 \
--inversion_loss reg_free \
--trans_robust True

## reg-free + trans robust + fft

python -m human_nn_alignment.reg_free_loss \
--dataset cifar10 \
--checkpoint_path /NS/robustness_2/work/vnanda/adv-trades/checkpoints/resnet18/beta_10.0_eps_1.000/checkpoint.pt.best \
--model resnet18 \
--append_path tradesbeta10eps1 \
--inversion_loss reg_free \
--trans_robust True \
--fft True
