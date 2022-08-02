#### beta = 0.1
## reg-free

python -m human_nn_alignment.reg_free_loss \
--dataset cifar10 \
--checkpoint_path /NS/robustness_2/work/vnanda/MART/checkpoints/densenet121/beta_0.1_eps_1.000/checkpoint.pt.best \
--model densenet121 \
--append_path martbeta0.1eps1 \
--inversion_loss reg_free

## transforms

python -m human_nn_alignment.reg_free_loss \
--dataset cifar10 \
--checkpoint_path /NS/robustness_2/work/vnanda/MART/checkpoints/densenet121/beta_0.1_eps_1.000/checkpoint.pt.best \
--model densenet121 \
--append_path martbeta0.1eps1 \
--inversion_loss reg_free \
--trans_robust True

## reg-free + trans robust + fft

python -m human_nn_alignment.reg_free_loss \
--dataset cifar10 \
--checkpoint_path /NS/robustness_2/work/vnanda/MART/checkpoints/densenet121/beta_0.1_eps_1.000/checkpoint.pt.best \
--model densenet121 \
--append_path martbeta0.1eps1 \
--inversion_loss reg_free \
--trans_robust True \
--fft True

#### beta = 1
## reg-free

python -m human_nn_alignment.reg_free_loss \
--dataset cifar10 \
--checkpoint_path /NS/robustness_2/work/vnanda/MART/checkpoints/densenet121/beta_1.0_eps_1.000/checkpoint.pt.best \
--model densenet121 \
--append_path martbeta1eps1 \
--inversion_loss reg_free

## transforms

python -m human_nn_alignment.reg_free_loss \
--dataset cifar10 \
--checkpoint_path /NS/robustness_2/work/vnanda/MART/checkpoints/densenet121/beta_1.0_eps_1.000/checkpoint.pt.best \
--model densenet121 \
--append_path martbeta1eps1 \
--inversion_loss reg_free \
--trans_robust True

## reg-free + trans robust + fft

python -m human_nn_alignment.reg_free_loss \
--dataset cifar10 \
--checkpoint_path /NS/robustness_2/work/vnanda/MART/checkpoints/densenet121/beta_1.0_eps_1.000/checkpoint.pt.best \
--model densenet121 \
--append_path martbeta1eps1 \
--inversion_loss reg_free \
--trans_robust True \
--fft True

#### beta = 6
## reg-free

python -m human_nn_alignment.reg_free_loss \
--dataset cifar10 \
--checkpoint_path /NS/robustness_2/work/vnanda/MART/checkpoints/densenet121/beta_6.0_eps_1.000/checkpoint.pt.best \
--model densenet121 \
--append_path martbeta6eps1 \
--inversion_loss reg_free

## transforms

python -m human_nn_alignment.reg_free_loss \
--dataset cifar10 \
--checkpoint_path /NS/robustness_2/work/vnanda/MART/checkpoints/densenet121/beta_6.0_eps_1.000/checkpoint.pt.best \
--model densenet121 \
--append_path martbeta6eps1 \
--inversion_loss reg_free \
--trans_robust True

## reg-free + trans robust + fft

python -m human_nn_alignment.reg_free_loss \
--dataset cifar10 \
--checkpoint_path /NS/robustness_2/work/vnanda/MART/checkpoints/densenet121/beta_6.0_eps_1.000/checkpoint.pt.best \
--model densenet121 \
--append_path martbeta6eps1 \
--inversion_loss reg_free \
--trans_robust True \
--fft True

#### beta = 10
## reg-free

python -m human_nn_alignment.reg_free_loss \
--dataset cifar10 \
--checkpoint_path /NS/robustness_2/work/vnanda/MART/checkpoints/densenet121/beta_10.0_eps_1.000/checkpoint.pt.best \
--model densenet121 \
--append_path martbeta10eps1 \
--inversion_loss reg_free

## transforms

python -m human_nn_alignment.reg_free_loss \
--dataset cifar10 \
--checkpoint_path /NS/robustness_2/work/vnanda/MART/checkpoints/densenet121/beta_10.0_eps_1.000/checkpoint.pt.best \
--model densenet121 \
--append_path martbeta10eps1 \
--inversion_loss reg_free \
--trans_robust True

## reg-free + trans robust + fft

python -m human_nn_alignment.reg_free_loss \
--dataset cifar10 \
--checkpoint_path /NS/robustness_2/work/vnanda/MART/checkpoints/densenet121/beta_10.0_eps_1.000/checkpoint.pt.best \
--model densenet121 \
--append_path martbeta10eps1 \
--inversion_loss reg_free \
--trans_robust True \
--fft True
