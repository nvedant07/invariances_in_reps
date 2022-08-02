## Reg Free
python -m human_nn_alignment.reg_free_loss \
--dataset imagenet \
--model densenet161 \
--append_path nonrob \
--inversion_loss reg_free

## robustness to transforms
python -m human_nn_alignment.reg_free_loss \
--dataset imagenet \
--model densenet161 \
--append_path nonrob \
--inversion_loss reg_free \
--trans_robust True

## fft precondition
python -m human_nn_alignment.reg_free_loss \
--dataset imagenet \
--model densenet161 \
--append_path nonrob \
--inversion_loss reg_free \
--fft True

## Freq based reg
python -m human_nn_alignment.reg_free_loss \
--dataset imagenet \
--model densenet161 \
--append_path nonrob \
--inversion_loss freq

## Freq + robustness to transforms
python -m human_nn_alignment.reg_free_loss \
--dataset imagenet \
--model densenet161 \
--append_path nonrob \
--inversion_loss freq \
--trans_robust True

## fft precondition + robustness to transforms
python -m human_nn_alignment.reg_free_loss \
--dataset imagenet \
--model densenet161 \
--append_path nonrob \
--inversion_loss reg_free \
--trans_robust True \
--fft True

## Freq + robustness to transforms + fft precondition

python -m human_nn_alignment.reg_free_loss \
--dataset imagenet \
--model densenet161 \
--append_path nonrob \
--inversion_loss freq \
--trans_robust True \
--fft True
