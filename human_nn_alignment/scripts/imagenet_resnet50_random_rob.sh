python -m human_nn_alignment.reg_free_loss \
--source_dataset imagenet \
--dataset random_0_1 \
--model resnet50 \
--batch_size 100 \
--checkpoint_path /NS/robustness_2/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/resnet-50-l2-eps3.ckpt \
--inversion_loss reg_free \
--trans_robust True \
--iters 500 \
--append_path rob_l2eps3 \
--step_size 0.1

python -m human_nn_alignment.reg_free_loss \
--source_dataset imagenet \
--dataset random_0_1 \
--model resnet50 \
--batch_size 100 \
--checkpoint_path /NS/robustness_2/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/resnet-50-l2-eps3.ckpt \
--inversion_loss adv_alex_finetuned \
--iters 500 \
--append_path rob_l2eps3 \
--step_size 0.1

python -m human_nn_alignment.reg_free_loss \
--source_dataset imagenet \
--dataset random_0.5_2 \
--model resnet50 \
--batch_size 100 \
--checkpoint_path /NS/robustness_2/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/resnet-50-l2-eps3.ckpt \
--inversion_loss reg_free \
--trans_robust True \
--iters 500 \
--append_path rob_l2eps3 \
--step_size 0.1

python -m human_nn_alignment.reg_free_loss \
--source_dataset imagenet \
--dataset random_0.5_2 \
--model resnet50 \
--batch_size 100 \
--checkpoint_path /NS/robustness_2/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/resnet-50-l2-eps3.ckpt \
--inversion_loss adv_alex_finetuned \
--iters 500 \
--append_path rob_l2eps3 \
--step_size 1