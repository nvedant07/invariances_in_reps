for seed in {1..5}
do
python -m partially_inverted_reps.partial_inversion \
--source_dataset imagenet \
--dataset imagenet \
--model resnet50 \
--batch_size 50 \
--type regularized \
--append_path robustl2eps3regularized \
--checkpoint_path /NS/robustness_2/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/resnet-50-l2-eps3.ckpt \
--iters 1000 \
--mode random \
--fraction 0.0005 \
--seed $seed \
--step_size 1

python -m partially_inverted_reps.partial_inversion \
--source_dataset imagenet \
--dataset imagenet \
--model resnet50 \
--batch_size 50 \
--type regularized \
--append_path robustl2eps3regularized \
--checkpoint_path /NS/robustness_2/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/resnet-50-l2-eps3.ckpt \
--iters 1000 \
--mode random \
--fraction 0.001 \
--seed $seed \
--step_size 1

python -m partially_inverted_reps.partial_inversion \
--source_dataset imagenet \
--dataset imagenet \
--model resnet50 \
--batch_size 50 \
--type regularized \
--append_path robustl2eps3regularized \
--checkpoint_path /NS/robustness_2/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/resnet-50-l2-eps3.ckpt \
--iters 1000 \
--mode random \
--fraction 0.002 \
--seed $seed \
--step_size 0.1

python -m partially_inverted_reps.partial_inversion \
--source_dataset imagenet \
--dataset imagenet \
--model resnet50 \
--batch_size 50 \
--type regularized \
--append_path robustl2eps3regularized \
--checkpoint_path /NS/robustness_2/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/resnet-50-l2-eps3.ckpt \
--iters 1000 \
--mode random \
--fraction 0.003 \
--seed $seed \
--step_size 1

python -m partially_inverted_reps.partial_inversion \
--source_dataset imagenet \
--dataset imagenet \
--model resnet50 \
--batch_size 50 \
--type regularized \
--append_path robustl2eps3regularized \
--checkpoint_path /NS/robustness_2/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/resnet-50-l2-eps3.ckpt \
--iters 1000 \
--mode random \
--fraction 0.004 \
--seed $seed \
--step_size 1

python -m partially_inverted_reps.partial_inversion \
--source_dataset imagenet \
--dataset imagenet \
--model resnet50 \
--batch_size 50 \
--type regularized \
--append_path robustl2eps3regularized \
--checkpoint_path /NS/robustness_2/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/resnet-50-l2-eps3.ckpt \
--iters 1000 \
--mode random \
--fraction 0.005 \
--seed $seed \
--step_size 1

python -m partially_inverted_reps.partial_inversion \
--source_dataset imagenet \
--dataset imagenet \
--model resnet50 \
--batch_size 50 \
--type regularized \
--append_path robustl2eps3regularized \
--checkpoint_path /NS/robustness_2/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/resnet-50-l2-eps3.ckpt \
--iters 1000 \
--mode random \
--fraction 0.01 \
--seed $seed \
--step_size 1

python -m partially_inverted_reps.partial_inversion \
--source_dataset imagenet \
--dataset imagenet \
--model resnet50 \
--batch_size 50 \
--type regularized \
--append_path robustl2eps3regularized \
--checkpoint_path /NS/robustness_2/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/resnet-50-l2-eps3.ckpt \
--iters 1000 \
--mode random \
--fraction 0.05 \
--seed $seed \
--step_size 1
done