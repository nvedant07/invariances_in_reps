for seed in {1..5}
do
python -m partially_inverted_reps.feature_attacks \
--source_dataset imagenet \
--dataset imagenet \
--model resnet50 \
--batch_size 50 \
--type reg_free \
--append_path robustl2eps3 \
--seed_type same \
--checkpoint_path /NS/robustness_2/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/resnet-50-l2-eps3.ckpt \
--iters 1000 \
--mode max \
--fraction 0.0005 \
--seed $seed \
--step_size 5

python -m partially_inverted_reps.feature_attacks \
--source_dataset imagenet \
--dataset imagenet \
--model resnet50 \
--batch_size 50 \
--type reg_free \
--append_path robustl2eps3 \
--seed_type same \
--checkpoint_path /NS/robustness_2/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/resnet-50-l2-eps3.ckpt \
--iters 1000 \
--mode max \
--fraction 0.001 \
--seed $seed \
--step_size 5

python -m partially_inverted_reps.feature_attacks \
--source_dataset imagenet \
--dataset imagenet \
--model resnet50 \
--batch_size 50 \
--type reg_free \
--append_path robustl2eps3 \
--seed_type same \
--checkpoint_path /NS/robustness_2/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/resnet-50-l2-eps3.ckpt \
--iters 1000 \
--mode max \
--fraction 0.002 \
--seed $seed \
--step_size 5

python -m partially_inverted_reps.feature_attacks \
--source_dataset imagenet \
--dataset imagenet \
--model resnet50 \
--batch_size 50 \
--type reg_free \
--append_path robustl2eps3 \
--seed_type same \
--checkpoint_path /NS/robustness_2/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/resnet-50-l2-eps3.ckpt \
--iters 1000 \
--mode max \
--fraction 0.003 \
--seed $seed \
--step_size 5

python -m partially_inverted_reps.feature_attacks \
--source_dataset imagenet \
--dataset imagenet \
--model resnet50 \
--batch_size 50 \
--type reg_free \
--append_path robustl2eps3 \
--seed_type same \
--checkpoint_path /NS/robustness_2/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/resnet-50-l2-eps3.ckpt \
--iters 1000 \
--mode max \
--fraction 0.004 \
--seed $seed \
--step_size 1

python -m partially_inverted_reps.feature_attacks \
--source_dataset imagenet \
--dataset imagenet \
--model resnet50 \
--batch_size 50 \
--type reg_free \
--append_path robustl2eps3 \
--seed_type same \
--checkpoint_path /NS/robustness_2/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/resnet-50-l2-eps3.ckpt \
--iters 1000 \
--mode max \
--fraction 0.005 \
--seed $seed \
--step_size 1

python -m partially_inverted_reps.feature_attacks \
--source_dataset imagenet \
--dataset imagenet \
--model resnet50 \
--batch_size 50 \
--type reg_free \
--append_path robustl2eps3 \
--seed_type same \
--checkpoint_path /NS/robustness_2/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/resnet-50-l2-eps3.ckpt \
--iters 1000 \
--mode max \
--fraction 0.01 \
--seed $seed \
--step_size 1

python -m partially_inverted_reps.feature_attacks \
--source_dataset imagenet \
--dataset imagenet \
--model resnet50 \
--batch_size 50 \
--type reg_free \
--append_path robustl2eps3 \
--seed_type same \
--checkpoint_path /NS/robustness_2/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/resnet-50-l2-eps3.ckpt \
--iters 1000 \
--mode max \
--fraction 0.05 \
--seed $seed \
--step_size 1

python -m partially_inverted_reps.feature_attacks \
--source_dataset imagenet \
--dataset imagenet \
--model resnet50 \
--batch_size 50 \
--type reg_free \
--append_path robustl2eps3 \
--seed_type same \
--checkpoint_path /NS/robustness_2/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/resnet-50-l2-eps3.ckpt \
--iters 1000 \
--mode max \
--fraction 0.1 \
--seed $seed \
--step_size 1

python -m partially_inverted_reps.feature_attacks \
--source_dataset imagenet \
--dataset imagenet \
--model resnet50 \
--batch_size 50 \
--type reg_free \
--append_path robustl2eps3 \
--seed_type same \
--checkpoint_path /NS/robustness_2/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/resnet-50-l2-eps3.ckpt \
--iters 1000 \
--mode max \
--fraction 0.3 \
--seed $seed \
--step_size 1

python -m partially_inverted_reps.feature_attacks \
--source_dataset imagenet \
--dataset imagenet \
--model resnet50 \
--batch_size 50 \
--type reg_free \
--append_path robustl2eps3 \
--seed_type same \
--checkpoint_path /NS/robustness_2/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/resnet-50-l2-eps3.ckpt \
--iters 1000 \
--mode max \
--fraction 0.5 \
--seed $seed \
--step_size 1

python -m partially_inverted_reps.feature_attacks \
--source_dataset imagenet \
--dataset imagenet \
--model resnet50 \
--batch_size 50 \
--type reg_free \
--append_path robustl2eps3 \
--seed_type same \
--checkpoint_path /NS/robustness_2/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/resnet-50-l2-eps3.ckpt \
--iters 1000 \
--mode max \
--fraction 0.8 \
--seed $seed \
--step_size 1

python -m partially_inverted_reps.feature_attacks \
--source_dataset imagenet \
--dataset imagenet \
--model resnet50 \
--batch_size 50 \
--type reg_free \
--append_path robustl2eps3 \
--seed_type same \
--checkpoint_path /NS/robustness_2/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/resnet-50-l2-eps3.ckpt \
--iters 1000 \
--mode max \
--fraction 1 \
--seed $seed \
--step_size 1
done