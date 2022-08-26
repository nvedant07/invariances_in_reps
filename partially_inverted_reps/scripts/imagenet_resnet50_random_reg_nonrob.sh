for seed in {1..5}
do
python -m partially_inverted_reps.partial_inversion \
--source_dataset imagenet \
--dataset imagenet \
--model resnet50 \
--batch_size 50 \
--append_path nonrobustregularized \
--iters 1000 \
--mode random \
--type regularized \
--fraction 0.0005 \
--seed $seed \
--step_size 1

python -m partially_inverted_reps.partial_inversion \
--source_dataset imagenet \
--dataset imagenet \
--model resnet50 \
--batch_size 50 \
--append_path nonrobustregularized \
--iters 1000 \
--mode random \
--type regularized \
--fraction 0.001 \
--seed $seed \
--step_size 1

python -m partially_inverted_reps.partial_inversion \
--source_dataset imagenet \
--dataset imagenet \
--model resnet50 \
--batch_size 50 \
--append_path nonrobustregularized \
--iters 1000 \
--mode random \
--type regularized \
--fraction 0.002 \
--seed $seed \
--step_size 1

python -m partially_inverted_reps.partial_inversion \
--source_dataset imagenet \
--dataset imagenet \
--model resnet50 \
--batch_size 50 \
--append_path nonrobustregularized \
--iters 1000 \
--mode random \
--type regularized \
--fraction 0.003 \
--seed $seed \
--step_size 1

python -m partially_inverted_reps.partial_inversion \
--source_dataset imagenet \
--dataset imagenet \
--model resnet50 \
--batch_size 50 \
--append_path nonrobustregularized \
--iters 1000 \
--mode random \
--type regularized \
--fraction 0.004 \
--seed $seed \
--step_size 1

python -m partially_inverted_reps.partial_inversion \
--source_dataset imagenet \
--dataset imagenet \
--model resnet50 \
--batch_size 50 \
--append_path nonrobustregularized \
--iters 1000 \
--mode random \
--type regularized \
--fraction 0.005 \
--seed $seed \
--step_size 1

python -m partially_inverted_reps.partial_inversion \
--source_dataset imagenet \
--dataset imagenet \
--model resnet50 \
--batch_size 50 \
--append_path nonrobustregularized \
--iters 1000 \
--mode random \
--type regularized \
--fraction 0.01 \
--seed $seed \
--step_size 1

python -m partially_inverted_reps.partial_inversion \
--source_dataset imagenet \
--dataset imagenet \
--model resnet50 \
--batch_size 50 \
--append_path nonrobustregularized \
--iters 1000 \
--mode random \
--type regularized \
--fraction 0.05 \
--seed $seed \
--step_size 1

python -m partially_inverted_reps.partial_inversion \
--source_dataset imagenet \
--dataset imagenet \
--model resnet50 \
--batch_size 50 \
--type regularized \
--append_path nonrobustregularized \
--iters 1000 \
--mode random \
--fraction 0.1 \
--seed $seed \
--step_size 1

python -m partially_inverted_reps.partial_inversion \
--source_dataset imagenet \
--dataset imagenet \
--model resnet50 \
--batch_size 50 \
--type regularized \
--append_path nonrobustregularized \
--iters 1000 \
--mode random \
--fraction 0.3 \
--seed $seed \
--step_size 1

python -m partially_inverted_reps.partial_inversion \
--source_dataset imagenet \
--dataset imagenet \
--model resnet50 \
--batch_size 50 \
--type regularized \
--append_path nonrobustregularized \
--iters 1000 \
--mode random \
--fraction 0.5 \
--seed $seed \
--step_size 1

python -m partially_inverted_reps.partial_inversion \
--source_dataset imagenet \
--dataset imagenet \
--model resnet50 \
--batch_size 50 \
--type regularized \
--append_path nonrobustregularized \
--iters 1000 \
--mode random \
--fraction 0.8 \
--seed $seed \
--step_size 1

python -m partially_inverted_reps.partial_inversion \
--source_dataset imagenet \
--dataset imagenet \
--model resnet50 \
--batch_size 50 \
--type regularized \
--append_path nonrobustregularized \
--iters 1000 \
--mode random \
--fraction 0.9 \
--seed $seed \
--step_size 1
done