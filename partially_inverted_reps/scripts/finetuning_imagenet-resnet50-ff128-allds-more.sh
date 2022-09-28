for seed in {1..5}
do
for ft_ds in {"cifar10","cifar100","flowers","oxford-iiit-pets"}
do
for num_features in {4,10,20,40,65,75,100,120}
do
python -m partially_inverted_reps.finetuning \
--source_dataset imagenet \
--finetuning_dataset $ft_ds \
--finetune_mode linear \
--model resnet50_ff128 \
--batch_size 256 \
--checkpoint_path /NS/robustness_2/work/vnanda/invariances_in_reps/deep-learning-base/checkpoints/imagenet1k/r50_mrl0_e0_ff128.pt \
--append_path nonrob \
--epochs 50 \
--mode random \
--step_lr 10 \
--num_features $num_features \
--seed $seed
done
done
done