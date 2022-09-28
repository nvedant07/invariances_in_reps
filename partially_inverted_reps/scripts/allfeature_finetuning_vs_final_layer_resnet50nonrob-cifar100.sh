for seed in {1..5}
do
for num_fts in {20,102,204,409,614,1024,1638,1843,2048}
do
python -m partially_inverted_reps.finetuning_all_features \
--source_dataset imagenet \
--finetuning_dataset cifar100 \
--finetune_mode linear \
--model resnet50 \
--batch_size 256 \
--append_path nonrob \
--epochs 50 \
--step_lr 10 \
--save_every 0 \
--mode random \
--num_features $num_fts \
--seed $seed
done
done