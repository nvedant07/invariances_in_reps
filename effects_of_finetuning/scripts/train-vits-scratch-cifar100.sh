EPOCHS=150

# for MODEL in {"vit_base_patch32_224","vit_base_patch16_224","vit_small_patch32_224","vit_small_patch16_224"}
for MODEL in {"vit_base_patch32_224","vit_base_patch16_224"}
do
for bs in {128,256,512,1024}
do
python -m partially_inverted_reps.finetuning \
--source_dataset cifar100 \
--finetuning_dataset cifar100 \
--finetune_mode full \
--base_dir effects_of_finetuning \
--save_every 0 \
--model $MODEL \
--batch_size $bs \
--append_path scratch \
--epochs $EPOCHS \
--optimizer sgd \
--lr 0.001 \
--step_lr 20 \
--warmup_steps 100 \
--gradient_clipping 1.0

python -m partially_inverted_reps.finetuning \
--source_dataset cifar100 \
--finetuning_dataset cifar100 \
--finetune_mode full \
--base_dir effects_of_finetuning \
--save_every 0 \
--model $MODEL \
--batch_size $bs \
--append_path scratch \
--epochs $EPOCHS \
--optimizer sgd \
--lr 0.01 \
--step_lr 20 \
--warmup_steps 100 \
--gradient_clipping 1.0

python -m partially_inverted_reps.finetuning \
--source_dataset cifar100 \
--finetuning_dataset cifar100 \
--finetune_mode full \
--base_dir effects_of_finetuning \
--save_every 0 \
--model $MODEL \
--batch_size $bs \
--append_path scratch \
--epochs $EPOCHS \
--optimizer sgd \
--lr 0.1 \
--step_lr 20 \
--warmup_steps 100 \
--gradient_clipping 1.0

python -m partially_inverted_reps.finetuning \
--source_dataset cifar100 \
--finetuning_dataset cifar100 \
--finetune_mode full \
--base_dir effects_of_finetuning \
--save_every 0 \
--model $MODEL \
--batch_size $bs \
--append_path scratch \
--epochs $EPOCHS \
--optimizer sgd \
--lr 0.001 \
--step_lr 20

python -m partially_inverted_reps.finetuning \
--source_dataset cifar100 \
--finetuning_dataset cifar100 \
--finetune_mode full \
--base_dir effects_of_finetuning \
--save_every 0 \
--model $MODEL \
--batch_size $bs \
--append_path scratch \
--epochs $EPOCHS \
--optimizer sgd \
--lr 0.01 \
--step_lr 20

python -m partially_inverted_reps.finetuning \
--source_dataset cifar100 \
--finetuning_dataset cifar100 \
--finetune_mode full \
--base_dir effects_of_finetuning \
--save_every 0 \
--model $MODEL \
--batch_size $bs \
--append_path scratch \
--epochs $EPOCHS \
--optimizer sgd \
--lr 0.1 \
--step_lr 20
done
done