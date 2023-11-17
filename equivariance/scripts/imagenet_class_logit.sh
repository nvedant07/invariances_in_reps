python -m equivariance.stretch_class_logits \
--dataset imagenet \
--model1 resnet18 \
--model2 resnet50 \
--iters 500 \
--vector_type class_logit \
--vector_class_idx 10 \
--step_size 1 \
--batch_size 400 \
--total_imgs 10000 \
--append_path correct

python -m equivariance.stretch_class_logits \
--dataset imagenet \
--model1 resnet50 \
--model2 resnet18 \
--iters 500 \
--vector_type class_logit \
--vector_class_idx 10 \
--step_size 1 \
--batch_size 100 \
--total_imgs 10000 \
--append_path correct


python -m equivariance.stretch_class_logits \
--dataset imagenet \
--model1 resnet18 \
--checkpoint1_path /NS/robustness_1/work/vnanda/CKA-Centered-Kernel-Alignment/checkpoints/imagenet/resnet18/robust/l2/eps3/iters7da_True/checkpoint_rand_seed_1.pt.best \
--model2 resnet50 \
--checkpoint2_path /NS/robustness_2/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/resnet-50-l2-eps3.ckpt \
--iters 500 \
--append_path robl2eps3_seed_1_correct \
--vector_type class_logit \
--vector_class_idx 10 \
--step_size 1 \
--batch_size 400 \
--total_imgs 10000


python -m equivariance.stretch_class_logits \
--dataset imagenet \
--model2 resnet18 \
--checkpoint2_path /NS/robustness_1/work/vnanda/CKA-Centered-Kernel-Alignment/checkpoints/imagenet/resnet18/robust/l2/eps3/iters7da_True/checkpoint_rand_seed_1.pt.best \
--model1 resnet50 \
--checkpoint1_path /NS/robustness_2/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/resnet-50-l2-eps3.ckpt \
--iters 500 \
--append_path robl2eps3_seed_1_correct \
--vector_type class_logit \
--vector_class_idx 10 \
--step_size 1 \
--batch_size 100 \
--total_imgs 10000
