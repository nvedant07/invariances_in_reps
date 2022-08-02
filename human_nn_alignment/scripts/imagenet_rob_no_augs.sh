python -m human_nn_alignment.reg_free_loss \
--dataset imagenet \
--model resnet50 \
--append_path freerob_linf \
--checkpoint_path /NS/robustness_2/work/vnanda/FreeAdversarialTraining/checkpoints/imagenet/linf/eps4.0/resnet50-data-aug_best.pth \
--inversion_loss reg_free

python -m human_nn_alignment.reg_free_loss \
--dataset imagenet \
--model resnet18 \
--append_path freerob_linf \
--checkpoint_path /NS/robustness_2/work/vnanda/FreeAdversarialTraining/checkpoints/imagenet/linf/eps4.0/resnet18-data-aug_best.pth \
--inversion_loss reg_free


python -m human_nn_alignment.reg_free_loss \
--dataset imagenet \
--model resnet50 \
--append_path freerob_linf_noaug \
--checkpoint_path /NS/robustness_2/work/vnanda/FreeAdversarialTraining/checkpoints/imagenet/linf/eps4.0/resnet50-no-data-aug.pth \
--inversion_loss reg_free

python -m human_nn_alignment.reg_free_loss \
--dataset imagenet \
--model resnet18 \
--append_path freerob_linf_noaug \
--checkpoint_path /NS/robustness_2/work/vnanda/FreeAdversarialTraining/checkpoints/imagenet/linf/eps4.0/resnet18-no-data-aug.pth \
--inversion_loss reg_free

