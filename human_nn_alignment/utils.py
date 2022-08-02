import torch
from human_nn_alignment.fft_image import rfft2d_freqs
from attack.losses import LPNormLossSingleModel, CompositeLoss, TVLoss, \
        BlurLoss, L1Loss, LpLoss, LpNormLossSingleModelPerceptual
from human_nn_alignment.random_dataset import RandomDataModule

def initialize_seed(input_size, seed, fft=False):
    ## if fft then seed must be initialzed in the fourier domain
    ## then descent will happen in fourier domain
    shape = (3,input_size,input_size) if not fft else \
        (3,*rfft2d_freqs(input_size, input_size).shape,2)
    if seed == 'super-noise':
        return torch.randn(*shape)
    if seed == 'white':
        return torch.ones(*shape)
    if seed == 'black':
        return torch.zeros(*shape)
    if seed == 'light-noise':
        return torch.randn(*shape) * 0.01

LOSSES_MAPPING = {
    'reg_free': LPNormLossSingleModel(lpnorm_type=2), 
    'freq': CompositeLoss(
        [LPNormLossSingleModel(lpnorm_type=2),
        TVLoss(beta=2.),
        BlurLoss(),
        L1Loss(constant=0.5)],
        weights=[10., 0.0005, 0.0005, 0.00005]
    ),
    'freq_tv': CompositeLoss(
        [LPNormLossSingleModel(lpnorm_type=2),
        TVLoss(beta=2.)],
        weights=[10., 0.0005]
    ),
    'freq_blur': CompositeLoss(
        [LPNormLossSingleModel(lpnorm_type=2),
        BlurLoss()],
        weights=[10., 0.0005]
    ),
    'freq_lp': CompositeLoss(
        [LPNormLossSingleModel(lpnorm_type=2),
        L1Loss(constant=0.5)],
        weights=[10., 0.00005]
    ),
    'freq_tv_l6': CompositeLoss(
        [LPNormLossSingleModel(lpnorm_type=2),
        TVLoss(beta=2.),
        LpLoss(p=6)],
        weights=[10., 0.0005, 1.]
    ), # to match https://arxiv.org/abs/1412.0035
    'adv_alex_finetuned': LpNormLossSingleModelPerceptual(lpips_model='alex', 
        lpips_model_path='/NS/twitter_archive2/work/vnanda/PerceptualSimilarity/'\
            'lpips/weights/v0.1/alex.pth', lpnorm_type=2, scaling_factor=10),
    'adv_alex_finetuned_seed': LpNormLossSingleModelPerceptual(lpips_model='alex', 
        lpips_model_path='/NS/twitter_archive2/work/vnanda/PerceptualSimilarity/'\
            'lpips/weights/v0.1/alex.pth', lpnorm_type=2, scaling_factor=-10),
    'adv_alex_imagenet': LpNormLossSingleModelPerceptual(lpips_model='alex', 
        lpips_model_path=None, lpnorm_type=2),
    'adv_alex_imagenet_seed': LpNormLossSingleModelPerceptual(lpips_model='alex', 
        lpips_model_path=None, lpnorm_type=2, scaling_factor=-10)
}

ADDITIONAL_DATAMODULES = {
    'random_0_1': RandomDataModule,
    'random_0.5_2': RandomDataModule,
}
