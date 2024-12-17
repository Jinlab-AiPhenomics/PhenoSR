from pytorch_msssim import SSIM
from torch import nn as nn
from basicsr.utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class SSIMLoss(nn.Module):
    def __init__(self,
                 data_range=255,
                 size_average=True,
                 win_size=11,
                 win_sigma=1.5,
                 channel=3,
                 spatial_dims=2,
                 K=(0.01, 0.03),
                 nonnegative_ssim=False, ):
        super(SSIMLoss, self).__init__()
        self.ssim = SSIM(data_range=data_range, win_size=win_size, size_average=size_average,
                         win_sigma=win_sigma, channel=channel, spatial_dims=spatial_dims, K=K,
                         nonnegative_ssim=nonnegative_ssim)

    def forward(self, sr, hr):
        ssim = self.ssim(sr, hr)
        loss = 1 - ssim
        return loss
