import torch
import torch.nn as nn
from basicsr.archs.hrnet_arch import HRnet
from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.phenosr_coarsenet_arch import (
    PhenoSRcoarseNet,
    compute_require_refine,
)
from basicsr.archs.phenosr_refinednet_arch import PhenoSRrefinedNet
from basicsr.archs.arch_util import trunc_normal_


@ARCH_REGISTRY.register()
class PhenoSR(nn.Module):
    def __init__(
        self,
        img_size,
        patch_size=1,
        in_chans=3,
        embed_dim=96,
        coarse_depths=(6, 6, 6, 6),
        coarse_num_heads=(6, 6, 6, 6),
        refine_depths=(6, 6, 6, 6),
        refine_num_heads=(6, 6, 6, 6),
        window_size=8,
        mlp_ratio=2.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        ape=False,
        patch_norm=True,
        use_checkpoint=False,
        upscale=2,
        img_range=1.0,
        resi_connection="1conv",
        num_classes=2,
        seg_dim=16,
        threshold=0.05,
        seg_model_path=None,
        **kwargs
    ):
        super(PhenoSR, self).__init__()
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)

        self.coarse = PhenoSRcoarseNet(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depths=coarse_depths,
            num_heads=coarse_num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            ape=ape,
            patch_norm=patch_norm,
            use_checkpoint=use_checkpoint,
            upscale=upscale,
            img_range=img_range,
            resi_connection=resi_connection,
            **kwargs
        )
        self.seg_model = HRnet(num_classes, backbone="hrnetv2_w32", pretrained=False)
        self.refine = PhenoSRrefinedNet(
            img_size=img_size,
            patch_size=1,
            in_chans=3,
            embed_dim=embed_dim,
            depths=refine_depths,
            num_heads=refine_num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            ape=ape,
            patch_norm=patch_norm,
            use_checkpoint=use_checkpoint,
            upscale=4,
            img_range=1,
            resi_connection="1conv",
            num_classes=num_classes,
            seg_dim=seg_dim,
            num_patches=4096,
            **kwargs
        )
        self.threshold = threshold
        self.apply(self._init_weights)

        checkpoint = torch.load(seg_model_path)
        self.seg_model.load_state_dict(checkpoint, strict=True)

    def get_seg_feature(self, x):
        with torch.no_grad():
            seg_map, seg_feat = self.seg_model(x)
        return seg_map, seg_feat

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, current_iter=None, is_training=True):
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range
        if is_training:
            coarse_sr, coarse_sr_feature, x_conv_first, x_size = self.coarse(x)
            coarse_sr_seg_map, coarse_sr_seg_feature = self.get_seg_feature(x)
            refined_sr = self.refine(
                x_size, coarse_sr_feature, coarse_sr_seg_feature, x_conv_first
            )
            refined_sr = refined_sr / self.img_range + self.mean
            coarse_sr = coarse_sr / self.img_range + self.mean
            return coarse_sr, refined_sr, coarse_sr_seg_map, current_iter
        else:
            coarse_sr, coarse_sr_feature, x_conv_first, x_size = self.coarse(x)
            coarse_sr_seg_map, coarse_sr_seg_feature = self.get_seg_feature(coarse_sr)
            require_refine, se_score = compute_require_refine(
                self.threshold, coarse_sr, coarse_sr_seg_map
            )
            if not require_refine:
                coarse_sr = coarse_sr / self.img_range + self.mean
                return coarse_sr, coarse_sr_seg_map, require_refine, se_score
            else:
                refined_sr = self.refine(
                    x_size, coarse_sr_feature, coarse_sr_seg_feature, x_conv_first
                )
                refined_sr = refined_sr / self.img_range + self.mean
                return refined_sr, coarse_sr_seg_map, require_refine, se_score
