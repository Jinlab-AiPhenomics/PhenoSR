import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange


def to_3d(x):
    return rearrange(x, "b c h w -> b (h w) c")


def to_4d(x, h, w):
    return rearrange(x, "b (h w) c -> b c h w", h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == "BiasFree":
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in_R = nn.Conv2d(dim, hidden_features, kernel_size=1, bias=bias)
        self.project_in_S = nn.Conv2d(dim, hidden_features, kernel_size=1, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        x = self.project_in_R(x)
        y = self.project_in_S(y)
        x = F.gelu(x) * F.gelu(y)
        x = self.project_out(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(
            dim * 2,
            dim * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim * 2,
            bias=bias,
        )
        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(
            dim, dim, kernel_size=3, stride=1, padding=1, bias=bias
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        b, c, h, w = x.shape

        kv = self.kv_dwconv(self.kv(x))
        k, v = kv.chunk(2, dim=1)
        q = self.q_dwconv(self.q(y))

        q = rearrange(q, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        k = rearrange(k, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        v = rearrange(v, "b (head c) h w -> b head c (h w)", head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = attn @ v

        out = rearrange(
            out, "b head c (h w) -> b (head c) h w", head=self.num_heads, h=h, w=w
        )

        out = self.project_out(out)
        return out


class SAFMBlock(nn.Module):
    def __init__(
        self,
        dim_2,
        dim,
        num_heads=2,
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type="WithBias",
    ):
        super(SAFMBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, input_R, input_S):
        input_R = self.norm1(input_R)
        input_S = self.norm1(input_S)
        input_R = input_R + self.attn(input_R, input_S)
        input_R = input_R + self.ffn(self.norm2(input_R), self.norm2(input_S))

        return input_R


class PreFuseBlock(nn.Module):
    def __init__(self, img_size=64, seg_dim=32, embed_dim=180):
        super(PreFuseBlock, self).__init__()
        self.seg_norm = LayerNorm(seg_dim, "WithBias")
        self.sr_norm = LayerNorm(embed_dim, "WithBias")
        self.seg_conv = nn.Conv2d(seg_dim, seg_dim, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.process_conv = nn.Conv2d(seg_dim + embed_dim, embed_dim, 1, 1, 0)

    def forward(self, sr_feat, seg_feat):
        seg_feat = F.interpolate(seg_feat, [sr_feat.shape[2], sr_feat.shape[3]])
        seg_feat = self.lrelu(self.seg_norm(self.seg_conv(seg_feat)))
        sr_feat = self.lrelu(self.sr_norm(sr_feat))
        res = torch.cat([seg_feat, sr_feat], dim=1)
        res = self.process_conv(res)
        return res
