"""
UVCGAN-v2-style generator and discriminator, written for this project.

Technical reference (ideas only; no code was copied or executed from it):
  Torbunov et al., "UVCGAN v2: An Improved Cycle-Consistent GAN for Unpaired Image-to-Image Translation",
  https://github.com/LS4GAN/uvcgan2
The surrounding training setup stays CycleGAN: two generators, two discriminators, unpaired data,
adversarial + cycle-consistency losses.

Generator (UVCGenerator): U-Net encoder -> residual bottleneck -> compact ViT bottleneck at 32x32 ->
decoder with skip connections and source-driven style modulation (AdaIN whose scale/shift come from a
style token derived from the *source* image, so the generator conditions on what it is translating).

Discriminator (SNPatchDiscriminator): PatchGAN with spectral normalisation on every conv, plus a
batch-statistics head that works at batch_size=1 by keeping a detached running feature cache
(no gradient path through the cache, so no stale gradients).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


# --------------------------------------------------------------------------------------- generator


class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.body = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(c, c, 3), nn.InstanceNorm2d(c, affine=True), nn.ReLU(True),
            nn.ReflectionPad2d(1), nn.Conv2d(c, c, 3), nn.InstanceNorm2d(c, affine=True),
        )

    def forward(self, x):
        return x + self.body(x)


class TransformerBlock(nn.Module):
    """Pre-norm transformer block (the compact ViT bottleneck)."""

    def __init__(self, dim, heads, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=drop, batch_first=True)
        self.n2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(),
                                 nn.Linear(int(dim * mlp_ratio), dim))

    def forward(self, x):
        h = self.n1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.mlp(self.n2(x))


class ViTBottleneck(nn.Module):
    """Patchify the 32x32 feature map, run a few transformer blocks, unpatchify.

    Token 0 is the style token: a learnable token plus a projection of the pooled *source* features,
    which is what makes the style source-driven. Its output is returned separately and drives the
    decoder's AdaIN modulation."""

    def __init__(self, ch, size=32, patch=2, dim=384, depth=4, heads=6):
        super().__init__()
        self.size, self.patch, self.dim = size, patch, dim
        self.grid = size // patch
        n_tok = self.grid * self.grid
        self.proj_in = nn.Conv2d(ch, dim, patch, patch)                 # patch embedding
        self.proj_out = nn.ConvTranspose2d(dim, ch, patch, patch)       # un-patchify
        self.pos = nn.Parameter(torch.zeros(1, n_tok + 1, dim))
        self.style_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.style_from_src = nn.Linear(ch, dim)                        # source-driven part of the style
        self.blocks = nn.ModuleList([TransformerBlock(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.style_token, std=0.02)

    def forward(self, x):
        b = x.size(0)
        src_style = self.style_from_src(x.mean(dim=(2, 3))).unsqueeze(1)   # (B,1,dim) from the source
        t = self.proj_in(x).flatten(2).transpose(1, 2)                      # (B, n_tok, dim)
        t = torch.cat([self.style_token.expand(b, -1, -1) + src_style, t], dim=1) + self.pos
        for blk in self.blocks:
            t = blk(t)
        t = self.norm(t)
        style, tokens = t[:, 0], t[:, 1:]
        feat = tokens.transpose(1, 2).reshape(b, self.dim, self.grid, self.grid)
        return self.proj_out(feat), style


class UVCGenerator(nn.Module):
    """U-Net + residual + ViT bottleneck with source-driven style modulation."""

    def __init__(self, ngf=64, n_res=4, vit_dim=384, vit_depth=4, vit_heads=6, img_size=256):
        super().__init__()
        c1, c2, c3, c4 = ngf, ngf * 2, ngf * 4, ngf * 6        # 64, 128, 256, 384
        self.enc0 = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(3, c1, 7),
                                  nn.InstanceNorm2d(c1, affine=True), nn.ReLU(True))                 # 256
        self.enc1 = nn.Sequential(nn.Conv2d(c1, c2, 4, 2, 1), nn.InstanceNorm2d(c2, affine=True), nn.ReLU(True))   # 128
        self.enc2 = nn.Sequential(nn.Conv2d(c2, c3, 4, 2, 1), nn.InstanceNorm2d(c3, affine=True), nn.ReLU(True))   # 64
        self.enc3 = nn.Sequential(nn.Conv2d(c3, c4, 4, 2, 1), nn.InstanceNorm2d(c4, affine=True), nn.ReLU(True))   # 32

        self.res = nn.Sequential(*[ResBlock(c4) for _ in range(n_res)])
        self.vit = ViTBottleneck(c4, size=img_size // 8, patch=2, dim=vit_dim, depth=vit_depth, heads=vit_heads)

        self.up3 = _AdaINUp(c4, c3, c3, vit_dim)   # 32 -> 64
        self.up2 = _AdaINUp(c3, c2, c2, vit_dim)   # 64 -> 128
        self.up1 = _AdaINUp(c2, c1, c1, vit_dim)   # 128 -> 256
        self.out = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(c1, 3, 7), nn.Tanh())

    def forward(self, x):
        e0 = self.enc0(x)
        e1 = self.enc1(e0)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        h = self.res(e3)
        h, style = self.vit(h)
        h = h + e3                                  # residual around the ViT bottleneck
        h = self.up3(h, e2, style)
        h = self.up2(h, e1, style)
        h = self.up1(h, e0, style)
        return self.out(h)


class _AdaINUp(nn.Module):
    """Upsample (nearest + conv), concatenate the skip, then AdaIN with style-driven scale/shift."""

    def __init__(self, c_in, c_skip, c_out, style_dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(c_in + c_skip, c_out, 3))
        self.norm = nn.InstanceNorm2d(c_out, affine=False)
        self.to_ss = nn.Linear(style_dim, c_out * 2)
        self.act = nn.ReLU(True)

    def forward(self, x, skip, style):
        x = self.conv(torch.cat([self.up(x), skip], dim=1))
        x = self.norm(x)
        scale, shift = self.to_ss(style).chunk(2, dim=1)
        x = x * (1 + scale.unsqueeze(-1).unsqueeze(-1)) + shift.unsqueeze(-1).unsqueeze(-1)
        return self.act(x)


# ----------------------------------------------------------------------------------- discriminator


class BatchStatHead(nn.Module):
    """Minibatch-standard-deviation feature that still works at batch_size=1.

    With B=1 the across-batch std is undefined, so we keep a running cache of feature mean/var as
    *buffers* updated under no_grad and used detached. The appended channel therefore carries no
    gradient path to earlier steps: no stale gradients, and nothing to backpropagate through time."""

    def __init__(self, channels, momentum=0.01):
        super().__init__()
        self.momentum = momentum
        self.register_buffer("running_mean", torch.zeros(channels))
        self.register_buffer("running_var", torch.ones(channels))

    def forward(self, x):
        if self.training:
            with torch.no_grad():                               # cache update never touches autograd
                m = x.mean(dim=(0, 2, 3))
                v = x.var(dim=(0, 2, 3), unbiased=False)
                self.running_mean.mul_(1 - self.momentum).add_(self.momentum * m)
                self.running_var.mul_(1 - self.momentum).add_(self.momentum * v)
        # deviation of the current features from the cached statistics, as one extra constant channel
        dev = (x - self.running_mean.detach().view(1, -1, 1, 1)).pow(2)
        dev = (dev / (self.running_var.detach().view(1, -1, 1, 1) + 1e-8)).mean(dim=1, keepdim=True).sqrt()
        return torch.cat([x, dev.detach() if not self.training else dev], dim=1)


class SNPatchDiscriminator(nn.Module):
    """70x70 PatchGAN with spectral normalisation and a batch-statistics head before the final conv."""

    def __init__(self, ndf=64, n_layers=3, use_batchstat=True):
        super().__init__()
        def sn(c):
            return spectral_norm(c)

        layers = [sn(nn.Conv2d(3, ndf, 4, 2, 1)), nn.LeakyReLU(0.2, True)]
        mult = 1
        for i in range(1, n_layers):
            prev, mult = mult, min(2 ** i, 8)
            layers += [sn(nn.Conv2d(ndf * prev, ndf * mult, 4, 2, 1)), nn.LeakyReLU(0.2, True)]
        prev, mult = mult, min(2 ** n_layers, 8)
        layers += [sn(nn.Conv2d(ndf * prev, ndf * mult, 4, 1, 1)), nn.LeakyReLU(0.2, True)]
        self.body = nn.Sequential(*layers)
        self.bstat = BatchStatHead(ndf * mult) if use_batchstat else None
        self.head = sn(nn.Conv2d(ndf * mult + (1 if use_batchstat else 0), 1, 4, 1, 1))

    def forward(self, x):
        h = self.body(x)
        if self.bstat is not None:
            h = self.bstat(h)
        return self.head(h)


def build_uvc_generator(mcfg):
    return UVCGenerator(ngf=mcfg.get("ngf", 64), n_res=mcfg.get("n_resblocks", 4),
                        vit_dim=mcfg.get("vit_dim", 384), vit_depth=mcfg.get("vit_depth", 4),
                        vit_heads=mcfg.get("vit_heads", 6), img_size=mcfg.get("image_size", 256))


def build_uvc_discriminator(mcfg):
    return SNPatchDiscriminator(ndf=mcfg.get("ndf", 64), n_layers=mcfg.get("d_layers", 3),
                                use_batchstat=mcfg.get("batchstat", True))
