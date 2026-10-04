"""
CycleGAN: two generators (ResNet-based) and two PatchGAN discriminators
(optionally multi-scale: the same PatchGAN design applied at several resolutions).
"""
import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    """ResNet generator: c7s1-64, d128, d256, R256 x n, u128, u64, c7s1-3."""

    def __init__(self, ngf: int = 64, n_resblocks: int = 6, upsample: str = "deconv"):
        super().__init__()
        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, ngf, 7),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True),
        ]

        ch = ngf
        for _ in range(2):
            layers += [
                nn.Conv2d(ch, ch * 2, 3, stride=2, padding=1),
                nn.InstanceNorm2d(ch * 2),
                nn.ReLU(inplace=True),
            ]
            ch *= 2

        layers += [ResidualBlock(ch) for _ in range(n_resblocks)]

        for _ in range(2):
            if upsample == "resize_conv":
                # nearest-neighbour resize then conv: avoids the checkerboard artifacts of strided
                # transposed convolutions (Odena et al., 2016, "Deconvolution and Checkerboard Artifacts")
                up = [nn.Upsample(scale_factor=2, mode="nearest"), nn.ReflectionPad2d(1), nn.Conv2d(ch, ch // 2, 3)]
            else:
                up = [nn.ConvTranspose2d(ch, ch // 2, 3, stride=2, padding=1, output_padding=1)]
            layers += up + [nn.InstanceNorm2d(ch // 2), nn.ReLU(inplace=True)]
            ch //= 2

        layers += [nn.ReflectionPad2d(3), nn.Conv2d(ch, 3, 7), nn.Tanh()]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def build_generator(mcfg: dict) -> Generator:
    """Generator from the config's model section (v1 configs have no `upsample` key -> transposed convs)."""
    return Generator(mcfg["ngf"], mcfg["n_resblocks"], mcfg.get("upsample", "deconv"))


def init_weights(module):
    """N(0, 0.02) init used in the CycleGAN paper."""
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(module.weight, 0.0, 0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def count_params(*modules) -> int:
    return sum(p.numel() for m in modules for p in m.parameters())


class PatchDiscriminator(nn.Module):
    """70x70 PatchGAN discriminator."""

    def __init__(self, ndf: int = 64):
        super().__init__()

        def block(in_c, out_c, stride=2, norm=True):
            layers = [nn.Conv2d(in_c, out_c, 4, stride=stride, padding=1)]
            if norm:
                layers.append(nn.InstanceNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.net = nn.Sequential(
            *block(3, ndf, norm=False),
            *block(ndf, ndf * 2),
            *block(ndf * 2, ndf * 4),
            *block(ndf * 4, ndf * 8, stride=1),
            nn.Conv2d(ndf * 8, 1, 4, stride=1, padding=1),
        )

    def forward(self, x):
        return self.net(x)


class MultiScaleDiscriminator(nn.Module):
    """One discriminator made of PatchGANs at full, 1/2, 1/4 ... resolution (Wang et al. 2018, pix2pixHD).
    A 70x70 patch at half resolution covers 140x140 input pixels, so the coarser scale judges composition
    and colour layout while the full-resolution scale judges brushstroke texture. Returns one map per scale."""

    def __init__(self, ndf: int = 64, n_scales: int = 2):
        super().__init__()
        self.scales = nn.ModuleList(PatchDiscriminator(ndf) for _ in range(n_scales))
        self.down = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)

    def forward(self, x):
        outs = []
        for i, d in enumerate(self.scales):
            if i:
                x = self.down(x)
            outs.append(d(x))
        return outs


def build_discriminator(mcfg: dict) -> nn.Module:
    n = mcfg.get("d_scales", 1)
    return PatchDiscriminator(mcfg["ndf"]) if n == 1 else MultiScaleDiscriminator(mcfg["ndf"], n)
