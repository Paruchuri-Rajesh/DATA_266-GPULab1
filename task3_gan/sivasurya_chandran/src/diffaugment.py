"""
Differentiable Augmentation (Zhao et al., NeurIPS 2020, "Differentiable Augmentation for
Data-Efficient GAN Training"), ported from the authors' PyTorch reference code.

The same random augmentation family is applied to every image the discriminator sees, real and
fake, in both the D update and the G update. Because the ops are differentiable, G still gets
gradients through them. With only ~300 Monet paintings, this stops D from memorising the
training paintings. It only changes what D sees; the generator's outputs are never augmented.
Inputs are in [-1, 1].
"""
import torch
import torch.nn.functional as F


def rand_brightness(x):
    return x + (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) - 0.5)


def rand_saturation(x):
    mean = x.mean(dim=1, keepdim=True)
    return (x - mean) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) * 2) + mean


def rand_contrast(x):
    mean = x.mean(dim=[1, 2, 3], keepdim=True)
    return (x - mean) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) + 0.5) + mean


def rand_translation(x, ratio=0.125):
    shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    tx = torch.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device)
    ty = torch.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device)
    gb, gx, gy = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(x.size(2), dtype=torch.long, device=x.device),
        torch.arange(x.size(3), dtype=torch.long, device=x.device),
        indexing="ij",
    )
    gx = torch.clamp(gx + tx + 1, 0, x.size(2) + 1)
    gy = torch.clamp(gy + ty + 1, 0, x.size(3) + 1)
    x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    return x_pad.permute(0, 2, 3, 1).contiguous()[gb, gx, gy].permute(0, 3, 1, 2)


def rand_cutout(x, ratio=0.5):
    cut = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    ox = torch.randint(0, x.size(2) + (1 - cut[0] % 2), size=[x.size(0), 1, 1], device=x.device)
    oy = torch.randint(0, x.size(3) + (1 - cut[1] % 2), size=[x.size(0), 1, 1], device=x.device)
    gb, gx, gy = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(cut[0], dtype=torch.long, device=x.device),
        torch.arange(cut[1], dtype=torch.long, device=x.device),
        indexing="ij",
    )
    gx = torch.clamp(gx + ox - cut[0] // 2, min=0, max=x.size(2) - 1)
    gy = torch.clamp(gy + oy - cut[1] // 2, min=0, max=x.size(3) - 1)
    mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    mask[gb, gx, gy] = 0
    return x * mask.unsqueeze(1)


AUGMENT_FNS = {
    "color": [rand_brightness, rand_saturation, rand_contrast],
    "translation": [rand_translation],
    "cutout": [rand_cutout],
}


def diff_augment(x, policy: str = ""):
    """policy: comma-separated subset of color,translation,cutout ("" = identity)."""
    for p in filter(None, policy.split(",")):
        for fn in AUGMENT_FNS[p.strip()]:
            x = fn(x)
    return x.contiguous()
