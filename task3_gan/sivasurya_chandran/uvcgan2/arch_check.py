import torch, sys
from models_uvcgan import build_uvc_generator, build_uvc_discriminator

dev = "cuda" if torch.cuda.is_available() else "cpu"
m = {"ngf": 64, "ndf": 64, "n_resblocks": 4, "vit_dim": 384, "vit_depth": 4, "vit_heads": 6, "image_size": 256}

G_AB, G_BA = build_uvc_generator(m).to(dev), build_uvc_generator(m).to(dev)
D_A, D_B = build_uvc_discriminator(m).to(dev), build_uvc_discriminator(m).to(dev)
print("1. instantiate: OK")
print("   G params: %.2fM | D params: %.2fM" % (
    sum(p.numel() for p in G_AB.parameters())/1e6, sum(p.numel() for p in D_A.parameters())/1e6))

x = torch.randn(1, 3, 256, 256, device=dev)
with torch.no_grad():
    fb, fa = G_AB(x), G_BA(x)
    da, db = D_A(x), D_B(x)
print("2. forward both directions: G_AB(x)=%s  G_BA(x)=%s" % (tuple(fb.shape), tuple(fa.shape)))
print("   D output (patch map):", tuple(da.shape))
assert fb.shape == x.shape and fa.shape == x.shape, "generator must preserve shape"
assert fb.min() >= -1.001 and fb.max() <= 1.001, "tanh range"

# spectral norm present?
sn = [n for n, mod in D_A.named_modules() if hasattr(mod, "weight_orig") or any(
    h.__class__.__name__ == "SpectralNorm" for h in getattr(mod, "_forward_pre_hooks", {}).values())]
print("6. spectral-norm'd modules in D_A:", len(sn))
assert len(sn) >= 4, "spectral norm missing"

# batch-stat cache: must not create a graph edge through the running buffers
D_A.train()
x1 = torch.randn(1, 3, 256, 256, device=dev, requires_grad=True)
out1 = D_A(x1).mean(); out1.backward()
rm_before = D_A.bstat.running_mean.clone()
x2 = torch.randn(1, 3, 256, 256, device=dev, requires_grad=True)
out2 = D_A(x2).mean(); out2.backward()   # would error if a stale graph were retained
print("8. batch-stat cache: two sequential backwards OK; buffers require_grad=%s, cache moved=%s" % (
    D_A.bstat.running_mean.requires_grad, not torch.equal(rm_before, D_A.bstat.running_mean)))
assert D_A.bstat.running_mean.requires_grad is False
print("   batch_size=1 handled (no NaN in output):", bool(torch.isfinite(out2).all()))
print("\nALL ARCHITECTURE CHECKS PASSED")
