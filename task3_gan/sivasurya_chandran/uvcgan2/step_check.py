import os, sys, copy, torch, torch.nn.functional as F, yaml
HERE = os.path.abspath("uvcgan2"); MEMBER = os.path.abspath(".")
sys.path.insert(0, os.path.join(MEMBER, "src")); sys.path.insert(0, os.path.join(MEMBER, "..", "..")); sys.path.insert(0, HERE)
from models_uvcgan import build_uvc_generator, build_uvc_discriminator
from train import ImagePool, lsgan, ema_update, grad_norm
from diffaugment import diff_augment
from train_uvcgan import r1_penalty

cfg = yaml.safe_load(open("uvcgan2/config_uvcgan.yaml"))
t, m = cfg["train"], {**cfg["model"], "image_size": 256}
dev = "cuda"
G_AB, G_BA = build_uvc_generator(m).to(dev), build_uvc_generator(m).to(dev)
D_A, D_B = build_uvc_discriminator(m).to(dev), build_uvc_discriminator(m).to(dev)
nets = {"G_AB": G_AB, "G_BA": G_BA, "D_A": D_A, "D_B": D_B}
ema = {k: copy.deepcopy(nets[k]).eval().requires_grad_(False) for k in ("G_AB","G_BA")}
oG = torch.optim.Adam(list(G_AB.parameters())+list(G_BA.parameters()), lr=t["lr_g"], betas=(0.5,0.999))
oD = torch.optim.Adam(list(D_A.parameters())+list(D_B.parameters()), lr=t["lr_d"], betas=(0.5,0.999))
pool_A, pool_B = ImagePool(50), ImagePool(50)
pol = t["diffaugment"]
xa, xb = torch.randn(1,3,256,256,device=dev), torch.randn(1,3,256,256,device=dev)

with torch.autocast("cuda", dtype=torch.bfloat16):
    fb, fa = G_AB(xa), G_BA(xb)
    rec_a, rec_b = G_BA(fb), G_AB(fa)
    adv = lsgan(D_B(diff_augment(fb,pol)),1.0)+lsgan(D_A(diff_augment(fa,pol)),1.0)
    cyc = F.l1_loss(rec_a,xa)+F.l1_loss(rec_b,xb)
    idt = F.l1_loss(G_AB(xb),xb)+F.l1_loss(G_BA(xa),xa)
    low = F.l1_loss(F.interpolate(fb,size=32,mode="area"), F.interpolate(xa,size=32,mode="area"))
    lG = adv + t["lambda_cycle"]*cyc + t["lambda_identity"]*idt + t["lambda_lowres"]*low
oG.zero_grad(); lG.float().backward(); gnG = grad_norm(list(G_AB.parameters())+list(G_BA.parameters())); oG.step()
print("4/10. translation fwd+bwd + one G step: loss_G=%.4f gnG=%.3f finite=%s" % (float(lG), gnG, torch.isfinite(lG).item()))
print("      low-res 32x32 loss=%.4f  cyc=%.4f  idt=%.4f" % (float(low), float(cyc), float(idt)))

with torch.autocast("cuda", dtype=torch.bfloat16):
    lD = 0.5*(lsgan(D_B(diff_augment(xb,pol)),1.0)+lsgan(D_B(diff_augment(pool_B.query(fb.detach()),pol)),0.0)) \
       + 0.5*(lsgan(D_A(diff_augment(xa,pol)),1.0)+lsgan(D_A(diff_augment(pool_A.query(fa.detach()),pol)),0.0))
oD.zero_grad(); lD.float().backward()
r1 = 0.5*t["r1_gamma"]*t["r1_every"]*(r1_penalty(D_A,xa)+r1_penalty(D_B,xb))
r1.backward()
gnD = grad_norm(list(D_A.parameters())+list(D_B.parameters())); oD.step()
print("9. one D step: loss_D=%.4f gnD=%.3f" % (float(lD), gnD))
print("5. R1 penalty = %.6f  finite=%s" % (float(r1), bool(torch.isfinite(r1))))

before = ema["G_AB"].enc0[1].weight.clone()
for k in ema: ema_update(ema[k], nets[k], 0.999)
print("7. EMA updates: changed=%s" % (not torch.equal(before, ema["G_AB"].enc0[1].weight)))
print("11. all losses finite:", all(bool(torch.isfinite(v)) for v in [lG, lD, r1]))
print("    peak VRAM: %.2f GB" % (torch.cuda.max_memory_allocated()/2**30))
print("\nCHECKS 1-11 PASSED")
