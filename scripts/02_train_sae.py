"""
Train Sparse Autoencoder on MVTec normal patch tokens.

Input:  sae_training/mvtec_normal_patches_vitl14.pt   (N, 1024)
Output: sae_training/sae_vitl14_C4096_k32.pt
        sae_training/sae_vitl14_atom_stats.pt

Run from project root:
    python scripts/02_train_sae.py
"""

import torch
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import sys, time

sys.path.insert(0, str(Path(__file__).parent.parent))
from features.sae import SparseAutoencoder, SAEConfig

D_INPUT   = 1024
D_HIDDEN  = 4096
K         = 64
BATCH     = 8192
LR        = 3e-4
EPOCHS    = 15
RESAMPLE  = 200    # resample dead atoms every N steps
DEVICE    = torch.device("cuda:0")
TOKENS_IN = Path("sae_training/mvtec_normal_patches_vitl14reg.pt")
OUT_SAE   = Path("sae_training/sae_vitl14reg_C4096_k64.pt")
OUT_STATS = Path("sae_training/sae_vitl14reg_atom_stats_k64.pt")


def resample_dead(sae, counts, threshold=50, device=DEVICE):
    dead = (counts < threshold).nonzero(as_tuple=True)[0]
    n    = len(dead)
    if n == 0:
        return 0
    with torch.no_grad():
        v = torch.randn(n, D_INPUT, device=device)
        sae.W_dec.data[dead] = v / v.norm(dim=1, keepdim=True)
        e = torch.randn(D_INPUT, n, device=device)
        sae.W_enc.data[:, dead] = e / e.norm(dim=0, keepdim=True)
        sae.b_enc.data[dead] = 0.0
    counts[dead] = 0
    return n


def main():
    print(f"Loading tokens from {TOKENS_IN} ...")
    tokens = torch.load(TOKENS_IN, map_location="cpu", weights_only=True)
    print(f"Tokens : {tokens.shape}  ({tokens.nbytes/1e9:.2f} GB)\n")

    loader = DataLoader(
        TensorDataset(tokens), batch_size=BATCH,
        shuffle=True, num_workers=2, pin_memory=True, drop_last=True,
    )

    config = SAEConfig(d_input=D_INPUT, d_hidden=D_HIDDEN, k=K)
    sae    = SparseAutoencoder(config).to(DEVICE)
    opt    = torch.optim.Adam(sae.parameters(), lr=LR)
    sch    = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=EPOCHS * len(loader), eta_min=LR * 0.1,
    )

    counts = torch.zeros(D_HIDDEN, dtype=torch.long)
    step   = 0

    total_steps = EPOCHS * len(loader)
    print(f"SAE config : d={D_INPUT}  C={D_HIDDEN}  k={K}")
    print(f"Training   : {EPOCHS} epochs  ×  {len(loader)} steps  "
          f"=  {total_steps:,} total steps")
    print(f"Batch size : {BATCH} patch tokens\n")

    for ep in range(EPOCHS):
        t0       = time.time()
        ep_loss  = 0.0
        ep_spar  = 0.0

        for (batch,) in loader:
            batch       = batch.to(DEVICE, non_blocking=True)
            z, x_hat, x_norm = sae(batch)
            loss        = (x_norm - x_hat).pow(2).sum(-1).mean()

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
            opt.step()
            sch.step()
            sae._normalize_decoder()

            with torch.no_grad():
                counts += (z > 0).sum(0).cpu()

            if step > 0 and step % RESAMPLE == 0:
                n = resample_dead(sae, counts)
                if n:
                    print(f"  [step {step:5d}] resampled {n} dead atoms")
                counts.zero_()

            ep_loss += loss.item()
            ep_spar += (z > 0).float().mean().item()
            step    += 1

        avg_loss = ep_loss / len(loader)
        avg_act  = ep_spar / len(loader) * 100
        print(f"Epoch {ep+1:2d}/{EPOCHS} | "
              f"loss={avg_loss:.5f} | "
              f"active={avg_act:.1f}% | "
              f"time={time.time()-t0:.0f}s")

    sae.save(str(OUT_SAE))

    print("\nComputing atom activation frequencies (50k sample)...")
    sae.eval()
    idx  = torch.randperm(len(tokens))[:50_000]
    samp = tokens[idx].to(DEVICE)
    with torch.no_grad():
        z_samp = sae.encode(samp)
    freq = (z_samp > 0).float().mean(0).cpu()
    dead = (freq == 0).sum().item()

    print(f"Dead atoms      : {dead}/{D_HIDDEN}  ({dead/D_HIDDEN*100:.1f}%)")
    print(f"Activation freq : mean={freq.mean():.4f}  "
          f"median={freq.median():.4f}  "
          f"min={freq.min():.4f}  max={freq.max():.4f}")

    torch.save({"activation_freq": freq,
                "config": config,
                "d_input": D_INPUT,
                "d_hidden": D_HIDDEN,
                "k": K}, OUT_STATS)
    print(f"Atom stats saved → {OUT_STATS}")


if __name__ == "__main__":
    main()
