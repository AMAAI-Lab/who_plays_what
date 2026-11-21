"""Training: fake mixes of the solo clips, all GPUs.

  torchrun --nproc_per_node 4 -m whoplays.train --name ours                  # batch is per GPU

The second clip of each training mix is a different instrument (20%), the same instrument from another video (30%),
the same instrument from another video, pitch / tempo matched (25%), or the same video at another moment (25%); each
mix also gets 0-3 other instruments (like URMP). Each player's video is mirrored left-right with probability 0.5.
Validation: SNR of the clicked player on 500 fixed mixes of the val recordings; the best checkpoint is ckpt/<name>.pt.
Loss (BS-RoFormer's): L1 on the waveform + L1 on the complex spectrogram at 5 resolutions (4096 .. 256, hop 147).
"""
import argparse
import copy
import csv
import math
import os

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from . import load_cfg
from .model import AVMega
from .pairs import Pairs

MIXES = dict(pair_types=[0.3, 0.25, 0.25, 0.2], max_other=3)  # random, matched, self, hetero; 0-3 other instruments
VAL = dict(pair_types=[0.4, 0.3, 0.3], max_other=3)  # the 500 validation mixes: same instrument + 0-3 others


def snr(ref, est):
    return 10 * torch.log10(ref.pow(2).sum(-1) / (ref - est).pow(2).sum(-1).clamp_min(1e-8))


def loss_fn(est, ref):
    """BS-RoFormer's loss: L1 on the waveform + L1 on the complex spectrogram at 5 resolutions."""
    loss = (est - ref).abs().mean()
    for n in (4096, 2048, 1024, 512, 256):
        w = torch.hann_window(n, device=est.device)
        S = lambda x: torch.view_as_real(torch.stft(x, n, 147, window=w, return_complex=True))
        loss = loss + (S(est) - S(ref)).abs().mean()
    return loss


def clicked(mix, tgt, vj, kp, k):
    """A Pairs batch and the clicked player k per item -> recording, clicked player's track, its V-JEPA 2 grid and keypoints."""
    i = torch.arange(len(mix))
    return mix.cuda(), tgt[i, k].cuda(), vj[i, k].cuda().float().view(len(mix), vj.shape[2], 16, -1), kp[i, k].cuda()


@torch.no_grad()
def validate(model, loader):
    model.eval()
    ours, mix = [], []
    for batch in loader:
        m, t, vj, kp = clicked(*batch, torch.arange(len(batch[0])) % 2)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            est = model(m, vj, kp)
        ours.append(snr(t, est.float()))
        mix.append(snr(t, m))
    model.train()
    return torch.cat(ours).mean().item(), torch.cat(mix).mean().item()


def train(a):
    ddp = "WORLD_SIZE" in os.environ  # started with torchrun: one process per GPU, each draws its own random pairs
    rank = 0
    if ddp:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    cfg = load_cfg()
    cfg["train"] = dict(flip=True, **MIXES)
    stats = dict(np.load(cfg["paths"]["stats"]))
    val_cfg = copy.deepcopy(cfg)
    val_cfg["train"].update(VAL)
    val_dl = DataLoader(Pairs(val_cfg, "val", stats), a.batch, num_workers=a.workers)
    model = AVMega(cfg).cuda()
    net = torch.nn.parallel.DistributedDataParallel(model) if ddp else model
    opt = torch.optim.AdamW(model.param_groups(a.lr), a.lr, betas=(0.9, 0.95), weight_decay=0.01)
    lr_at = lambda s: min(1, (s + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(1, s / a.max_steps)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)

    os.makedirs("ckpt", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    if rank == 0:
        print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters", flush=True)
        log = csv.writer(open(f"logs/{a.name}.csv", "w", buffering=1))
        log.writerow(["step", "loss", "val_snr", "val_snr_mix"])
    step, best, bad, stop, running = 0, -1e9, 0, torch.zeros(1, device="cuda"), []
    it = iter(DataLoader(Pairs(cfg, "train", stats), a.batch, num_workers=a.workers))
    for group in zip(*[it] * a.accum):  # a.accum batches per optimizer step
        step += 1
        opt.zero_grad()
        for batch in group:
            m, t, vj, kp = clicked(*batch, torch.randint(0, 2, (len(batch[0]),)))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                est = net(m, vj, kp)
            loss = loss_fn(est.float(), t)
            (loss / a.accum).backward()
            running.append(loss.item())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        opt.step()
        sched.step()
        if step % a.val_every == 0:
            if rank == 0:  # the other GPUs wait at the broadcast below
                val, val_mix = validate(model, val_dl)
                log.writerow([step, np.mean(running), val, val_mix])
                print(f"step {step} loss {np.mean(running):.4f} val SNR {val:.2f} dB (mix {val_mix:.2f})", flush=True)
                if val > best:
                    best, bad = val, 0
                    torch.save(dict(model=model.state_dict(), args=vars(a), step=step), f"ckpt/{a.name}.pt")
                else:
                    bad += 1
                stop[0] = bad >= a.patience or step >= a.max_steps
            running = []
            if ddp:
                dist.broadcast(stop, 0)
            if stop.item():
                break
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--batch", type=int, default=8, help="per GPU (8 uses about 70 GB)")
    ap.add_argument("--workers", type=int, default=6, help="per GPU")
    ap.add_argument("--accum", type=int, default=1, help="batches per optimizer step (gradient accumulation)")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--max-steps", type=int, default=52000, help="length of the cosine schedule and most steps")
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--patience", type=int, default=5)
    train(ap.parse_args())
