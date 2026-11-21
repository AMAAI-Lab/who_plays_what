"""Training pairs: two solo clips of one instrument mixed together, with the motion of each player."""
import json
import os

import librosa
import numpy as np
import soundfile as sf
import torch
from torch.nn.functional import interpolate
from torch.utils.data import Dataset

from .pose import mirror, normalize
from .solos import all_clips


def rms(x):
    return np.sqrt(np.mean(x ** 2)) + 1e-8


class Pairs(Dataset):
    """Infinite stream of random pairs for training, or `n_val_pairs` fixed pairs for validation and test."""

    def __init__(self, cfg, split, stats):
        self.cfg, self.stats, self.fixed = cfg, stats, split != "train"
        split_of = json.load(open(f"{cfg['paths']['cache']}/solos/split.json"))  # solos.py split
        self.clips = [c for c in all_clips(cfg) if split_of[c["rec"]] == split
                      and os.path.exists(f"{cfg['paths']['solos']}/audio/{c['rec']}.wav")]  # videos gone from YouTube are skipped
        self.by_rec = {}
        for c in self.clips:
            self.by_rec.setdefault((c["instrument"], c["rec"]), []).append(c)
        self.n_recs = {}
        for inst, _ in self.by_rec:
            self.n_recs[inst] = self.n_recs.get(inst, 0) + 1

    def __len__(self):
        return self.cfg["n_val_pairs"] if self.fixed else 10 ** 7

    def load(self, clip, rate=1.0, steps=0, flip=False):
        """4 s of audio and 100 motion frames of a clip, optionally pitch-shifted and time-stretched, and with the video
        mirrored left-right (flip: V-JEPA 2 of the mirrored frames, keypoints mirrored)."""
        cfg = self.cfg
        sr, n = cfg["sr"], int(cfg["clip_s"] * cfg["sr"])
        path = f"{cfg['paths']['solos']}/audio/{clip['rec']}.wav"
        a = sf.read(path, start=int(clip["start"] * sr), frames=n + int(cfg["clip_extra_s"] * sr), dtype="float32")[0]
        # V-JEPA 2 grid (T, 16, 1024) at half the frame rate, used as is; the 48 body and hand keypoints (144, z-scored)
        vj = clip["feat"][:-4] + ("_vjepa_flip.npy" if flip else "_vjepa.npy")
        kp = np.load(clip["feat"]).astype(np.float32)
        feats = [(np.load(vj).reshape(-1, 16 * 1024), cfg["fps"] / 2),
                 (normalize(mirror(kp) if flip else kp, self.stats), cfg["fps"])]
        if steps:
            a = librosa.effects.pitch_shift(a, sr=sr, n_steps=steps)
        if rate != 1:  # stretch the motion by the same factor
            a = librosa.effects.time_stretch(a, rate=rate)
            feats = [(interpolate(torch.from_numpy(m).float().T[None], size=round(len(m) / rate), mode="linear")[0].T.numpy()
                      .astype(m.dtype), fps) for m, fps in feats]
        out = [np.pad(a, (0, max(0, n - len(a))))[:n]]
        for m, fps in feats:
            n_fr = int(cfg["clip_s"] * fps)
            out.append(np.pad(m, ((0, max(0, n_fr - len(m))), (0, 0)))[:n_fr])
        return out

    def pick(self, rng):
        """Clips A and B and how to treat B: 'random', 'matched', 'self' or, with a 4th pair_types entry, 'hetero'
        (B plays a different instrument)."""
        p = self.cfg["train"]["pair_types"]
        kind = rng.choice(["random", "matched", "self", "hetero"][:len(p)], p=p)
        while True:
            A = self.clips[rng.integers(len(self.clips))]  # so instruments are weighted by their number of clips
            inst = A["instrument"]
            if kind == "hetero":
                other = [c for c in self.clips if c["instrument"] != inst]
                return A, other[rng.integers(len(other))], "hetero"
            if kind == "self" or self.n_recs[inst] < 2:
                same = [c for c in self.by_rec[inst, A["rec"]]
                        if abs(c["start"] - A["start"]) >= self.cfg["clip_s"] + self.cfg["clip_extra_s"]]
                if same:
                    return A, same[rng.integers(len(same))], "self"
            else:
                recs = [r for i, r in self.by_rec if i == inst and r != A["rec"]]
                B = self.by_rec[inst, recs[rng.integers(len(recs))]]
                return A, B[rng.integers(len(B))], "random" if kind == "random" else "matched"

    def __getitem__(self, i):
        cfg = self.cfg
        rng = np.random.default_rng(cfg["seed"] * 1000 + i if self.fixed else None)
        A, B, kind = self.pick(rng)
        rate, steps = 1.0, 0
        if kind == "matched":  # make B fit A: nearest of -2..+2 semitones by chroma, tempo if within 8%
            cA, cB = np.array(A["chroma"]), np.array(B["chroma"])
            steps = max(range(-2, 3), key=lambda s: np.dot(cA, np.roll(cB, s)) / (np.linalg.norm(cA) * np.linalg.norm(cB)))
            r = A["tempo"] / B["tempo"]
            rate = r if 0.92 <= r <= 1.08 else 1.0
        flip = cfg["train"].get("flip", False) and not self.fixed  # each player's video mirrored with probability 0.5
        fA, fB = (rng.random(2) < 0.5) if flip else (False, False)
        (a, *fa), (b, *fb) = self.load(A, flip=fA), self.load(B, rate, steps, flip=fB)
        a, b = a / rms(a) * 10 ** (rng.uniform(-5, 5) / 20), b / rms(b)
        mix = a + b
        if cfg["train"].get("max_other", 0):  # 0..max_other other instruments at about the players' volume, as in URMP
            other = [c for c in self.clips if c["instrument"] != A["instrument"]]
            for _ in range(rng.integers(cfg["train"]["max_other"] + 1)):
                o = self.load(other[rng.integers(len(other))])[0]
                mix = mix + o / rms(o) * 10 ** (rng.uniform(-5, 5) / 20)
        elif rng.random() < 0.3:  # leakage from a different instrument
            other = [c for c in self.clips if c["instrument"] != A["instrument"]]
            leak = self.load(other[rng.integers(len(other))])[0]
            mix = mix + leak / rms(leak) * rms(mix) * 10 ** (rng.uniform(-20, -10) / 20)
        scale = 0.9 / np.abs(mix).max()
        return (torch.from_numpy(mix * scale).float(), torch.from_numpy(np.stack([a, b]) * scale).float(),
                *[torch.from_numpy(np.stack([x, y])) for x, y in zip(fa, fb)])
