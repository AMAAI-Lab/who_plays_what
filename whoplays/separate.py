"""Inference: click a player in a video, get that player's sound.

  python -m whoplays.separate video --video in.mp4 --click 640,360 [--click 1200,380] --out out/
  python -m whoplays.separate urmp --name ours [--shard i/n]   # every URMP player clicked in turn
  (--ckpt: another trained model instead of config.yaml's paths.model)

Clicks are pixels in the original video (video; a few points on the player, e.g. head, chest and hips, track better than
one) or in the 960 px frames (URMP, assets/urmp_clicks.json). Steps: SAM 2.1 follows each
clicked player through the video (re-prompted with the click every 60 s); RTMW keypoints on the player's crop of the
960 px frames; V-JEPA 2 features of the player's crop of the full-resolution frames; the model on 4 s windows with 50%
overlap, blended with Hann weights. Frames, masks and features are cached in cache/videos/<video id>/.
Outputs: video mode, <out>/player_<k>.wav per click (44.1 kHz); URMP, cache/outputs/<name>/<pair id>.wav, channel k the
output when player k is clicked.
"""
import argparse
import json
import os
import subprocess
import tempfile

import cv2
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

from . import load_cfg
from .audio import load_audio
from .model import load_model
from .pose import normalize, player_feats
from .video import extract_frames, prepare_pair
from .vjepa import features

SR = 44100


def player_features(pair, clicks, cfg):
    """Cached V-JEPA 2 grids (full-resolution frames) and raw keypoints of every clicked player of a video."""
    d = f"{cfg['paths']['cache']}/videos/{pair['id']}"
    files = [(f"{d}/vjepa_{k}.npy", f"{d}/keypoints_{k}.npy") for k in range(len(clicks))]
    if all(os.path.exists(f) for fs in files for f in fs):  # e.g. the released URMP features: no tracking needed
        return [(np.load(v), np.load(k)) for v, k in files]
    frames, _, boxes = prepare_pair(pair, clicks, cfg)
    width = int(cv2.VideoCapture(pair["video"]).get(cv2.CAP_PROP_FRAME_WIDTH))
    out = []
    for k in range(len(boxes)):
        if not os.path.exists(f"{d}/keypoints_{k}.npy"):
            np.save(f"{d}/keypoints_{k}.npy", player_feats(frames, boxes[k], cfg["pose_thr"]))
        if not os.path.exists(f"{d}/vjepa_{k}.npy"):  # the boxes are in frame_width pixels
            full = extract_frames(pair["video"], f"{d}/frames_full", cfg["fps"], width)
            assert len(full) == len(frames), (len(full), len(frames))
            np.save(f"{d}/vjepa_{k}.npy", features(full, boxes[k] * width / cfg["frame_width"]))
        out.append((np.load(f"{d}/vjepa_{k}.npy"), np.load(f"{d}/keypoints_{k}.npy")))
    return out


@torch.no_grad()
def slide(model, wave, vj, kp, win_s=4.0):
    """4 s windows with 50% overlap over a whole piece, blended with Hann weights. wave (L,) scaled recording,
    vj (Tv, 16, 1024) at 12.5/s, kp (Tk, 144) at 25/s, both from frame 0."""
    n, hop = int(win_s * SR), int(win_s * SR / 2)
    nv, nk = int(win_s * 12.5), int(win_s * 25)
    x = torch.from_numpy(wave).float().cuda()
    vj, kp = torch.from_numpy(vj).cuda().float(), torch.from_numpy(kp).cuda()
    w = torch.hann_window(n, periodic=False, device="cuda").clamp_min(1e-3)
    out, wsum = torch.zeros_like(x), torch.zeros_like(x)
    for s in range(0, max(len(x) - hop, 1), hop):
        seg = x[s:s + n]
        v, k = vj[round(s / SR * 12.5):][:nv], kp[round(s / SR * 25):][:nk]
        v, k = F.pad(v, (0, 0, 0, 0, 0, nv - len(v))), F.pad(k, (0, 0, 0, nk - len(k)))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y = model(F.pad(seg, (0, n - len(seg)))[None], v[None], k[None])[0].float()
        out[s:s + len(seg)] += y[:len(seg)] * w[:len(seg)]
        wsum[s:s + len(seg)] += w[:len(seg)]
    return (out / wsum).cpu().numpy()


def separate(model, mix, feats, stats):
    """mix (L,) at 44.1 kHz and each clicked player's (V-JEPA 2 grid, raw keypoints) -> one track per player."""
    scale = 0.9 / max(np.abs(mix).max(), 1e-8)  # the level the model was trained on
    return [slide(model, mix * scale, vj, normalize(kp, stats)) / scale for vj, kp in feats]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["video", "urmp"])
    ap.add_argument("--ckpt", help="default: config.yaml's paths.model")
    ap.add_argument("--video", help="video mode: the video")
    ap.add_argument("--click", action="append", help="video mode: x,y in the video's pixels, one per player; several points "
                    "on the same player as x1,y1;x2,y2")
    ap.add_argument("--out", default="out", help="video mode: output folder")
    ap.add_argument("--name", help="urmp mode: outputs go to cache/outputs/<name>")
    ap.add_argument("--shard", default="0/1", help="urmp mode: i/n, one process per GPU")
    a = ap.parse_args()
    cfg = load_cfg()
    stats = dict(np.load(cfg["paths"]["stats"]))
    model = load_model(cfg, a.ckpt or cfg["paths"]["model"])
    if a.mode == "video":
        width = int(cv2.VideoCapture(a.video).get(cv2.CAP_PROP_FRAME_WIDTH))
        clicks = [[[float(v) * cfg["frame_width"] / width for v in pt.split(",")] for pt in c.split(";")] for c in a.click]
        pair = dict(id=os.path.splitext(os.path.basename(a.video))[0], video=a.video)
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(["ffmpeg", "-loglevel", "error", "-i", a.video, "-vn", "-ac", "1", "-ar", str(SR), f"{tmp}/a.wav"], check=True)
            mix = load_audio(f"{tmp}/a.wav", SR)
        os.makedirs(a.out, exist_ok=True)
        for k, y in enumerate(separate(model, mix, player_features(pair, clicks, cfg), stats)):
            sf.write(f"{a.out}/player_{k}.wav", y, SR)
        print(f"{len(clicks)} players -> {a.out}/; check the tracking in {cfg['paths']['cache']}/videos/{pair['id']}/overlay.jpg")
    else:
        clicks = json.load(open(cfg["paths"]["urmp_clicks"]))
        out = f"{cfg['paths']['cache']}/outputs/{a.name}"
        os.makedirs(out, exist_ok=True)
        i, n = map(int, a.shard.split("/"))
        for p in json.load(open("urmp_si.json"))[i::n]:
            ys = separate(model, load_audio(p["mix"], SR), player_features(p, clicks[p["id"]], cfg), stats)
            sf.write(f"{out}/{p['id']}.wav", np.stack(ys, 1), SR)
            print(p["id"], flush=True)
