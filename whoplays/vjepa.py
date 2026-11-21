"""V-JEPA 2 (ViT-L, 256 px) features of a player's crop: a 4x4 grid of 1024 numbers per 2 video frames (12.5 per second). Frozen.

  python -m whoplays.vjepa --shard i/n           # training clips, one process per GPU (CUDA_VISIBLE_DEVICES=i)
  python -m whoplays.vjepa --flip --shard i/n    # the same mirrored left-right (flip augmentation)

Crop: the player's box made square, resized to 256 x 256 (outside the frame is black). Solo clips: the largest person
RT-DETRv2 finds on every 5th frame, interpolated, smoothed and padded 20%. Tracked players (separate.py): the SAM 2 box,
on the full-resolution frames. Frames go through V-JEPA 2 in windows of 64 (2.56 s); a short last window is padded with
its last frame. Saved as float16 (ceil(frames / 2), 16, 1024): solo clips next to their keypoints as <feat>_vjepa.npy
(mirrored: <feat>_vjepa_flip.npy).
"""
import argparse
import os

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import median_filter
from torchvision.ops import roi_align

from . import load_cfg
from .solos import all_clips, read_frames

NAME, DET, WIN, SIZE = "facebook/vjepa2-vitl-fpc64-256", "PekingU/rtdetr_v2_r50vd", 64, 256
MEAN, STD = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1), torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
_models = {}


def _get(name):
    if name not in _models:
        if name == "vjepa":
            from transformers import VJEPA2Model
            _models[name] = VJEPA2Model.from_pretrained(NAME, dtype=torch.bfloat16).cuda().eval()
        else:
            from transformers import RTDetrImageProcessor, RTDetrV2ForObjectDetection
            _models[name] = (RTDetrV2ForObjectDetection.from_pretrained(DET).cuda().eval(), RTDetrImageProcessor.from_pretrained(DET))
    return _models[name]


@torch.no_grad()
def features(frames, boxes):
    """frames: T RGB uint8 arrays or jpg paths (read one window at a time), boxes (T, 4) x0 y0 x1 y1 in pixels
    -> (ceil(T / 2), 16, 1024) float16."""
    c, s = (boxes[:, :2] + boxes[:, 2:]) / 2, np.maximum(boxes[:, 2:] - boxes[:, :2], 16).max(1, keepdims=True) / 2
    sq = torch.from_numpy(np.concatenate([np.arange(len(c))[:, None], c - s, c + s], 1)).float().cuda()
    out = []
    for i in range(0, len(frames), WIN):
        x = np.stack([np.array(Image.open(f).convert("RGB")) if isinstance(f, str) else f for f in frames[i:i + WIN]])
        x = torch.from_numpy(x).cuda().permute(0, 3, 1, 2).float() / 255
        b = sq[i:i + WIN].clone()
        b[:, 0] = torch.arange(len(b))
        x = (roi_align(x, b, SIZE, aligned=True).cpu() - MEAN) / STD
        x = torch.cat([x, x[-1:].expand(WIN - len(x), -1, -1, -1)]) if len(x) < WIN else x
        h = _get("vjepa")(pixel_values_videos=x[None].cuda().bfloat16(), skip_predictor=True).last_hidden_state
        g = SIZE // 16  # 16 x 16 patches per tubelet of 2 frames, pooled to 4 x 4
        h = h.view(WIN // 2, g, g, -1).permute(0, 3, 1, 2)
        out.append(torch.nn.functional.avg_pool2d(h.float(), g // 4).flatten(2).transpose(1, 2).half().cpu())
    return torch.cat(out)[:(len(frames) + 1) // 2].numpy()


def solo_boxes(frames, step=5, pad=0.2):
    """Largest detected person every `step` frames, interpolated over the clip and padded; the full frame if nobody is found."""
    H, W = frames[0].shape[:2]
    model, proc = _get("det")
    with torch.no_grad():
        inp = proc(images=frames[::step], return_tensors="pt").to("cuda")
        res = proc.post_process_object_detection(model(**inp), target_sizes=[(H, W)] * len(inp["pixel_values"]), threshold=0.5)
    t, b = [], []
    for i, r in zip(range(0, len(frames), step), res):
        d = r["boxes"][r["labels"] == 0].cpu().numpy()  # label 0 is person
        if len(d):
            t.append(i)
            b.append(d[np.argmax((d[:, 2] - d[:, 0]) * (d[:, 3] - d[:, 1]))])
    if not t:
        return np.tile([0, 0, W, H], (len(frames), 1)).astype(float)
    b = np.array(b)
    boxes = np.stack([np.interp(np.arange(len(frames)), t, b[:, j]) for j in range(4)], 1)
    boxes = median_filter(boxes, size=(5, 1), mode="nearest")
    w, h = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
    return boxes + np.stack([-pad * w, -pad * h, pad * w, pad * h], 1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--flip", action="store_true", help="frames mirrored left-right")
    a = ap.parse_args()
    cfg = load_cfg()
    i, n = map(int, a.shard.split("/"))
    n_fr = round((cfg["clip_s"] + cfg["clip_extra_s"]) * cfg["fps"])
    clips = sorted(all_clips(cfg), key=lambda c: c["feat"])[i::n]
    for j, c in enumerate(clips):
        out = c["feat"][:-4] + ("_vjepa_flip.npy" if a.flip else "_vjepa.npy")
        if not os.path.exists(out):
            frames = read_frames(f"{cfg['paths']['solos']}/video/{c['rec']}.mp4", c["start"], n_fr)
            frames = [np.ascontiguousarray(f[:, ::-1]) for f in frames] if a.flip else frames
            np.save(out, features(frames, solo_boxes(frames)))
        if j % 500 == 0:
            print(f"{j} / {len(clips)}", flush=True)
    print("done", flush=True)
