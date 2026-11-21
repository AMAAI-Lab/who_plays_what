"""Frames, SAM 2 player masks and player boxes of a video with clicks on its players."""
import glob
import os
import subprocess
import tempfile

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import median_filter


def extract_frames(video, out_dir, fps=25, width=960):
    """JPEG frames 00000.jpg, 00001.jpg, ... at `fps`, `width` px wide."""
    if not glob.glob(f"{out_dir}/*.jpg"):
        os.makedirs(out_dir, exist_ok=True)
        subprocess.run(["ffmpeg", "-loglevel", "error", "-i", video, "-vf", f"fps={fps},scale={width}:-2",
                        "-q:v", "2", "-start_number", "0", f"{out_dir}/%05d.jpg"], check=True)
    return sorted(glob.glob(f"{out_dir}/*.jpg"))


def track_masks(frames, clicks, cfg, device="cuda"):
    """SAM 2 masks for each click. frames: list of jpg paths. Returns bool (K, T, H, W) at half resolution.

    The video is processed in chunks; each chunk is prompted with the click on its first frame.
    """
    from sam2.build_sam import build_sam2_video_predictor
    predictor = build_sam2_video_predictor(cfg["paths"]["sam2_cfg"], cfg["paths"]["sam2_ckpt"], device=device)
    n = int(cfg["sam2_chunk_s"] * cfg["fps"])
    H, W = np.array(Image.open(frames[0])).shape[:2]
    masks = np.zeros((len(clicks), len(frames), H // 2, W // 2), bool)
    for s in range(0, len(frames), n):
        with tempfile.TemporaryDirectory() as tmp:  # SAM 2 reads a directory of frames named by index
            for i, f in enumerate(frames[s:s + n]):
                os.symlink(os.path.abspath(f), f"{tmp}/{i:05d}.jpg")
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                state = predictor.init_state(video_path=tmp)
                for k, pts in enumerate(clicks):  # a click is one point or a list of points on the player
                    pts = np.atleast_2d(np.array(pts, np.float32))
                    predictor.add_new_points_or_box(state, frame_idx=0, obj_id=k, points=pts,
                                                    labels=np.ones(len(pts), np.int32))
                for i, ids, logits in predictor.propagate_in_video(state):
                    for k, m in zip(ids, (logits > 0).cpu().numpy()[:, 0]):
                        masks[k, s + i] = m[::2, ::2]
    return masks


def boxes_from_masks(masks, pad=0.1):
    """Per-frame (x0, y0, x1, y1) in full-frame pixels: mask bbox padded by `pad`, 5-frame median."""
    boxes = np.zeros((len(masks), 4))
    for t, m in enumerate(masks):
        ys, xs = np.nonzero(m)
        boxes[t] = [xs.min() * 2, ys.min() * 2, xs.max() * 2, ys.max() * 2] if len(xs) else boxes[t - 1]
    w, h = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
    boxes += np.stack([-pad * w, -pad * h, pad * w, pad * h], 1)
    return median_filter(boxes, size=(5, 1), mode="nearest")


def save_overlay(frame, masks, path):
    """First frame with each player's mask drawn, to check the clicks."""
    img = Image.open(frame).convert("RGB")
    for k, m in enumerate(masks):
        color = [(255, 0, 0), (0, 128, 255), (0, 200, 0), (255, 200, 0)][k % 4]
        m = Image.fromarray((m[0] * 120).astype(np.uint8)).resize(img.size)
        img.paste(Image.new("RGB", img.size, color), mask=m)
    img.save(path)


def prepare_pair(pair, clicks, cfg):
    """Cached frames, masks and boxes of the clicked players of a video (pair: {"id", "video"}); clicks in frame_width pixels."""
    out = f"{cfg['paths']['cache']}/videos/{pair['id']}"
    frames = extract_frames(pair["video"], f"{out}/frames", cfg["fps"], cfg["frame_width"])
    if not os.path.exists(f"{out}/masks.npz"):
        masks = track_masks(frames, clicks, cfg)
        np.savez_compressed(f"{out}/masks.npz", masks=masks)
        save_overlay(frames[0], masks, f"{out}/overlay.jpg")
    masks = np.load(f"{out}/masks.npz")["masks"]
    if not os.path.exists(f"{out}/boxes.npy"):
        np.save(f"{out}/boxes.npy", np.stack([boxes_from_masks(m) for m in masks]))
    return frames, masks, np.load(f"{out}/boxes.npy")

