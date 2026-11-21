"""Training data: download Solos + MUSIC-21, build the clip index, keypoints, normalisation stats and the split.

  python -m whoplays.solos download
  python -m whoplays.solos clips [--shard i/n]    # one process per GPU: CUDA_VISIBLE_DEVICES=i ... --shard i/n
  python -m whoplays.solos split
  python -m whoplays.solos stats

Our clips instead of new ones: `clips --list clips.csv` (columns rec, instrument, start, split; from the dataset on
Hugging Face) indexes exactly those clips, and `split` / `stats` are not needed (split.json and stats.npz are released).
"""
import argparse
import glob
import importlib.util
import json
import os
import re
import subprocess
import sys
from multiprocessing.pool import ThreadPool

import cv2
import librosa
import numpy as np
import pandas as pd
import soundfile as sf

from . import load_cfg
from .pose import clean, solo_feats


def name_of(s):
    return re.sub(r"(?<!^)(?=[A-Z])", " ", s).replace("_", " ").lower()


def recordings(cfg):
    """Table of (id, instrument, split) for the downloaded recordings. Split by recording, fixed seed."""
    root = cfg["paths"]["solos"]
    rows = {}
    solos_dir = importlib.util.find_spec("Solos").submodule_search_locations[0]
    solos = json.load(open(f"{solos_dir}/json_files/solos_ids.json"))
    music = json.load(open(f"{root}/MUSIC21_solo_videos.json"))["videos"]
    for inst, ids in list(solos.items()) + list(music.items()):
        for i in ids:
            rows.setdefault(i, name_of(inst))  # remove duplicate YouTube ids
    df = pd.DataFrame(sorted(rows.items()), columns=["id", "instrument"])
    df["split"] = np.where(np.random.default_rng(cfg["seed"]).random(len(df)) < cfg["val_frac"], "val", "train")
    return df


def download_one(args):
    root, i = args
    video, audio = f"{root}/video/{i}.mp4", f"{root}/audio/{i}.wav"
    if os.path.exists(video) and os.path.exists(audio):
        return
    raw = f"{root}/raw/{i}"
    subprocess.run([sys.executable, "-m", "yt_dlp", "-q", "-f", "bv*[height<=480]+ba/b[height<=480]", "--js-runtimes", "node", "-o", raw + ".%(ext)s",
                    "--merge-output-format", "mkv", f"https://www.youtube.com/watch?v={i}"])
    src = glob.glob(raw + ".*")
    if src:  # videos that are gone are skipped
        ok = subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", src[0], "-an", "-vf", "fps=25", "-c:v", "libx264",
                             "-crf", "28", video]).returncode == 0
        ok = ok and subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", src[0], "-vn", "-ac", "1", "-ar", str(44100),
                                    audio]).returncode == 0
        for f in src + ([] if ok else [video, audio]):  # a download without video is skipped like a missing video
            if os.path.exists(f):
                os.remove(f)


def read_frames(path, start, n):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, round(start * 25))
    frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in (cap.read()[1] for _ in range(n)) if f is not None]
    return frames


def index_recording(row, cfg, starts=None):
    """Pick up to 40 loud segments with a person in view (or take the given start times, in s); save their keypoints
    and a JSON entry."""
    root, cache = cfg["paths"]["solos"], cfg["paths"]["cache"]
    out = f"{cache}/solos/clips/{row.id}.json"
    if os.path.exists(out):
        return
    audio, sr = sf.read(f"{root}/audio/{row.id}.wav", dtype="float32")
    audio /= max(np.abs(audio).max(), 1e-8)
    n, extra = int(cfg["clip_s"] * sr), int(cfg["clip_extra_s"] * sr)
    starts = np.arange(0, len(audio) - n - extra, int(cfg["clip_hop_s"] * sr))
    loud = [s for s in starts if 20 * np.log10(np.sqrt(np.mean(audio[s:s + n] ** 2)) + 1e-9) > cfg["min_dbfs"]]
    # look at a spread of up to 60 candidates, keep the first 40 that pass the person test
    cands = [loud[i] for i in np.unique(np.linspace(0, len(loud) - 1, min(60, len(loud))).astype(int))] if loud else []
    if starts is not None:
        cands = [round(t * sr) for t in starts]
    clips = []
    for s in cands:
        if len(clips) == cfg["clips_per_recording"]:
            break
        frames = read_frames(f"{root}/video/{row.id}.mp4", s / sr, round((cfg["clip_s"] + cfg["clip_extra_s"]) * 25))
        if len(frames) < 110 and starts is None:
            continue
        feats, found = solo_feats(frames, cfg["pose_thr"])
        if found < cfg["min_person_frac"] and starts is None:
            continue
        name = f"{cache}/solos/feats/{row.id}_{s // sr}"
        os.makedirs(os.path.dirname(name), exist_ok=True)
        np.save(name + ".npy", feats)
        seg = audio[s:s + n]
        clips.append(dict(rec=row.id, instrument=row.instrument, split=row.split, start=float(s / sr), feat=name + ".npy",
                          chroma=librosa.feature.chroma_stft(y=seg, sr=sr).mean(1).tolist(),
                          tempo=float(librosa.feature.tempo(y=seg, sr=sr)[0])))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(clips, open(out, "w"))


def all_clips(cfg):
    return [c for f in sorted(glob.glob(f"{cfg['paths']['cache']}/solos/clips/*.json")) for c in json.load(open(f))]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["download", "clips", "split", "stats"])
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--list", help="clips: csv of the clips to index (rec, instrument, start, split)")
    a = ap.parse_args()
    cfg = load_cfg()
    root = cfg["paths"]["solos"]
    df = recordings(cfg)
    if a.step == "download":
        for d in ("raw", "video", "audio"):
            os.makedirs(f"{root}/{d}", exist_ok=True)
        ThreadPool(8).map(download_one, [(root, i) for i in df.id])
    elif a.step == "clips":
        i, n = map(int, a.shard.split("/"))
        if a.list:  # our clips: their start times, and the split as the recording's split
            given = pd.read_csv(a.list)
            df = given.groupby("rec").first().reset_index().rename(columns={"rec": "id"})[["id", "instrument", "split"]]
            starts = given.groupby("rec").start.apply(list).to_dict()
        have = df[[os.path.exists(f"{root}/audio/{r}.wav") for r in df.id]]
        for row in have.iloc[i::n].itertuples():
            index_recording(row, cfg, starts[row.id] if a.list else None)
    elif a.step == "split":  # recording -> train / val / test / excluded: the MUSIC / MUSIC-21 benchmark test videos are
        # left out, the first random split's val recordings become the test split (never used to pick a checkpoint), a new
        # val split is drawn from the rest
        bench = set()
        for f in glob.glob(cfg["paths"]["benchmark_lists"]):
            bench |= {os.path.basename(x.strip())[:-4] for line in open(f) for x in line.split(",") if x.strip().endswith((".wav", ".mp4"))}
        recs = sorted({(c["rec"], c["split"]) for c in all_clips(cfg)})
        rng = np.random.default_rng(1)
        split = {r: "test" if s == "val" else "excluded" if r in bench else "val" if rng.random() < cfg["val_frac"] else "train"
                 for r, s in recs}
        json.dump(split, open(f"{cfg['paths']['cache']}/solos/split.json", "w"), indent=0)
        print(pd.Series(split).value_counts().to_dict())
    else:  # z-score statistics of the keypoints, from the first random split's training clips
        clips = [c for c in all_clips(cfg) if c["split"] == "train"]
        sel = np.random.default_rng(0).choice(len(clips), min(3000, len(clips)), replace=False)
        x = np.concatenate([clean(np.load(clips[i]["feat"])) for i in sel])
        os.makedirs(os.path.dirname(cfg["paths"]["stats"]), exist_ok=True)
        np.savez(cfg["paths"]["stats"], kp_mean=x.mean(0), kp_std=x.std(0) + 1e-6)
        print(len(clips), "training clips,", len({c["rec"] for c in clips}), "recordings")
