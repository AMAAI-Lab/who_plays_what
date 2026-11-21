"""URMP test set: every piece with exactly two players of one instrument (29 pairs, 58 clicked players), and a click on
each of the two players.

  python -m whoplays.urmp      # writes urmp_si.json (and the clicks file, if it is missing)

Clicks (assets/urmp_clicks.json, checked by hand; check cache/videos/<id>/overlay.jpg after tracking): three points
(nose, shoulders, hips) on each player from person detection on the first frame, players ordered left to right as in
URMP's track numbers. Clicks are in frame_width (960 px) pixels.
"""
import json
import os
import re
from collections import defaultdict

import numpy as np
from PIL import Image

from . import load_cfg
from .pose import detect_all
from .video import extract_frames

NAMES = dict(vn="violin", va="viola", vc="cello", db="double bass", fl="flute", ob="oboe", cl="clarinet",
             sax="saxophone", bn="bassoon", tpt="trumpet", hn="horn", tbn="trombone", tba="tuba")


def build_index(root):
    pairs = []
    for folder in sorted(os.listdir(root)):
        d = os.path.join(root, folder)
        if not (os.path.isdir(d) and re.match(r"\d\d_", folder)):
            continue
        tracks = defaultdict(list)  # instrument code -> track numbers
        codes = folder.split("_")[2:]  # the folder name is right where a file name is not (piece 15, track 3)
        for f in os.listdir(d):
            if f.startswith("AuSep_"):
                n = int(f.split("_")[1])
                tracks[codes[n - 1]].append(n)
        n_players = sum(len(v) for v in tracks.values())
        for code, ns in tracks.items():
            if len(ns) != 2:
                continue
            N = folder.split("_", 1)[0]
            ns.sort()
            f = lambda pat, n: os.path.join(d, [x for x in os.listdir(d) if x.startswith(pat.format(n))][0])
            pairs.append(dict(id=f"{N}_{code}", piece=N, folder=folder, instrument=NAMES[code], n_players=n_players,
                              tracks=ns, mix=f("AuMix_", ""), video=f("Vid_", ""), sep=[f("AuSep_{}_", n) for n in ns]))
    return pairs


def clicks_of(pair, cfg):
    first = extract_frames(pair["video"], f"{cfg['paths']['cache']}/videos/{pair['id']}/frames", cfg["fps"], cfg["frame_width"])[0]
    people = []
    for k, s in zip(*detect_all(np.array(Image.open(first).convert("RGB")))):
        pts = [k[g][s[g] > 0.3].mean(0) for g in ([0], [5, 6], [11, 12]) if (s[g] > 0.3).any()]
        if len(pts) == 3:
            people.append(np.array(pts))
    if len(people) != pair["n_players"]:
        print(f"{pair['id']}: found {len(people)} people for {pair['n_players']} players; check its clicks")
    people = sorted(people, key=lambda c: c[:, 0].mean())
    return [people[n - 1].tolist() for n in pair["tracks"]]


if __name__ == "__main__":
    cfg = load_cfg()
    pairs = build_index(cfg["paths"]["urmp"])
    json.dump(pairs, open("urmp_si.json", "w"), indent=1)
    print(len(pairs), "pairs,", len({p["piece"] for p in pairs}), "pieces")
    if not os.path.exists(cfg["paths"]["urmp_clicks"]):  # never overwrite clicks fixed by hand
        json.dump({p["id"]: clicks_of(p, cfg) for p in pairs}, open(cfg["paths"]["urmp_clicks"], "w"), indent=1)
