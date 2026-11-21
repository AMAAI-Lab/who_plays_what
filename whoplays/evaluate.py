"""URMP score table (run separate.py urmp first).

  python -m whoplays.evaluate ours [other output names ...]

SDR in dB of the output against the clicked player's true track, the MUSDB18 way (museval, BSS Eval v4): per 1 s window
at URMP's 48 kHz (outputs resampled), median over the piece, then median over clicked players; higher is better.
Columns: all 58 clicked players, strings, winds & brass, the 8 pure duets (only the two same-instrument players).
Rows: doing nothing (the recording), the recording divided by the number of players (separates nothing, only turns the
volume down; SDR is not volume-proof), and each method. Writes tables/urmp.csv.
"""
import json
import os
import sys
from multiprocessing import Pool

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

from .audio import load_audio
from .score import sdr

SR = 48000
STRINGS = ("violin", "viola", "cello", "double bass")


def score(args):
    methods, p = args
    mix, refs = load_audio(p["mix"], SR), [load_audio(f, SR) for f in p["sep"]]
    L = min(len(mix), *map(len, refs))
    mix, refs = mix[:L], [r[:L] for r in refs]
    outs = {}
    for m in methods:
        x, sr = sf.read(f"cache/outputs/{m}/{p['id']}.wav", dtype="float32")
        x = x.T if sr == SR else librosa.resample(x.T, orig_sr=sr, target_sr=SR)
        outs[m] = np.pad(x[:, :L], ((0, 0), (0, max(0, L - x.shape[1]))))
    rows = []
    for k in range(2):
        r = dict(pair=p["id"], player=k, strings=p["instrument"] in STRINGS, duet=p["n_players"] == 2,
                 do_nothing=sdr(refs[k], mix, SR), mix_over_players=sdr(refs[k], mix / p["n_players"], SR))
        for m in methods:
            r[m] = sdr(refs[k], outs[m][k], SR)
        rows.append(r)
    return rows


if __name__ == "__main__":
    methods = sys.argv[1:]
    with Pool(12) as pool:
        d = pd.DataFrame([r for rows in pool.map(score, [(methods, p) for p in json.load(open("urmp_si.json"))]) for r in rows])
    os.makedirs("tables", exist_ok=True)
    d.round(2).to_csv("tables/urmp.csv", index=False)
    print("URMP, median SDR (dB), higher is better")
    print(f"{'':28s} {'all':>6s} {'strings':>8s} {'winds&brass':>12s} {'duets':>6s}")
    for m, label in [("do_nothing", "doing nothing"), ("mix_over_players", "recording / number of players")] + [(m, m) for m in methods]:
        print(f"{label:28s} {d[m].median():6.2f} {d[d.strings][m].median():8.2f} {d[~d.strings][m].median():12.2f} "
              f"{d[d.duet][m].median():6.2f}")
