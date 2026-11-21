import museval
import numpy as np


def sdr(ref, est, sr):
    """The MUSDB18 / SiSEC 2018 way (museval, BSS Eval v4): SDR per 1 s window, median over the windows (silent windows
    give NaN and are skipped). Only the clicked player's true track is the reference."""
    return float(np.nanmedian(museval.evaluate(ref[None, :, None], est[None, :, None], win=sr, hop=sr)[0][0]))
