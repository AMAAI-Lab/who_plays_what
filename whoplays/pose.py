"""Keypoints per player and frame at 25 fps: 48 body and hand points (shoulders, elbows, wrists, both hands) of RTMW
whole-body pose, as box-normalised velocity (dx, dy) + score = 144 numbers per frame."""
import numpy as np
from PIL import Image
from scipy.ndimage import median_filter

KP = list(range(5, 11)) + list(range(91, 133))  # shoulders, elbows, wrists, both hands: 48 points
# KP positions of each point's left/right partner: shoulders, elbows, wrists, then the two hands (21 points each)
MIRROR = [1, 0, 3, 2, 5, 4] + list(range(27, 48)) + list(range(6, 27))
_pose = []


def detect_all(img):
    """Every person in an RGB image: keypoints (n, 133, 2) and scores (n, 133)."""
    if not _pose:
        from rtmlib import Wholebody
        _pose.append(Wholebody(to_openpose=False, mode="balanced", backend="onnxruntime", device="cuda"))
    return _pose[0](np.ascontiguousarray(img[..., ::-1]))  # rtmlib expects BGR


def detect(img):
    """Best person in an RGB image: keypoints (133, 2) and scores (133,); zeros if nobody is found."""
    kps, scs = detect_all(img)
    if len(kps) == 0:
        return np.zeros((133, 2)), np.zeros(133)
    best = scs.mean(1).argmax()
    return kps[best], scs[best]


def crop(img, box):
    x0, y0, x1, y1 = np.round(box).astype(int)
    return img[max(y0, 0):max(y1, y0 + 2), max(x0, 0):max(x1, x0 + 2)], (max(x0, 0), max(y0, 0))


def keypoint_feats(kp, sc, boxes, thr=0.3):
    """kp (T,133,2) in frame pixels, sc (T,133), boxes (T,4) -> (T, 144): box-normalised velocity (dx, dy) + score."""
    k, s = kp[:, KP], sc[:, KP]
    k = (k - boxes[:, None, :2]) / np.maximum(boxes[:, None, 2:] - boxes[:, None, :2], 16)
    v = np.zeros_like(k)
    v[1:] = k[1:] - k[:-1]
    ok = s > thr
    ok[1:] &= ok[:-1]
    ok[0] = False
    return np.concatenate([v * ok[..., None], s[..., None]], -1).reshape(len(k), -1)


def player_feats(frames, boxes, thr=0.3):
    """A tracked player (URMP or any video): pose on the player's crop. frames: jpg paths, boxes: (T, 4).
    Returns (T, 144) float16."""
    kp, sc = np.zeros((len(frames), 133, 2)), np.zeros((len(frames), 133))
    for t, f in enumerate(frames):
        c, (ox, oy) = crop(np.array(Image.open(f).convert("RGB")), boxes[t])
        k, sc[t] = detect(c)
        kp[t] = k + [ox, oy]
    return keypoint_feats(kp, sc, boxes, thr).astype(np.float16)


def solo_feats(frames, thr=0.3):
    """Training clips: frames are RGB arrays. Pose on the full frame (640 px wide); the player box is the best
    person's keypoints (score > thr), padded 20%. Returns (T, 144) float16 and the fraction of frames with a person."""
    scale = 640 / frames[0].shape[1]
    kp, sc, boxes, found = np.zeros((len(frames), 133, 2)), np.zeros((len(frames), 133)), np.full((len(frames), 4), np.nan), []
    for t, img in enumerate(frames):
        small = np.array(Image.fromarray(img).resize((640, round(img.shape[0] * scale))))
        k, sc[t] = detect(small)
        kp[t] = k / scale
        ok = sc[t] > thr
        found.append(ok.sum() > 1)
        if ok.sum() > 1:
            lo, hi = kp[t][ok].min(0), kp[t][ok].max(0)
            boxes[t] = np.r_[lo - 0.2 * (hi - lo), hi + 0.2 * (hi - lo)]
    if not any(found):
        return None, 0.0
    boxes = np.stack([np.interp(np.arange(len(boxes)), np.flatnonzero(found), boxes[found, i]) for i in range(4)], 1)
    boxes = median_filter(boxes, size=(5, 1), mode="nearest")
    return keypoint_feats(kp, sc, boxes, thr).astype(np.float16), float(np.mean(found))


def clean(feats):
    """(T, 144) keypoint features without inf/NaN."""
    f = np.nan_to_num(feats.astype(np.float32), nan=0, posinf=0, neginf=0)
    kp = f.reshape(len(f), 48, 3)
    kp[..., :2] = np.clip(kp[..., :2], -1, 1)
    return f


def mirror(feats):
    """Keypoint features of the left-right mirrored video: x velocities negated, left and right points swapped."""
    kp = feats.reshape(len(feats), 48, 3)[:, MIRROR].copy()
    kp[..., 0] *= -1
    return kp.reshape(len(feats), 144)


def normalize(feats, stats):
    """z-scored with the training clips' statistics (solos.py stats)."""
    return ((clean(feats) - stats["kp_mean"]) / stats["kp_std"]).astype(np.float32)
