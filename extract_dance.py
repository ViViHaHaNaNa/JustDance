"""
Extract a dance timeline from a video, ready for Pose Party.

Runs MediaPipe PoseLandmarker (heavy) on every frame, keeps the largest
person, computes the same joint angles as pose_party.ANGLES from the 3D
world landmarks, smooths them offline (zero-lag Savitzky-Golay) and flags
frames where the dancer was lost.

Usage:
    python extract_dance.py <video> [--start S] [--end E] [--out DIR]

Writes to DIR (default: output/<video name>[_S-E]/):
    timeline.npz   timestamps, angles, 3D + 2D joints, tracked/filled flags
    audio.wav      the matching audio, starting at timeline t=0
    preview.mp4    original video with the smoothed skeleton drawn on top
    video_720p.mp4 plain 720p copy of the same span (for pose_party --dance)

Joint names follow pose_party: "L" = left side of the SCREEN. That lines
up with a player mirroring the dancer in the game's mirrored webcam view.

Needs: pip install opencv-python mediapipe numpy scipy imageio-ffmpeg
(imageio-ffmpeg is only used if ffmpeg isn't already on PATH)
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from scipy.signal import savgol_filter

from pose_party import ANGLES, BONES, NOSE, PAIRS, joint_angle

# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task")
MODEL_PATH = "pose_landmarker_heavy.task"
NUM_POSES = 3

MIN_VIS = 0.5        # angle needs all three of its joints at least this visible
CORE_MIN_VIS = 0.5   # frame is "lost" if upper-body visibility averages below this
CORE = [s + j for j in ("shoulder", "elbow", "wrist", "hip") for s in "LR"]
SMOOTH_WINDOW = 0.25  # Savitzky-Golay window in seconds
SMOOTH_ORDER = 2
MAX_GAP = 0.25       # bridge dropouts up to this long (s) by interpolation

JOINTS = ["head"] + [s + j for j in PAIRS for s in "LR"]
ANGLE_NAMES = list(ANGLES)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading heavy pose model (one time, ~30 MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH + ".part")
        os.replace(MODEL_PATH + ".part", MODEL_PATH)


def find_ffmpeg():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        sys.exit("ffmpeg not found: install it or `pip install imageio-ffmpeg`")


def largest_person(result, w, h):
    """Index of the pose with the biggest 2D bounding box, or None."""
    best, best_area = None, 0.0
    for i, lms in enumerate(result.pose_landmarks):
        xs = np.array([p.x for p in lms]) * w
        ys = np.array([p.y for p in lms]) * h
        area = (xs.max() - xs.min()) * (ys.max() - ys.min())
        if area > best_area:
            best, best_area = i, area
    return best


def named_joints(img_lms, world_lms, w, h):
    """-> (2D pixel points, 3D world points, visibility), each (len(JOINTS), ...).

    Uses the same screen-side naming as pose_party.landmarks_to_points; the
    left/right swap is decided in image space and applied to both.
    """
    swap = img_lms[11].x > img_lms[12].x
    idx = {"head": NOSE}
    for joint, (a, b) in PAIRS.items():
        left, right = (b, a) if swap else (a, b)
        idx["L" + joint], idx["R" + joint] = left, right
    order = [idx[j] for j in JOINTS]
    p2 = np.array([[img_lms[i].x * w, img_lms[i].y * h] for i in order])
    p3 = np.array([[world_lms[i].x, world_lms[i].y, world_lms[i].z] for i in order])
    vis = np.array([img_lms[i].visibility for i in order])
    return p2, p3, vis


def runs(mask):
    """(start, stop) index pairs of each run of True in a 1D bool array."""
    padded = np.concatenate([[False], mask, [False]]).astype(np.int8)
    edges = np.flatnonzero(np.diff(padded))
    return list(zip(edges[::2], edges[1::2]))


def fill_short_gaps(x, max_gap):
    """Linearly interpolate NaN runs of at most max_gap frames (1D, in place)."""
    good = ~np.isnan(x)
    if good.sum() < 2:
        return x
    for a, b in runs(~good):
        if a > 0 and b < len(x) and b - a <= max_gap:
            x[a:b] = np.interp(np.arange(a, b), [a - 1, b], [x[a - 1], x[b]])
    return x


def smooth_series(x, window, max_gap):
    """Zero-lag smoothing of a 1D series with NaN gaps.

    Short gaps are bridged first; each remaining contiguous run is smoothed
    on its own so long gaps never bleed into neighbouring good data.
    """
    x = fill_short_gaps(x.astype(float).copy(), max_gap)
    out = np.full_like(x, np.nan)
    for a, b in runs(~np.isnan(x)):
        seg = x[a:b]
        win = min(window, (b - a) if (b - a) % 2 else (b - a) - 1)
        if win > SMOOTH_ORDER + 1:
            out[a:b] = savgol_filter(seg, win, SMOOTH_ORDER, mode="interp")
        else:
            out[a:b] = seg
    return out


def fmt_time(t):
    return f"{int(t // 60)}:{t % 60:05.2f}"


def default_out_dir(video, start, end):
    stem = re.sub(r"[^A-Za-z0-9]+", "_", os.path.splitext(os.path.basename(video))[0])
    stem = stem.strip("_")[:40] or "dance"
    if start or end is not None:
        stem += f"_{start:g}-{'end' if end is None else f'{end:g}'}"
    return os.path.join("output", stem)


def open_at(video, start):
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        sys.exit(f"Can't open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, round(start * fps))
    return cap, fps


# --------------------------------------------------------------------------
# Pass 1: detection
# --------------------------------------------------------------------------
def detect(video, start, end):
    ensure_model()
    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=NUM_POSES,
    )
    cap, fps = open_at(video, start)
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    span = (min(end, total) if end is not None else total) - start
    n_est = max(1, int(span * fps))

    src_t, p2s, p3s, viss, found = [], [], [], [], []
    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if end is not None and t >= end:
                break
            h, w = frame.shape[:2]
            i = len(src_t)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = landmarker.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
                int(round(i * 1000 / fps)))

            k = largest_person(result, w, h)
            if k is None:
                p2, p3, vis = (np.full((len(JOINTS), d), np.nan) for d in (2, 3, 1))
                vis = vis[:, 0]
            else:
                p2, p3, vis = named_joints(result.pose_landmarks[k],
                                           result.pose_world_landmarks[k], w, h)
            src_t.append(t)
            p2s.append(p2)
            p3s.append(p3)
            viss.append(vis)
            found.append(k is not None)

            if i % max(1, n_est // 20) == 0:
                print(f"  detecting... {min(100, 100 * i // n_est):3d}%  ({fmt_time(t)})",
                      flush=True)
    cap.release()
    if not src_t:
        sys.exit("No frames read - check --start/--end")
    return (fps, (w, h), np.array(src_t), np.array(p2s), np.array(p3s),
            np.array(viss), np.array(found))


# --------------------------------------------------------------------------
# Timeline: flags, angles, smoothing
# --------------------------------------------------------------------------
def build_timeline(fps, frame_size, src_t, p2, p3, vis, found):
    n = len(src_t)
    j = {name: i for i, name in enumerate(JOINTS)}
    core = [j[name] for name in CORE]
    core_vis = np.where(found, np.nan_to_num(vis[:, core]).mean(axis=1), 0.0)
    tracked = found & (core_vis >= CORE_MIN_VIS)

    # Raw angles from 3D world joints; NaN when lost or a joint isn't visible
    raw = np.full((n, len(ANGLE_NAMES)), np.nan)
    for f in np.flatnonzero(tracked):
        for k, (a, b, c) in enumerate(ANGLES.values()):
            if min(vis[f, j[a]], vis[f, j[b]], vis[f, j[c]]) >= MIN_VIS:
                raw[f, k] = joint_angle(p3[f, j[a]], p3[f, j[b]], p3[f, j[c]])

    window = int(round(SMOOTH_WINDOW * fps)) | 1
    gap = int(round(MAX_GAP * fps))
    angles = np.stack([smooth_series(raw[:, k], window, gap)
                       for k in range(raw.shape[1])], axis=1)
    angles = np.clip(angles, 0.0, 180.0)

    def smooth_points(p):
        p = np.where(tracked[:, None, None], p, np.nan)
        out = np.empty_like(p)
        for a in range(p.shape[1]):
            for d in range(p.shape[2]):
                out[:, a, d] = smooth_series(p[:, a, d], window, gap)
        return out

    world = smooth_points(p3)
    image = smooth_points(p2)
    filled = ~tracked & ~np.isnan(world[:, 0, 0])  # short dropout, interpolated

    return {
        "t": src_t - src_t[0],          # seconds from the start of audio.wav
        "source_t": src_t,              # seconds in the original video
        "fps": np.float64(fps),
        "frame_size": np.array(frame_size),  # (w, h) of the source video
        "angle_names": np.array(ANGLE_NAMES),
        "angles": angles,               # (n, 8) smoothed degrees, NaN = unknown
        "angles_raw": raw,              # (n, 8) unsmoothed
        "joint_names": np.array(JOINTS),
        "world": world,                 # (n, 13, 3) metres, hip-centred, smoothed
        "image": image,                 # (n, 13, 2) pixels, smoothed
        "visibility": vis,              # (n, 13)
        "tracked": tracked,             # (n,) person found with good visibility
        "filled": filled,               # (n,) lost but bridged by interpolation
    }


# --------------------------------------------------------------------------
# Audio + preview
# --------------------------------------------------------------------------
def extract_audio(ffmpeg, video, start, duration, path):
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-ss", f"{start:.3f}", "-i", video,
           "-t", f"{duration:.3f}", "-vn", "-ac", "2", "-ar", "44100", path]
    subprocess.run(cmd, check=True)


def write_video_copy(ffmpeg, video, start, duration, path, height=720):
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-ss", f"{start:.3f}", "-i", video,
           "-t", f"{duration:.3f}", "-an", "-vf", f"scale=-2:{height}",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
           "-pix_fmt", "yuv420p", path]
    subprocess.run(cmd, check=True)


def draw_skeleton(img, pts, color, thick):
    for a, b in BONES:
        pa, pb = pts[JOINTS.index(a)], pts[JOINTS.index(b)]
        if not (np.isnan(pa).any() or np.isnan(pb).any()):
            cv2.line(img, tuple(int(v) for v in pa), tuple(int(v) for v in pb),
                     color, thick, cv2.LINE_AA)
    for p in pts:
        if not np.isnan(p).any():
            cv2.circle(img, tuple(int(v) for v in p), thick + 2, color, -1, cv2.LINE_AA)


def write_preview(ffmpeg, video, tl, path):
    start, n, fps = tl["source_t"][0], len(tl["t"]), float(tl["fps"])
    cap, _ = open_at(video, start)
    ok, frame = cap.read()
    if not ok:
        sys.exit("Couldn't re-read video for preview")
    h, w = frame.shape[:2]
    thick = max(2, h // 180)
    cmd = [ffmpeg, "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", f"{fps:.6f}",
           "-i", "-",
           "-ss", f"{start:.3f}", "-t", f"{n / fps:.3f}", "-i", video,
           "-map", "0:v", "-map", "1:a?", "-c:v", "libx264", "-preset", "veryfast",
           "-crf", "23", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", path]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    for f in range(n):
        if f > 0:
            ok, frame = cap.read()
            if not ok:
                break
        tracked, filled = tl["tracked"][f], tl["filled"][f]
        color = (90, 230, 90) if tracked else (0, 200, 255) if filled else (90, 90, 255)
        draw_skeleton(frame, tl["image"][f], (0, 0, 0), thick + 3)
        draw_skeleton(frame, tl["image"][f], color, thick)
        label = "TRACKED" if tracked else "FILLED" if filled else "LOST"
        text = f"{fmt_time(tl['source_t'][f])}  {label}"
        org, scale = (20, 20 + int(h / 25)), h / 900
        cv2.putText(frame, text, org, cv2.FONT_HERSHEY_DUPLEX, scale, (0, 0, 0),
                    thick + 3, cv2.LINE_AA)
        cv2.putText(frame, text, org, cv2.FONT_HERSHEY_DUPLEX, scale, color,
                    max(1, thick - 1), cv2.LINE_AA)
        proc.stdin.write(frame.tobytes())
    cap.release()
    proc.stdin.close()
    if proc.wait() != 0:
        sys.exit("ffmpeg failed writing the preview")


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------
def report(tl, top=5):
    tracked, fps, src_t = tl["tracked"], float(tl["fps"]), tl["source_t"]
    n = len(tracked)
    print(f"\nFrames: {n}  ({n / fps:.1f} s at {fps:g} fps)")
    print(f"Tracked: {100 * tracked.mean():.1f}%"
          f"   (+{100 * tl['filled'].mean():.1f}% short gaps bridged)")
    valid = ~np.isnan(tl["angles"])
    print("Angle coverage: " + "  ".join(
        f"{name} {100 * valid[:, k].mean():.0f}%" for k, name in enumerate(ANGLE_NAMES)))

    lost = sorted(runs(~tracked), key=lambda r: r[1] - r[0], reverse=True)[:top]
    if not lost:
        print("No lost frames.")
        return
    print(f"Longest lost stretches (video time):")
    for a, b in lost:
        end = src_t[b - 1] + 1 / fps
        print(f"  {fmt_time(src_t[a])} - {fmt_time(end)}   {(b - a) / fps:5.2f} s"
              f"  ({b - a} frames)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    ap.add_argument("video")
    ap.add_argument("--start", type=float, default=0.0, help="start time (s)")
    ap.add_argument("--end", type=float, default=None, help="end time (s)")
    ap.add_argument("--out", help="output folder")
    args = ap.parse_args()
    if args.end is not None and args.end <= args.start:
        sys.exit("--end must be after --start")

    out = args.out or default_out_dir(args.video, args.start, args.end)
    os.makedirs(out, exist_ok=True)
    ffmpeg = find_ffmpeg()

    print(f"Detecting poses in {args.video} ...")
    tl = build_timeline(*detect(args.video, args.start, args.end))
    np.savez_compressed(os.path.join(out, "timeline.npz"), **tl)

    print("Extracting audio ...")
    duration = len(tl["t"]) / float(tl["fps"])
    extract_audio(ffmpeg, args.video, tl["source_t"][0], duration,
                  os.path.join(out, "audio.wav"))

    print("Writing 720p copy ...")
    write_video_copy(ffmpeg, args.video, tl["source_t"][0], duration,
                     os.path.join(out, "video_720p.mp4"))

    print("Writing preview ...")
    write_preview(ffmpeg, args.video, tl, os.path.join(out, "preview.mp4"))

    print(f"Saved timeline.npz, audio.wav, video_720p.mp4, preview.mp4 to {out}")
    report(tl)


if __name__ == "__main__":
    main()
