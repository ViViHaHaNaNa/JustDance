"""
Extract a dance timeline from a video, ready for Pose Party.

Runs MediaPipe PoseLandmarker (heavy) on every frame, keeps the largest
person, computes the same joint angles as pose_party.ANGLES from the 3D
world landmarks, smooths them offline (zero-lag Savitzky-Golay) and flags
frames where the dancer was lost.

Usage:
    python extract_dance.py <video> [--start S] [--end E] [--crop X0,Y0,X1,Y1]
                            [--title NAME] [--out DIR]
    python extract_dance.py --batch songs.json [--jobs N]      # several songs at once

Writes to DIR (default: output/<video name>[_S-E]/):
    timeline.npz   timestamps, angles, 3D + 2D joints, tracked/filled flags
    audio.wav      the matching audio, starting at timeline t=0
    preview.mp4    original video with the smoothed skeleton drawn on top
    video_720p.mp4 plain 720p copy of the same span (for pose_party --dance)
    song.json      title shown in the game's song menu (and how it was cut)

--crop keeps only part of the frame, for everything: pose detection, the 720p
copy and the preview. Use it on game footage to cut out on-screen UI and the
game's own move pictograms. Pixels, or fractions of the frame if all <= 1.

--batch takes a JSON file listing several songs (see songs.example.json) and
extracts them, --jobs at a time in parallel, then prints a summary.

Joint names follow pose_party: "L" = left side of the SCREEN. That lines
up with a player mirroring the dancer in the game's mirrored webcam view.

Needs: pip install opencv-python mediapipe numpy scipy imageio-ffmpeg
(imageio-ffmpeg is only used if ffmpeg isn't already on PATH)
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
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


def pick_person(result, w, h, prev, follow):
    """Which detected person to track this frame, as (index, hip centre) or (None, prev).

    With several people in view (two coaches, backdrop silhouettes) "the biggest" can
    flip between them from frame to frame. So once someone is being followed, stick
    with whoever is closest to where they were last frame. The first pick is the
    biggest person, or the leftmost / rightmost one for follow="left" / "right"."""
    poses = result.pose_landmarks
    if not poses:
        return None, prev
    centres = [np.array([(p[23].x + p[24].x) / 2 * w, (p[23].y + p[24].y) / 2 * h])
               for p in poses]
    if prev is not None:
        k = int(np.argmin([np.linalg.norm(c - prev) for c in centres]))
    elif follow in ("left", "right"):
        xs = [c[0] for c in centres]
        k = int(np.argmin(xs) if follow == "left" else np.argmax(xs))
    else:
        k = largest_person(result, w, h)
    return k, centres[k]


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
def parse_crop(spec, video):
    """'x0,y0,x1,y1' (pixels, or fractions of the frame if all <= 1) -> (x, y, w, h)
    with even sizes, or None."""
    if not spec:
        return None
    cap = cv2.VideoCapture(video)
    fw, fh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    try:
        v = [float(a) for a in spec.split(",")]
        assert len(v) == 4
    except (ValueError, AssertionError):
        sys.exit("--crop wants four numbers: X0,Y0,X1,Y1")
    if max(v) <= 1.0:
        v = [v[0] * fw, v[1] * fh, v[2] * fw, v[3] * fh]
    x0, y0, x1, y1 = (int(round(a)) for a in v)
    x, y, w, h = x0 & ~1, y0 & ~1, (x1 - x0) & ~1, (y1 - y0) & ~1
    if x < 0 or y < 0 or w < 64 or h < 64 or x + w > fw or y + h > fh:
        sys.exit(f"--crop {spec} doesn't fit inside the {fw}x{fh} frame")
    return x, y, w, h


def detect(video, start, end, crop=None, follow="largest"):
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
    prev, lost = None, 0   # who we're following, and for how many frames we've lost them
    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if end is not None and t >= end:
                break
            if crop:
                x, y, cw, ch = crop
                frame = frame[y:y + ch, x:x + cw]
            h, w = frame.shape[:2]
            i = len(src_t)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = landmarker.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
                int(round(i * 1000 / fps)))

            k, centre = pick_person(result, w, h, prev, follow)
            if k is None:
                lost += 1
                if lost > fps:   # gone for a second: pick again from scratch when they return
                    prev = None
            else:
                prev, lost = centre, 0
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


def write_video_copy(ffmpeg, video, start, duration, path, height=720, crop=None):
    vf = (f"crop={crop[2]}:{crop[3]}:{crop[0]}:{crop[1]}," if crop else "") + f"scale=-2:{height}"
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-ss", f"{start:.3f}", "-i", video,
           "-t", f"{duration:.3f}", "-an", "-vf", vf,
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


def write_preview(ffmpeg, video, tl, path, crop=None):
    start, n, fps = tl["source_t"][0], len(tl["t"]), float(tl["fps"])
    cap, _ = open_at(video, start)
    ok, frame = cap.read()
    if not ok:
        sys.exit("Couldn't re-read video for preview")
    if crop:
        frame = frame[crop[1]:crop[1] + crop[3], crop[0]:crop[0] + crop[2]]
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
            if crop:
                frame = frame[crop[1]:crop[1] + crop[3], crop[0]:crop[0] + crop[2]]
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


def default_title(path):
    """'MovesLikeJagger.mp4' -> 'Moves Like Jagger'; 'my_song-live.mp4' -> 'my song live'."""
    stem = os.path.splitext(os.path.basename(path))[0]
    stem = re.sub(r"[_\-]+", " ", stem)
    if " " not in stem.strip():  # CamelCase with no spaces
        stem = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", stem)
    return re.sub(r"\s+", " ", stem).strip()


def run_batch(manifest, jobs, only=(), skip_existing=False):
    """Extract every song in a manifest, `jobs` at a time, each in its own process.

    The manifest is a list of songs (or {"defaults": {...}, "songs": [...]}). A song
    takes the same options as the command line: video, title, start, end, crop, follow, out.
    Relative video paths are relative to the manifest file. `only` keeps songs whose title
    contains any of those words; skip_existing leaves out songs already extracted."""
    with open(manifest, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        defaults, songs = {}, data
    else:
        defaults, songs = data.get("defaults", {}), data["songs"]
    base = os.path.dirname(os.path.abspath(manifest))
    log_dir = os.path.join("output", "_batch_logs")
    os.makedirs(log_dir, exist_ok=True)

    todo = []
    for n, entry in enumerate(songs, 1):
        e = {**defaults, **entry}
        video = e["video"] if os.path.isabs(e["video"]) else os.path.join(base, e["video"])
        if not os.path.exists(video):
            print(f"[{n}/{len(songs)}] SKIPPED, no such file: {video}")
            continue
        title = e.get("title") or default_title(video)
        if only and not any(w.lower() in title.lower() for w in only):
            continue
        out = e.get("out") or default_out_dir(video, float(e.get("start", 0)), e.get("end"))
        if skip_existing and os.path.exists(os.path.join(out, "song.json")):
            print(f"[{n}/{len(songs)}] already extracted, skipping: {title}")
            continue
        cmd = [sys.executable, os.path.abspath(__file__), video]
        for key in ("start", "end", "crop", "follow", "title", "out"):
            if e.get(key) is not None:
                cmd += [f"--{key}", str(e[key])]
        todo.append((n, title, cmd))

    print(f"Extracting {len(todo)} songs, {jobs} at a time. Logs in {log_dir}/")
    running, results, t0 = [], [], time.time()
    while todo or running:
        while todo and len(running) < jobs:
            n, title, cmd = todo.pop(0)
            log_path = os.path.join(log_dir, f"{n:02d}_" + re.sub(r"[^A-Za-z0-9]+", "_", title) + ".log")
            log = open(log_path, "w", encoding="utf-8")
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
            running.append((title, proc, log, log_path, time.time()))
            print(f"  started   {title}")
        time.sleep(2)
        for item in running[:]:
            title, proc, log, log_path, started = item
            if proc.poll() is None:
                continue
            running.remove(item)
            log.close()
            with open(log_path, encoding="utf-8", errors="replace") as f:
                found = re.search(r"Tracked: ([\d.]+)%", f.read())
            ok = proc.returncode == 0 and found is not None
            results.append((title, ok, found.group(1) if found else None))
            detail = f"{found.group(1)}% tracked" if ok else f"see {log_path}"
            print(f"  {'done   ' if ok else 'FAILED '}   {title}: {detail}"
                  f"  ({(time.time() - started) / 60:.0f} min)")
    print(f"\nFinished {sum(r[1] for r in results)}/{len(results)} in "
          f"{(time.time() - t0) / 60:.0f} min.")
    for title, ok, tracked in results:
        note = "   <- low, check its preview.mp4" if ok and float(tracked) < 90 else ""
        print(f"  {title:32s} " + (f"{tracked}% tracked{note}" if ok else "FAILED"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    ap.add_argument("video", nargs="?")
    ap.add_argument("--batch", metavar="FILE", help="extract every song listed in a JSON file")
    ap.add_argument("--jobs", type=int, default=2, help="songs to extract at once with --batch")
    ap.add_argument("--only", nargs="+", metavar="WORD",
                    help="with --batch, only the songs whose title contains one of these")
    ap.add_argument("--skip-existing", action="store_true",
                    help="with --batch, skip songs whose output folder already has a song.json")
    ap.add_argument("--start", type=float, default=0.0, help="start time (s)")
    ap.add_argument("--end", type=float, default=None, help="end time (s)")
    ap.add_argument("--crop", metavar="X0,Y0,X1,Y1",
                    help="only use this part of the frame (pixels, or fractions)")
    ap.add_argument("--follow", choices=["largest", "left", "right"], default="largest",
                    help="with several dancers, who to start on (it then sticks with them)")
    ap.add_argument("--title", help="name shown in the song menu (default: from the file name)")
    ap.add_argument("--out", help="output folder")
    args = ap.parse_args()
    if args.batch:
        return run_batch(args.batch, max(1, args.jobs), args.only or (), args.skip_existing)
    if not args.video:
        ap.error("give a video, or --batch FILE")
    if args.end is not None and args.end <= args.start:
        sys.exit("--end must be after --start")

    out = args.out or default_out_dir(args.video, args.start, args.end)
    os.makedirs(out, exist_ok=True)
    ffmpeg = find_ffmpeg()
    crop = parse_crop(args.crop, args.video)

    print(f"Detecting poses in {args.video} ...")
    tl = build_timeline(*detect(args.video, args.start, args.end, crop, args.follow))
    np.savez_compressed(os.path.join(out, "timeline.npz"), **tl)

    print("Extracting audio ...")
    duration = len(tl["t"]) / float(tl["fps"])
    extract_audio(ffmpeg, args.video, tl["source_t"][0], duration,
                  os.path.join(out, "audio.wav"))

    print("Writing 720p copy ...")
    write_video_copy(ffmpeg, args.video, tl["source_t"][0], duration,
                     os.path.join(out, "video_720p.mp4"), crop=crop)

    print("Writing preview ...")
    write_preview(ffmpeg, args.video, tl, os.path.join(out, "preview.mp4"), crop=crop)

    title = args.title or default_title(args.video)
    with open(os.path.join(out, "song.json"), "w", encoding="utf-8") as f:
        json.dump({"title": title, "source": os.path.basename(args.video),
                   "start": args.start, "end": args.end,
                   "crop": list(crop) if crop else None}, f, indent=2)

    print(f"Saved timeline.npz, audio.wav, video_720p.mp4, preview.mp4, song.json to {out}")
    report(tl)


if __name__ == "__main__":
    main()
