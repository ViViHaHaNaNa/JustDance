"""
Pose Party - a tiny Just Dance-style game with OpenCV + MediaPipe.

Copy the stick figure in the top-right corner before the timer runs out.
Hold the pose steady to lock in your best score for that round.

Setup:
    pip install opencv-python mediapipe numpy sounddevice
    python pose_party.py                              # random poses
    python pose_party.py --dance output/<folder>      # dance along to a song
    python pose_party.py --dance output/<folder> --difficulty hard

Dance mode plays a timeline made by extract_dance.py: the dancer (mirrored)
on the left, you on the right with a ghost of the target pose on your body.
You are scored on every beat against the pose the dancer struck a few seconds
earlier (easy 2 s, medium 1.5 s, hard 0.3 s), best joint-angle match within
+-0.3 s. Start screen: type your name, TAB = difficulty, ENTER = dance (a 3-2-1
count-in waits until you stand inside the outline; SPACE skips that check).
At the end you get a grade and a photo of your best moment saved in photos/.
Result screen: R = replay, N = new player, Q = quit (20 s idle -> start screen).

Keys:
    Q / ESC  quit
    R        restart
    SPACE    judge the current pose right now (pause/resume in dance mode)
"""

import argparse
import os
import random
import time
import urllib.request
import wave
from collections import deque

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# --------------------------------------------------------------------------
# Settings - tweak these to change how the game feels
# --------------------------------------------------------------------------
ROUNDS = 10            # poses per game
COUNTDOWN = 3.0        # seconds before the first pose
POSE_TIME = 4.0        # seconds you get for each pose
GRACE = 0.6            # ignore the first bit of each round (reaction time)
FEEDBACK_TIME = 1.2    # how long PERFECT/GOOD/... stays on screen
HOLD_FRAMES = 8        # pose must be held this many frames to count
SMOOTH = 0.5           # 0 = no smoothing, 0.9 = very smooth but laggy
ANGLE_TOLERANCE = 75   # avg degrees off that gives a 0% match (higher = easier)
WORST_TOLERANCE = 150  # penalty for your single worst arm joint (higher = easier)

# (name, minimum match, points, BGR color)
RATINGS = [
    ("PERFECT", 0.75, 300, (0, 215, 255)),
    ("GOOD",    0.60, 200, (90, 230, 90)),
    ("OK",      0.45, 100, (255, 190, 70)),
    ("MISS",    0.00,   0, (90, 90, 255)),
]

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task")
MODEL_PATH = "pose_landmarker_lite.task"

# --------------------------------------------------------------------------
# Skeleton definitions
# Joint names use SCREEN sides: "L" = left side of the screen. Since the
# camera is mirrored, that's also the player's own left.
# --------------------------------------------------------------------------
# MediaPipe landmark indices for each left/right joint pair
PAIRS = {
    "shoulder": (11, 12), "elbow": (13, 14), "wrist": (15, 16),
    "hip": (23, 24), "knee": (25, 26), "ankle": (27, 28),
}
NOSE = 0

BONES = [
    ("Lshoulder", "Rshoulder"), ("Lhip", "Rhip"),
    ("Lshoulder", "Lhip"), ("Rshoulder", "Rhip"),
    ("Lshoulder", "Lelbow"), ("Lelbow", "Lwrist"),
    ("Rshoulder", "Relbow"), ("Relbow", "Rwrist"),
    ("Lhip", "Lknee"), ("Lknee", "Lankle"),
    ("Rhip", "Rknee"), ("Rknee", "Rankle"),
]

# Angles we compare: name -> (a, joint, c) = angle measured at `joint`
ANGLES = {
    "Lelbow":    ("Lshoulder", "Lelbow", "Lwrist"),
    "Relbow":    ("Rshoulder", "Relbow", "Rwrist"),
    "Lshoulder": ("Lhip", "Lshoulder", "Lelbow"),
    "Rshoulder": ("Rhip", "Rshoulder", "Relbow"),
    "Lhip":      ("Lshoulder", "Lhip", "Lknee"),
    "Rhip":      ("Rshoulder", "Rhip", "Rknee"),
    "Lknee":     ("Lhip", "Lknee", "Lankle"),
    "Rknee":     ("Rhip", "Rknee", "Rankle"),
}
# Arms matter most; legs are often out of frame so they count less
WEIGHTS = {"Lelbow": 0.8, "Relbow": 0.8, "Lshoulder": 1.5, "Rshoulder": 1.5,
           "Lhip": 0.6, "Rhip": 0.6, "Lknee": 0.3, "Rknee": 0.3}
REQUIRED = ("Lshoulder", "Rshoulder")  # need at least both arms visible
ARM_JOINTS = ("Lelbow", "Relbow", "Lshoulder", "Rshoulder")

# --------------------------------------------------------------------------
# Target poses, built from a base body plus arm/leg shapes.
# Coordinates: x right, y down, hips at y=0. Shapes are given for the
# screen-left side and mirrored for the right.
# --------------------------------------------------------------------------
BODY = {
    "head": (0.0, -0.9),
    "Lshoulder": (-0.25, -0.6), "Rshoulder": (0.25, -0.6),
    "Lhip": (-0.15, 0.0), "Rhip": (0.15, 0.0),
}
ARMS = {  # (elbow, wrist)
    "down": ((-0.32, -0.30), (-0.35, 0.00)),
    "out":  ((-0.55, -0.60), (-0.85, -0.60)),
    "up":   ((-0.45, -0.95), (-0.65, -1.30)),
    "flex": ((-0.55, -0.60), (-0.55, -0.95)),
    "hip":  ((-0.50, -0.30), (-0.22, -0.05)),
}
LEGS = {  # (knee, ankle)
    "stand": ((-0.15, 0.45), (-0.15, 0.90)),
    "wide":  ((-0.32, 0.45), (-0.50, 0.88)),
}

POSE_SPECS = [  # name, left arm, right arm, legs
    ("T-Pose",       "out",  "out",  "stand"),
    ("Touchdown!",   "up",   "up",   "stand"),
    ("Double Flex",  "flex", "flex", "stand"),
    ("Disco",        "up",   "hip",  "stand"),
    ("Disco (flip)", "hip",  "up",   "stand"),
    ("Point Left",   "out",  "down", "stand"),
    ("Point Right",  "down", "out",  "stand"),
    ("Teapot",       "hip",  "out",  "stand"),
    ("Robot",        "flex", "down", "stand"),
    ("Hands on Hips", "hip", "hip",  "wide"),
]


def build_pose(left_arm, right_arm, legs):
    pts = {k: np.array(v, float) for k, v in BODY.items()}
    mirror = np.array([-1.0, 1.0])
    for side, arm in (("L", left_arm), ("R", right_arm)):
        elbow, wrist = (np.array(p) for p in ARMS[arm])
        knee, ankle = (np.array(p) for p in LEGS[legs])
        if side == "R":
            elbow, wrist, knee, ankle = (p * mirror for p in (elbow, wrist, knee, ankle))
        pts[side + "elbow"], pts[side + "wrist"] = elbow, wrist
        pts[side + "knee"], pts[side + "ankle"] = knee, ankle
    return pts


# --------------------------------------------------------------------------
# Pose math
# --------------------------------------------------------------------------
def joint_angle(a, b, c):
    """Angle at point b (degrees) formed by a-b-c."""
    ba, bc = a - b, c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def pose_angles(pts, vis=None, min_vis=0.5):
    out = {}
    for name, (a, b, c) in ANGLES.items():
        if vis is not None and min(vis[a], vis[b], vis[c]) < min_vis:
            continue  # joint not clearly visible, skip it
        out[name] = joint_angle(pts[a], pts[b], pts[c])
    return out


def match_score(player, target):
    """0..1 similarity between two angle dicts."""
    if any(k not in player for k in REQUIRED):
        return 0.0
    num = den = worst = 0.0
    for name, t in target.items():
        if name in player:
            diff = abs(player[name] - t)
            num += WEIGHTS[name] * diff
            den += WEIGHTS[name]
            if name in ARM_JOINTS:
                worst = max(worst, diff)
    # Average error keeps it forgiving; the worst-arm term stops you from
    # getting PERFECT with one arm in completely the wrong place.
    avg_part = 1.0 - (num / den) / ANGLE_TOLERANCE
    worst_part = 1.0 - worst / WORST_TOLERANCE
    return float(np.clip(avg_part * worst_part, 0.0, 1.0))


POSES = []
for _name, _la, _ra, _legs in POSE_SPECS:
    _pts = build_pose(_la, _ra, _legs)
    POSES.append({"name": _name, "pts": _pts, "angles": pose_angles(_pts)})


def side_indices(lms):
    """Screen-side joint name -> MediaPipe landmark index for this frame."""
    swap = lms[11].x > lms[12].x  # make "L" always the screen-left side
    idx = {"head": NOSE}
    for joint, (a, b) in PAIRS.items():
        idx["L" + joint], idx["R" + joint] = (b, a) if swap else (a, b)
    return idx


def landmarks_to_points(lms, w, h):
    """MediaPipe landmarks -> pixel points keyed by screen-side joint names."""
    idx = side_indices(lms)
    pts = {k: np.array([lms[i].x * w, lms[i].y * h]) for k, i in idx.items()}
    vis = {k: lms[i].visibility for k, i in idx.items()}
    return pts, vis


def landmarks_to_world(lms, world):
    """3D world points (metres) with the same screen-side names as the image."""
    return {k: np.array([world[i].x, world[i].y, world[i].z])
            for k, i in side_indices(lms).items()}


# --------------------------------------------------------------------------
# Game state
# --------------------------------------------------------------------------
class Game:
    def __init__(self):
        self.reset()

    def reset(self):
        self.playlist, last = [], None
        for _ in range(ROUNDS):
            pose = random.choice([p for p in POSES if p is not last])
            self.playlist.append(pose)
            last = pose
        self.index = 0
        self.score = 0
        self.combo = 0
        self.best_combo = 0
        self.counts = {r[0]: 0 for r in RATINGS}
        self.recent = deque(maxlen=HOLD_FRAMES)
        self.best = 0.0
        self.rating = None
        self.gained = 0
        self._go("countdown")

    @property
    def pose(self):
        return self.playlist[min(self.index, len(self.playlist) - 1)]

    def elapsed(self):
        return time.monotonic() - self.t0

    def _go(self, state):
        self.state = state
        self.t0 = time.monotonic()

    def _start_round(self):
        self.recent.clear()
        self.best = 0.0
        self._go("posing")

    def _judge(self):
        for name, threshold, points, _ in RATINGS:
            if self.best >= threshold:
                break
        self.rating = name
        self.counts[name] += 1
        if name == "MISS":
            self.combo = 0
            self.gained = 0
        else:
            self.combo += 1
            self.best_combo = max(self.best_combo, self.combo)
            self.gained = int(points * (1 + 0.1 * (self.combo - 1)))
            self.score += self.gained
        self._go("feedback")

    def skip(self):
        if self.state == "posing":
            self._judge()

    def update(self, live):
        el = self.elapsed()
        if self.state == "countdown" and el >= COUNTDOWN:
            self._start_round()
        elif self.state == "posing":
            if live is None:
                self.recent.clear()
            elif el >= GRACE:
                self.recent.append(live)
                if len(self.recent) == self.recent.maxlen:
                    held = sum(self.recent) / len(self.recent)
                    self.best = max(self.best, held)
            if el >= POSE_TIME:
                self._judge()
        elif self.state == "feedback" and el >= FEEDBACK_TIME:
            self.index += 1
            if self.index >= len(self.playlist):
                self._go("done")
            else:
                self._start_round()


# --------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------
FONT = cv2.FONT_HERSHEY_DUPLEX


def put(img, text, org, scale=1.0, color=(255, 255, 255), thick=2, center=False):
    if center:
        (tw, th), _ = cv2.getTextSize(text, FONT, scale, thick)
        org = (int(org[0] - tw / 2), int(org[1] + th / 2))
    cv2.putText(img, text, org, FONT, scale, (0, 0, 0), thick + 4, cv2.LINE_AA)
    cv2.putText(img, text, org, FONT, scale, color, thick, cv2.LINE_AA)


def shade(img, x0, y0, x1, y1, alpha=0.55):
    x0, y0 = max(0, x0), max(0, y0)
    roi = img[y0:y1, x0:x1]
    roi[:] = (roi * (1 - alpha)).astype(np.uint8)


def score_color(s):
    return (60, int(80 + 175 * s), int(255 * (1 - s)))  # red -> green (BGR)


def bar(img, x, y, w, h, frac, color):
    cv2.rectangle(img, (x, y), (x + w, y + h), (40, 40, 40), -1)
    cv2.rectangle(img, (x, y), (x + int(w * np.clip(frac, 0, 1)), y + h), color, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 1)


def draw_player(img, pts, vis, color_of):
    """color_of(a, b) -> BGR colour for each bone (see limb_color)."""
    for a, b in BONES:
        if vis[a] > 0.5 and vis[b] > 0.5:
            cv2.line(img, tuple(pts[a].astype(int)), tuple(pts[b].astype(int)),
                     color_of(a, b), 6, cv2.LINE_AA)
    for k, p in pts.items():
        if k != "head" and vis[k] > 0.5:
            cv2.circle(img, tuple(p.astype(int)), 7, (255, 255, 255), -1, cv2.LINE_AA)
    if vis["head"] > 0.5:
        r = int(np.linalg.norm(pts["Lshoulder"] - pts["Rshoulder"]) * 0.3)
        cv2.circle(img, tuple(pts["head"].astype(int)), max(r, 8), color_of("head", "head"),
                   4, cv2.LINE_AA)


def draw_figure(img, pts, box, color, thick=6):
    """Draw a target stick figure scaled to fit inside box=(x0,y0,x1,y1)."""
    x0, y0, x1, y1 = box
    arr = np.array(list(pts.values()))
    mn, mx = arr.min(0), arr.max(0)
    mn[1] -= 0.15  # room for the head circle
    scale = 0.85 * min((x1 - x0) / (mx[0] - mn[0]), (y1 - y0) / (mx[1] - mn[1]))
    src_c = (mn + mx) / 2
    dst_c = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
    P = {k: tuple(((v - src_c) * scale + dst_c).astype(int)) for k, v in pts.items()}
    for a, b in BONES:
        cv2.line(img, P[a], P[b], color, thick, cv2.LINE_AA)
    cv2.circle(img, P["head"], max(8, int(0.13 * scale)), color, thick, cv2.LINE_AA)


def draw_hud(img, game, live, has_player):
    h, w = img.shape[:2]

    # Score + combo + live match meter (top-left)
    shade(img, 10, 10, 360, 150)
    put(img, f"SCORE {game.score}", (25, 50), 1.1, (255, 255, 255))
    put(img, f"COMBO x{game.combo}", (25, 90), 0.8, (0, 215, 255))
    m = live or 0.0
    bar(img, 25, 108, 320, 26, m, score_color(m))
    put(img, f"{int(m * 100)}%", (150, 129), 0.7, (255, 255, 255), 1)

    if game.state == "done":
        shade(img, 0, 0, w, h, 0.7)
        put(img, "SHOW'S OVER!", (w // 2, h // 2 - 120), 2.0, (0, 215, 255), 3, center=True)
        put(img, f"Final score: {game.score}", (w // 2, h // 2 - 40), 1.3, center=True)
        put(img, f"Best combo: x{game.best_combo}", (w // 2, h // 2 + 10), 1.0, center=True)
        summary = "   ".join(f"{n} {game.counts[n]}" for n, *_ in RATINGS)
        put(img, summary, (w // 2, h // 2 + 60), 0.8, center=True)
        put(img, "R = play again    Q = quit", (w // 2, h // 2 + 130), 0.9,
            (200, 200, 200), center=True)
        return

    # Target pose panel (top-right)
    pw, ph = int(w * 0.24), int(h * 0.55)
    px0, py0 = w - pw - 15, 15
    shade(img, px0, py0, px0 + pw, py0 + ph + 70)
    put(img, "COPY ME!", (px0 + pw // 2, py0 + 25), 0.8, (0, 215, 255), center=True)
    draw_figure(img, game.pose["pts"], (px0 + 10, py0 + 45, px0 + pw - 10, py0 + ph),
                (255, 255, 255))
    put(img, game.pose["name"], (px0 + pw // 2, py0 + ph + 20), 0.8, center=True)
    if game.state == "posing":
        left = 1 - game.elapsed() / POSE_TIME
        bar(img, px0 + 15, py0 + ph + 45, pw - 30, 14, left, (0, 215, 255))
    put(img, f"{min(game.index + 1, ROUNDS)}/{ROUNDS}", (25, h - 25), 0.8)

    # Center messages
    if game.state == "countdown":
        n = int(np.ceil(COUNTDOWN - game.elapsed()))
        put(img, str(max(n, 1)), (w // 2, h // 2), 5.0, (0, 215, 255), 8, center=True)
        put(img, "Get ready!", (w // 2, h // 2 + 110), 1.2, center=True)
    elif game.state == "feedback":
        color = next(c for n, *_, c in RATINGS if n == game.rating)
        put(img, game.rating, (w // 2, h // 2 - 40), 3.0, color, 6, center=True)
        if game.gained:
            put(img, f"+{game.gained}", (w // 2, h // 2 + 50), 1.4, color, 3, center=True)

    if not has_player:
        put(img, "Step back so I can see you!", (w // 2, h - 60), 1.1,
            (90, 90, 255), center=True)


# --------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------
def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading pose model (one time, ~6 MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


def make_landmarker():
    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
    )
    return vision.PoseLandmarker.create_from_options(options)


def main():
    ensure_model()
    landmarker = make_landmarker()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        raise SystemExit("Couldn't open the webcam. Try VideoCapture(1) if you have several.")

    window = "Pose Party"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    game = Game()
    smooth = None
    last_ts = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)  # mirror view
        h, w = frame.shape[:2]

        # Timestamps must strictly increase for VIDEO mode
        ts = max(int(time.monotonic() * 1000), last_ts + 1)
        last_ts = ts
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = landmarker.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)

        live = None
        if result.pose_landmarks:
            pts, vis = landmarks_to_points(result.pose_landmarks[0], w, h)
            if smooth is None:
                smooth = pts
            else:
                smooth = {k: SMOOTH * smooth[k] + (1 - SMOOTH) * v for k, v in pts.items()}
            angles = pose_angles(smooth, vis)
            live = match_score(angles, game.pose["angles"])
            errs = {k: abs(v - game.pose["angles"][k]) for k, v in angles.items()
                    if k in game.pose["angles"]}
            draw_player(frame, smooth, vis, limb_color(errs))
        else:
            smooth = None

        game.update(live)
        draw_hud(frame, game, live, result.pose_landmarks != [])
        cv2.imshow(window, frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("r"):
            game.reset()
        if key == ord(" "):
            game.skip()
        if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    landmarker.close()
    cv2.destroyAllWindows()


# --------------------------------------------------------------------------
# Dance mode - play along with a timeline from extract_dance.py
# --------------------------------------------------------------------------
PANEL_H = 540        # height of each half of the split screen
LOOKAHEAD = 0.7      # seconds past the scored pose for the faded "next pose" ghost
WINDOW = 0.3         # a pose counts if you match it within +-this many seconds
BEAT_HALF = 0.15     # each beat is rated on the frames within +-this of it
MIN_SCORED = 0.5     # a beat is only rated if this much of it was scorable
BEAT_FEEDBACK = 0.45 # how long the rating text stays up (beats come ~0.6 s apart)
HINTS = 3            # upcoming poses shown in the corner strip
# How many seconds late you copy the dancer: you're scored against the pose
# the dancer struck this long ago, so easy leaves more time to follow along.
DIFFICULTY = {"easy": 2.0, "medium": 1.5, "hard": 0.3}
LIMB_TOLERANCE = 60  # degrees off at which a limb turns fully red
# Compare 2D screen angles (like the classic game) or the timeline's 3D world
# angles. MediaPipe's 3D estimates are noisier, even a perfect copy of the
# dancer scores around half as many PERFECTs in 3D.
SCORE_IN_3D = False

# Which angle each limb bone shows the error of
LIMB_ANGLE = {
    ("Lshoulder", "Lelbow"): "Lshoulder", ("Lelbow", "Lwrist"): "Lelbow",
    ("Rshoulder", "Relbow"): "Rshoulder", ("Relbow", "Rwrist"): "Relbow",
    ("Lhip", "Lknee"): "Lhip", ("Lknee", "Lankle"): "Lknee",
    ("Rhip", "Rknee"): "Rhip", ("Rknee", "Rankle"): "Rknee",
}
CHAINS = [("Lshoulder", "Lelbow", "Lwrist"), ("Rshoulder", "Relbow", "Rwrist"),
          ("Lhip", "Lknee", "Lankle"), ("Rhip", "Rknee", "Rankle")]
TORSO = ("Lshoulder", "Rshoulder", "Lhip", "Rhip")
GHOST_COLOR = (255, 255, 255)
NEXT_COLOR = (255, 200, 80)


def flip_name(name):
    """Lelbow <-> Relbow (for the horizontally flipped video)."""
    return {"L": "R", "R": "L"}.get(name[0], name[0]) + name[1:]


def detect_beats(wav_path, min_bpm=70, max_bpm=170):
    """Beat times (s) of a song: onset strength, tempo by autocorrelation, then
    dynamic-programming beat tracking (Ellis 2007). numpy only."""
    with wave.open(wav_path) as wf:
        sr, ch = wf.getframerate(), wf.getnchannels()
        x = np.frombuffer(wf.readframes(wf.getnframes()), np.int16).reshape(-1, ch)
    x = x.mean(axis=1).astype(np.float32) / 32768.0

    n_fft, hop = 2048, 512
    n = 1 + (len(x) - n_fft) // hop
    win = np.hanning(n_fft).astype(np.float32)
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)
    band = (freqs > 40) & (freqs < 8000)
    logmag = np.empty((n, band.sum()), np.float32)
    for a in range(0, n, 512):  # in blocks, the full frame matrix would be huge
        idx = np.arange(a, min(a + 512, n))
        frames = x[idx[:, None] * hop + np.arange(n_fft)] * win
        logmag[a:a + len(idx)] = np.log1p(100 * np.abs(np.fft.rfft(frames, axis=1))[:, band])
    env = np.maximum(0, np.diff(logmag, axis=0)).sum(axis=1)
    fps = sr / hop
    k = int(fps * 0.5) | 1  # subtract a local average, keep only the peaks
    env = np.maximum(0, env - np.convolve(env, np.ones(k) / k, mode="same"))
    env /= env.std() + 1e-9

    # Tempo: strongest autocorrelation lag, biased toward 120 BPM
    ac = np.fft.irfft(np.abs(np.fft.rfft(env, 2 * len(env))) ** 2)[:len(env)]
    lags = np.arange(int(fps * 60 / max_bpm), int(fps * 60 / min_bpm) + 1)
    prior = np.exp(-0.5 * np.log2((60 * fps / lags) / 120) ** 2)
    period = int(lags[np.argmax(ac[lags] * prior)])

    # Beats: best-scoring chain of onsets spaced about one period apart
    score, back = env.copy(), np.full(len(env), -1)
    for t in range(period // 2, len(env)):
        lo, hi = max(0, t - 2 * period), t - period // 2
        if hi <= lo:
            continue
        prev = np.arange(lo, hi)
        cand = score[prev] - 100 * np.log((t - prev) / period) ** 2
        j = int(np.argmax(cand))
        if t < period or cand[j] > 0:
            if cand[j] > 0 or t >= period:
                score[t], back[t] = env[t] + cand[j], prev[j]
    end = len(env) - period + int(np.argmax(score[-period:]))
    beats = []
    while end >= 0:
        beats.append(end)
        end = back[end]
    # Frame f holds the energy rise just before its centre. Checked against the
    # bass onsets of a test song, the beats still came out ~45 ms early.
    return (np.array(beats[::-1]) * hop + n_fft / 2 - hop / 2) / sr + 0.045


def load_beats(folder):
    """Beat times for the song in `folder`, tracked once and cached as beats.npy."""
    path = os.path.join(folder, "beats.npy")
    if os.path.exists(path):
        return np.load(path)
    beats = detect_beats(os.path.join(folder, "audio.wav"))
    try:
        np.save(path, beats)
    except OSError:
        pass
    return beats


class Timeline:
    """timeline.npz from extract_dance.py, mirrored to match the flipped video.

    The video is flipped so the dancer moves like your mirror image. Flipping
    swaps which side of the screen each limb is on, so L/R names are swapped
    and x is mirrored. The angles themselves don't change.
    """

    def __init__(self, folder):
        d = np.load(os.path.join(folder, "timeline.npz"))
        self.fps = float(d["fps"])
        self.n = len(d["t"])
        self.duration = self.n / self.fps
        self.tracked = d["tracked"]
        src_w, src_h = d["frame_size"] if "frame_size" in d.files else (1920, 1080)
        self.img = d["image"] / np.array([src_w, src_h])  # normalised 0..1
        self.img[..., 0] = 1.0 - self.img[..., 0]
        self.joints = [flip_name(str(j)) for j in d["joint_names"]]
        self.angle_names = [flip_name(str(a)) for a in d["angle_names"]]
        self.angles = d["angles"]
        self.beats = load_beats(folder)
        if not SCORE_IN_3D:  # 2D angles where the 3D angle was trustworthy
            px = self.img * (src_w, src_h)
            j = {k: n for n, k in enumerate(self.joints)}
            for k, (a, b, c) in enumerate(ANGLES[name] for name in self.angle_names):
                for f in np.flatnonzero(~np.isnan(self.angles[:, k])):
                    self.angles[f, k] = joint_angle(px[f, j[a]], px[f, j[b]], px[f, j[c]])

    def index(self, t):
        return int(np.clip(round(t * self.fps), 0, self.n - 1))

    def angles_at(self, i):
        return {k: float(v) for k, v in zip(self.angle_names, self.angles[i])
                if not np.isnan(v)}

    def points_at(self, i, w, h):
        return {k: p * (w, h) for k, p in zip(self.joints, self.img[i])
                if not np.isnan(p).any()}


class AudioClock:
    """Plays a .wav; time() is the song position actually reaching the speakers."""

    def __init__(self, path):
        import sounddevice as sd
        with wave.open(path) as wf:
            self.sr, channels = wf.getframerate(), wf.getnchannels()
            raw = wf.readframes(wf.getnframes())
        data = np.frombuffer(raw, np.int16).reshape(-1, channels)
        self.data = (data / 32768.0).astype(np.float32)
        self.pos = 0
        self.paused = True
        self.stream = sd.OutputStream(samplerate=self.sr, channels=channels,
                                      dtype="float32", callback=self._fill)
        self.stream.start()

    def _fill(self, out, frames, time_info, status):
        if self.paused:
            out[:] = 0
            return
        chunk = self.data[self.pos:self.pos + frames]
        out[:len(chunk)] = chunk
        out[len(chunk):] = 0
        self.pos += len(chunk)

    def time(self):
        return max(0.0, self.pos / self.sr - self.stream.latency)

    def finished(self):
        return self.pos >= len(self.data)

    def restart(self):
        self.paused, self.pos = True, 0

    def close(self):
        self.stream.stop()
        self.stream.close()


class VideoPanel:
    """Random access into the 720p copy, cheap when moving forward."""

    def __init__(self, path):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise SystemExit(f"Couldn't open {path} - re-run extract_dance.py")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.idx, self.frame = -1, None
        self.frame_at(0)

    def frame_at(self, t):
        i = int(np.clip(round(t * self.fps), 0, max(self.n - 1, 0)))
        if i < self.idx or i > self.idx + 60:  # backwards or far ahead: seek
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            self.idx = i - 1
        while self.idx < i - 1 and self.cap.grab():
            self.idx += 1
        if self.idx < i:
            ok, frame = self.cap.read()
            if ok:
                self.frame, self.idx = frame, i
        return self.frame


def fit_ghost(target, player):
    """Hang the target's limbs off the player's own shoulders and hips.

    Limb directions come from the target and lengths are scaled by torso
    size, so the ghost sits on your body and shows where each limb should go.
    """
    if not all(k in target for k in TORSO) or not all(k in player for k in TORSO):
        return None

    def torso(p):
        sh = (p["Lshoulder"] + p["Rshoulder"]) / 2
        return np.linalg.norm(sh - (p["Lhip"] + p["Rhip"]) / 2), sh

    t_len, t_sh = torso(target)
    p_len, p_sh = torso(player)
    if t_len < 1:
        return None
    scale = p_len / t_len
    ghost = {k: player[k] for k in TORSO}
    if "head" in target:
        ghost["head"] = p_sh + (target["head"] - t_sh) * scale
    for chain in CHAINS:
        for a, b in zip(chain, chain[1:]):
            if a in ghost and a in target and b in target:
                ghost[b] = ghost[a] + (target[b] - target[a]) * scale
    return ghost


def ipt(p):
    return tuple(int(v) for v in p)


def draw_skeleton(img, pts, color_of, thick, head=True):
    for a, b in BONES:
        if a in pts and b in pts:
            cv2.line(img, ipt(pts[a]), ipt(pts[b]), color_of(a, b), thick, cv2.LINE_AA)
    if head and "head" in pts and "Lshoulder" in pts and "Rshoulder" in pts:
        r = max(8, int(np.linalg.norm(pts["Lshoulder"] - pts["Rshoulder"]) * 0.3))
        cv2.circle(img, ipt(pts["head"]), r, color_of("head", "head"), thick, cv2.LINE_AA)


def draw_ghost(img, pts, color, alpha, thick):
    if not pts:
        return
    layer = img.copy()
    draw_skeleton(layer, pts, lambda a, b: color, thick)
    cv2.addWeighted(layer, alpha, img, 1 - alpha, 0, img)


def limb_color(errs):
    """Per-bone colour: green on target -> red at LIMB_TOLERANCE, grey if unknown."""
    def color_of(a, b):
        angle = LIMB_ANGLE.get((a, b))
        if angle is None:
            return (235, 235, 235)  # torso
        if angle not in errs:
            return (130, 130, 130)
        return score_color(1 - min(errs[angle] / LIMB_TOLERANCE, 1.0))
    return color_of


def dance_score(player, tl, tt):
    """(match, frame) - the best match against the dancer's pose within +-WINDOW
    of song time tt, and the frame it came from.

    (None, None) when there is nothing to score against: tt is before the song
    starts, or the timeline lost the dancer at tt.
    """
    i = tl.index(tt)
    if tt < 0 or not tl.tracked[i]:
        return None, None
    best, best_j = 0.0, i
    step = max(1, int(tl.fps * 0.05))
    for j in range(tl.index(tt - WINDOW), tl.index(tt + WINDOW) + 1, step):
        target = tl.angles_at(j)
        if tl.tracked[j] and all(k in target for k in REQUIRED):
            m = match_score(player, target)
            if m > best:
                best, best_j = m, j
    return best, best_j


PHOTO_DIR = "photos"
RESULT_TIMEOUT = 20.0   # seconds on the result screen before going back to idle
NAME_MAX = 12
FX_TIME = 0.4           # how long the hit effect lasts
# Grade = 70% average accuracy + 30% how much of the song the best streak covered
GRADES = [("S", 0.68, (0, 215, 255)), ("A", 0.52, (90, 230, 90)),
          ("B", 0.38, (255, 190, 70)), ("C", 0.00, (110, 110, 255))]


def grade_for(avg_match, best_combo, judged):
    streak = best_combo / judged if judged else 0.0
    value = 0.7 * avg_match + 0.3 * streak
    return next((g, c) for g, th, c in GRADES if value >= th)


class DanceGame:
    def __init__(self, beats=()):
        self.beats = np.asarray(beats, float)
        self.reset()

    def reset(self):
        self.score = self.combo = self.best_combo = self.gained = 0
        self.counts = {r[0]: 0 for r in RATINGS}
        self.matches = []  # match 0..1 of every rated beat
        self.samples = deque()  # (song time, match or None) of recent frames
        self.beat_i = 0
        self.rating, self.rating_t = None, -1e9
        self.done = False

    @property
    def accuracy(self):
        return float(np.mean(self.matches)) if self.matches else 0.0

    def grade(self):
        return grade_for(self.accuracy, self.best_combo, len(self.matches))

    def update(self, t, live):
        """live = match 0..1 this frame, or None if the frame isn't scored.
        Rates each beat once BEAT_HALF has passed it; True if one was rated."""
        self.samples.append((t, live))
        rated = False
        while self.beat_i < len(self.beats) and t >= self.beats[self.beat_i] + BEAT_HALF:
            b = self.beats[self.beat_i]
            self.beat_i += 1
            near = [m for tt, m in self.samples if abs(tt - b) <= BEAT_HALF]
            scored = [m for m in near if m is not None]
            if scored and len(scored) >= MIN_SCORED * len(near):
                self._judge(float(np.mean(scored)), t)
                rated = True
        while self.samples and self.samples[0][0] < t - 4 * BEAT_HALF:
            self.samples.popleft()
        return rated

    def _judge(self, match, t):
        name, points = next((n, p) for n, th, p, _ in RATINGS if match >= th)
        self.rating, self.rating_t = name, t
        self.counts[name] += 1
        self.matches.append(match)
        if name == "MISS":
            self.combo = self.gained = 0
        else:
            self.combo += 1
            self.best_combo = max(self.best_combo, self.combo)
            self.gained = int(points * (1 + 0.1 * (self.combo - 1)))
            self.score += self.gained


def draw_fx(cam, fx, t):
    """Hit effect on the player panel: colour flash (stronger with combo),
    border glow, an expanding ring and sparks. Returns False once it's over."""
    age = t - fx["t"]
    if age < 0 or age > FX_TIME:
        return age < 0
    k = 1 - age / FX_TIME
    color, combo, hit = fx["color"], fx["combo"], fx["rating"] != "MISS"
    h, w = cam.shape[:2]

    flash = (min(0.10 + 0.025 * combo, 0.38) if hit else 0.18) * k
    cv2.addWeighted(np.full_like(cam, color), flash, cam, 1 - flash, 0, cam)

    layer = cam.copy()
    cv2.rectangle(layer, (0, 0), (w - 1, h - 1), color, 4 + 2 * min(combo, 8))
    if hit:
        cx, cy = ipt(fx["center"])
        for lag, width in ((0.0, 6), (0.12, 3)) if combo >= 5 else ((0.0, 6),):
            r = int(30 + max(age - lag, 0) * 700)
            cv2.circle(layer, (cx, cy), r, color, max(1, int(width * k)), cv2.LINE_AA)
        for ang, speed in zip(fx["angles"], fx["speeds"]):
            d = np.array([np.cos(ang), np.sin(ang)])
            p0 = (cx + d[0] * speed * age * 0.8, cy + d[1] * speed * age * 0.8)
            p1 = (cx + d[0] * speed * age, cy + d[1] * speed * age)
            cv2.line(layer, ipt(p0), ipt(p1), (255, 255, 255), 3, cv2.LINE_AA)
    cv2.addWeighted(layer, 0.9 * k, cam, 1 - 0.9 * k, 0, cam)
    return True


def photo_card(raw, name, score, grade, peak):
    """Best-moment webcam frame with name, score and grade stamped on it."""
    img = raw.copy()
    h, w = img.shape[:2]
    letter, color = grade
    shade(img, 0, h - 170, w, h, 0.65)
    put(img, name.upper(), (40, h - 100), 2.2, (255, 255, 255), 3)
    put(img, f"SCORE {score}", (40, h - 40), 1.5, (0, 215, 255), 2)
    put(img, f"best moment {int(peak * 100)}%", (w - 380, h - 100), 0.9,
        (230, 230, 230), 1)
    put(img, time.strftime("%d %b %Y"), (w - 380, h - 50), 0.9, (230, 230, 230), 1)
    cv2.circle(img, (w - 110, 110), 80, (0, 0, 0), -1, cv2.LINE_AA)
    cv2.circle(img, (w - 110, 110), 80, color, 6, cv2.LINE_AA)
    put(img, letter, (w - 110, 110), 3.4, color, 6, center=True)
    return img


def save_photo(img, name):
    safe = "".join(c for c in name if c.isalnum() or c in "-_ ").strip().replace(" ", "_")
    os.makedirs(PHOTO_DIR, exist_ok=True)
    path = os.path.join(PHOTO_DIR, f"{safe or 'player'}_{time.strftime('%Y%m%d_%H%M%S')}.jpg")
    return path if cv2.imwrite(path, img) else None


def draw_mini(img, pts, box, color):
    """Small stick figure of `pts`, scaled to fit box=(x0,y0,x1,y1)."""
    x0, y0, x1, y1 = box
    arr = np.array(list(pts.values()))
    mn, mx = arr.min(0), arr.max(0)
    span = np.maximum(mx - mn, 1e-3)
    scale = 0.85 * min((x1 - x0) / span[0], (y1 - y0) / span[1])
    mid = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
    P = {k: (v - (mn + mx) / 2) * scale + mid for k, v in pts.items()}
    for a, b in BONES:
        if a in P and b in P:
            cv2.line(img, ipt(P[a]), ipt(P[b]), color, 2, cv2.LINE_AA)
    if all(k in P for k in ("head", "Lshoulder", "Rshoulder")):
        r = max(3, int(np.linalg.norm(P["Lshoulder"] - P["Rshoulder"]) * 0.3))
        cv2.circle(img, ipt(P["head"]), r, color, 2, cv2.LINE_AA)


def check_in_frame(pts, vis, cw, h):
    """Upper body fully visible, centred, and neither too close nor too far."""
    need = ("Lshoulder", "Rshoulder", "Lelbow", "Relbow", "Lhip", "Rhip")
    if any(vis[k] < 0.5 for k in need):
        return False
    if any(not (0.02 * cw < pts[k][0] < 0.98 * cw and 0.02 * h < pts[k][1] < 0.98 * h)
           for k in need):
        return False
    shoulders = (pts["Lshoulder"] + pts["Rshoulder"]) / 2
    torso = np.linalg.norm(shoulders - (pts["Lhip"] + pts["Rhip"]) / 2) / h
    return 0.10 <= torso <= 0.40 and 0.3 * cw < shoulders[0] < 0.7 * cw


class DanceSession:
    """Everything for one song except the webcam, window and audio device."""

    def __init__(self, folder):
        self.tl = Timeline(folder)
        self.video = VideoPanel(os.path.join(folder, "video_720p.mp4"))
        self.game = DanceGame(self.tl.beats)
        self.rng = np.random.default_rng()
        self.split = 0
        self.start_run("PLAYER", "medium")

    def start_run(self, player, difficulty):
        """Fresh score, effects and best-moment photo for this player."""
        self.player, self.difficulty = player, difficulty
        self.delay = DIFFICULTY[difficulty]
        self.game.reset()
        self.pts = self.world = None
        self.in_frame, self.warmup = False, 0.0
        self.fx = []
        self.recent = deque(maxlen=5)   # best moment = best 5-frame average
        self.best_peak, self.best_frame = -1.0, None
        self.photo = self.photo_path = self.grade = None
        self.result_t0 = 0.0

    def finish(self):
        """Song over: grade it, stamp and save the best-moment photo."""
        g = self.game
        g.done = True
        self.grade = g.grade()
        if self.best_frame is not None:
            self.photo = photo_card(self.best_frame, self.player, g.score,
                                    self.grade, self.best_peak)
            self.photo_path = save_photo(self.photo, self.player)
            print(f"Photo: {self.photo_path or 'could not be saved'}")
        self.result_t0 = time.monotonic()

    def render(self, cam, result, t, playing):
        """cam: mirrored webcam frame, result: its PoseLandmarker result,
        t: song time from the audio clock. Returns the split-screen frame.

        The dancer on the left is live; you are scored against the pose the
        dancer struck `delay` seconds earlier (tt), best match within +-WINDOW.
        """
        tl, h = self.tl, PANEL_H
        tt = t - self.delay
        i, ti = tl.index(t), tl.index(max(tt, 0.0))
        tracked = bool(tl.tracked[i])
        target_ok = tt >= 0 and bool(tl.tracked[ti])
        raw = cam

        # Left: the dancer, flipped, with the extracted skeleton
        vid = cv2.flip(self.video.frame_at(t), 1)
        vw = int(h * vid.shape[1] / vid.shape[0])
        vid = cv2.resize(vid, (vw, h))
        dancer = tl.points_at(i, vw, h)
        line = (255, 255, 255) if tracked else (120, 120, 120)
        draw_skeleton(vid, dancer, lambda a, b: (0, 0, 0), 6)
        draw_skeleton(vid, dancer, lambda a, b: line, 3)

        # Right: you
        cw = int(h * cam.shape[1] / cam.shape[0])
        cam = cv2.resize(cam, (cw, h))
        player_angles, vis, has_player = {}, {}, bool(result.pose_landmarks)
        if has_player:
            lms = result.pose_landmarks[0]
            pts, vis = landmarks_to_points(lms, cw, h)
            world = landmarks_to_world(lms, result.pose_world_landmarks[0])
            if self.pts is None:
                self.pts, self.world = pts, world
            else:
                self.pts = {k: SMOOTH * self.pts[k] + (1 - SMOOTH) * v
                            for k, v in pts.items()}
                self.world = {k: SMOOTH * self.world[k] + (1 - SMOOTH) * v
                              for k, v in world.items()}
            player_angles = pose_angles(self.world if SCORE_IN_3D else self.pts, vis)
            self.in_frame = check_in_frame(self.pts, vis, cw, h)
        else:
            self.pts = self.world = None
            self.in_frame = False

        # Not in frame = 0, but an arm the camera can't see right now (mid-turn)
        # just isn't scored rather than counting as a miss
        arms_seen = all(k in player_angles for k in REQUIRED)
        live, best_j = ((None, None) if has_player and not arms_seen
                        else dance_score(player_angles, tl, tt))

        if has_player:
            visible = {k: p for k, p in self.pts.items() if vis[k] > 0.5}
            errs = {}
            if best_j is not None:  # error against the frame that scored best
                target = tl.angles_at(best_j)
                errs = {k: abs(v - target[k]) for k, v in player_angles.items()
                        if k in target}
            ahead = tl.index(tt + LOOKAHEAD)
            if tt + LOOKAHEAD >= 0 and tl.tracked[ahead]:
                draw_ghost(cam, fit_ghost(tl.points_at(ahead, vw, h), visible),
                           NEXT_COLOR, 0.35, 5)
            if target_ok:
                draw_ghost(cam, fit_ghost(tl.points_at(ti, vw, h), visible),
                           GHOST_COLOR, 0.55, 9)
            draw_skeleton(cam, visible, limb_color(errs), 5, head=False)
            for p in visible.values():
                cv2.circle(cam, ipt(p), 5, (255, 255, 255), -1, cv2.LINE_AA)

        if playing:
            if self.game.update(t, live):
                self._add_fx(t, (cw, h))
            self._track_best(raw, live, has_player)
        self.warmup = max(0.0, -tt) if playing else 0.0

        self.fx = [f for f in self.fx if draw_fx(cam, f, t)]
        self.draw_hints(cam, t, vw)

        self.split = vw
        canvas = np.hstack([vid, cam])
        self.draw_hud(canvas, vw, t, live, tracked, has_player)
        if self.game.done:
            self.draw_result(canvas)
        return canvas

    def draw_hints(self, cam, t, vw):
        """Corner strip: the poses you will be scored against on the next beats."""
        tl, g = self.tl, self.game
        h, w = cam.shape[:2]
        nxt = g.beats[np.searchsorted(g.beats, t, side="right"):][:HINTS]
        if not len(nxt):
            return
        bw, bh, pad = 100, 140, 8
        x0, y0 = w - HINTS * (bw + pad) - pad - 10, h - bh - 40
        shade(cam, x0 - pad, y0 - 28, w - 10 + pad // 2, y0 + bh + 6, 0.6)
        put(cam, "NEXT", (x0, y0 - 10), 0.5, (200, 200, 200), 1)
        for k, b in enumerate(nxt):
            bx = x0 + k * (bw + pad)
            tp = b - self.delay  # the dancer pose you are rated against on that beat
            if tp >= 0 and tl.tracked[tl.index(tp)]:
                v = int(255 * (1 - 0.22 * k))
                draw_mini(cam, tl.points_at(tl.index(tp), vw, h),
                          (bx, y0, bx + bw, y0 + bh - 18), (v, v, v))
            else:
                put(cam, "?", (bx + bw // 2, y0 + bh // 2 - 10), 1.0, (150, 150, 150),
                    1, center=True)
            put(cam, f"{b - t:.1f}s", (bx + bw // 2, y0 + bh - 6), 0.5,
                (0, 215, 255) if k == 0 else (170, 170, 170), 1, center=True)

    def draw_outline(self, img, ok):
        """'Get in frame' guide on the player panel: green once you fit it."""
        h, w = img.shape[:2]
        cx, u = self.split + (w - self.split) // 2, h  # figure in units of panel height
        color = (90, 230, 90) if ok else (90, 90, 255)
        pt = lambda dx, dy: (int(cx + dx * u), int(dy * u))
        layer = img.copy()
        cv2.circle(layer, pt(0, 0.20), int(0.065 * u), color, 6, cv2.LINE_AA)
        for a, b in [((-.11, .33), (.11, .33)), ((-.11, .33), (-.07, .60)),
                     ((.11, .33), (.07, .60)), ((-.07, .60), (.07, .60)),
                     ((-.11, .33), (-.20, .46)), ((-.20, .46), (-.22, .58)),
                     ((.11, .33), (.20, .46)), ((.20, .46), (.22, .58)),
                     ((-.07, .60), (-.08, .76)), ((-.08, .76), (-.09, .93)),
                     ((.07, .60), (.08, .76)), ((.08, .76), (.09, .93))]:
            cv2.line(layer, pt(*a), pt(*b), color, 6, cv2.LINE_AA)
        cv2.addWeighted(layer, 0.7, img, 0.3, 0, img)

    def _add_fx(self, t, panel):
        g = self.game
        color = next(c for n, *_, c in RATINGS if n == g.rating)
        p = self.pts
        center = (p["Lhip"] + p["Rhip"]) / 2 if p else np.array(panel) / 2
        n = 0 if g.rating == "MISS" else 10 + 2 * min(g.combo, 15)
        self.fx.append({"t": t, "rating": g.rating, "combo": g.combo, "color": color,
                        "center": center, "angles": self.rng.uniform(0, 2 * np.pi, n),
                        "speeds": self.rng.uniform(250, 520, n)})

    def _track_best(self, raw, live, has_player):
        if live is None or not has_player:
            self.recent.clear()
            return
        self.recent.append(live)
        if len(self.recent) == self.recent.maxlen:
            avg = float(np.mean(self.recent))
            if avg > self.best_peak:
                self.best_peak, self.best_frame = avg, raw.copy()

    def draw_hud(self, img, split, t, live, tracked, has_player):
        g, h, w = self.game, img.shape[0], img.shape[1]
        cx = split + (w - split) // 2  # centre of the player panel

        cv2.line(img, (split, 0), (split, h), (0, 0, 0), 3)
        put(img, "DANCER", (15, 35), 0.8, (0, 215, 255))
        put(img, f"you copy this {self.delay:g}s later", (15, 65), 0.55,
            (230, 230, 230), 1)
        if not tracked:
            put(img, "dancer lost - not scored", (15, h - 40), 0.7, (90, 90, 255), 1)

        shade(img, split + 10, 10, split + 330, 140)
        put(img, f"{self.player.upper()}  {self.difficulty.upper()}",
            (split + 25, 30), 0.6, (200, 200, 200), 1)
        put(img, f"SCORE {g.score}", (split + 25, 62), 1.0)
        put(img, f"COMBO x{g.combo}", (split + 25, 92), 0.7, (0, 215, 255))
        m = live or 0.0
        bar(img, split + 25, 106, 290, 22, m,
            score_color(m) if live is not None else (90, 90, 90))
        put(img, "--" if live is None else f"{int(m * 100)}%", (split + 150, 124), 0.6,
            (255, 255, 255), 1)

        age = t - g.rating_t
        if g.rating and age < BEAT_FEEDBACK:
            color = next(c for n, *_, c in RATINGS if n == g.rating)
            rx = (split + 340 + w) // 2  # top of the panel, clear of the score box
            pop = 1 + 0.35 * max(0.0, 1 - age / 0.2)  # lands big, settles
            put(img, g.rating, (rx, 55), 1.8 * pop, color, 4, center=True)
            if g.gained:
                put(img, f"+{g.gained}", (rx, 110), 1.0, color, 2, center=True)
        if self.warmup > 0:
            put(img, f"copy the dancer in {self.warmup:.1f}s", (cx, h // 2), 1.0,
                (0, 215, 255), 2, center=True)
        if not has_player:
            put(img, "Step back so I can see you!", (cx, h - 70), 0.9, (90, 90, 255),
                center=True)

        bar(img, 10, h - 16, w - 20, 8, t / self.tl.duration, (0, 215, 255))

    def draw_result(self, img):
        g, h, w = self.game, img.shape[0], img.shape[1]
        letter, gcolor = self.grade
        shade(img, 0, 0, w, h, 0.88)
        put(img, "SHOW'S OVER!", (w // 2, 45), 1.6, (0, 215, 255), 3, center=True)

        # Left: the best-moment photo
        if self.photo is not None:
            ph = 320
            pw = int(ph * self.photo.shape[1] / self.photo.shape[0])
            x0, y0 = (w // 2 - pw) // 2, 80
            img[y0:y0 + ph, x0:x0 + pw] = cv2.resize(self.photo, (pw, ph))
            cv2.rectangle(img, (x0 - 3, y0 - 3), (x0 + pw + 3, y0 + ph + 3),
                          (255, 255, 255), 3)
            caption = f"saved to {self.photo_path}" if self.photo_path else "best moment"
            put(img, caption, (w // 4, y0 + ph + 28), 0.6, (200, 200, 200), 1, center=True)
        else:
            put(img, "No photo - I never saw you!", (w // 4, h // 2), 0.9,
                (150, 150, 150), center=True)

        # Right: grade and stats
        put(img, letter, (w // 2 + 200, 220), 7.0, gcolor, 14, center=True)
        sx = w // 2 + 400
        put(img, self.player.upper(), (sx, 120), 1.0, (255, 255, 255))
        put(img, f"Score  {g.score}", (sx, 175), 1.2, (0, 215, 255))
        put(img, f"Accuracy  {int(g.accuracy * 100)}%", (sx, 220), 0.9)
        put(img, f"Best combo  x{g.best_combo}", (sx, 260), 0.9)
        put(img, "  ".join(f"{n} {g.counts[n]}" for n, *_ in RATINGS),
            (sx, 300), 0.6, (200, 200, 200), 1)
        put(img, f"{self.difficulty.upper()}  ({self.delay:g}s behind the dancer)",
            (sx, 335), 0.6, (200, 200, 200), 1)

        left = max(0, int(np.ceil(RESULT_TIMEOUT - (time.monotonic() - self.result_t0))))
        put(img, "R  play again        N  new player        Q  quit",
            (w // 2, h - 75), 1.0, (255, 255, 255), 2, center=True)
        put(img, f"back to the start screen in {left}s", (w // 2, h - 35), 0.7,
            (160, 160, 160), 1, center=True)

    def render_idle(self, cam, result, name, difficulty, now):
        """Start screen: type a name, TAB for difficulty, ENTER to dance."""
        h = PANEL_H
        vh, vw_src = self.video.frame.shape[:2]
        vw = int(h * vw_src / vh)
        cw = int(h * cam.shape[1] / cam.shape[0])

        left = np.linspace((60, 25, 30), (25, 10, 15), h).astype(np.uint8)[:, None, :]
        left = np.repeat(left, vw, axis=1)
        put(left, "POSE PARTY", (vw // 2, 100), 2.4, (0, 215, 255), 5, center=True)
        put(left, "Copy the dancer. Keep the combo going.", (vw // 2, 160), 0.8,
            (220, 220, 220), 1, center=True)
        put(left, "Your name", (vw // 2, 230), 0.7, (180, 180, 180), 1, center=True)
        cv2.rectangle(left, (vw // 2 - 220, 250), (vw // 2 + 220, 310), (255, 255, 255), 2)
        cursor = "_" if int(now * 2) % 2 == 0 else " "
        put(left, (name + cursor) if name else cursor, (vw // 2, 281), 1.1,
            (255, 255, 255), 2, center=True)
        put(left, f"TAB  difficulty: {difficulty.upper()}", (vw // 2, 365), 0.9,
            (255, 190, 70), 2, center=True)
        put(left, f"(you copy the dancer {DIFFICULTY[difficulty]:g}s later)",
            (vw // 2, 398), 0.6, (200, 200, 200), 1, center=True)
        put(left, "ENTER  start", (vw // 2, 450), 1.0, (90, 230, 90), 2, center=True)
        put(left, "ESC  quit", (vw // 2, 492), 0.8, (160, 160, 160), 1, center=True)

        right = cv2.resize(cam, (cw, h))
        seen = bool(result.pose_landmarks)
        if seen:
            pts, vis = landmarks_to_points(result.pose_landmarks[0], cw, h)
            draw_skeleton(right, {k: p for k, p in pts.items() if vis[k] > 0.5},
                          lambda a, b: (90, 230, 90), 4)
        put(right, "I can see you!" if seen else "Step back so I can see you!",
            (cw // 2, h - 40), 0.9, (90, 230, 90) if seen else (90, 90, 255), center=True)
        return np.hstack([left, right])


def dance_main(folder, camera=0, difficulty="medium"):
    ensure_model()
    landmarker = make_landmarker()
    session = DanceSession(folder)
    audio = AudioClock(os.path.join(folder, "audio.wav"))

    cap = cv2.VideoCapture(camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        raise SystemExit("Couldn't open the webcam. Try --camera 1 if you have several.")

    window = "Pose Party - Dance"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 1600, 450)
    state, name, countdown_t0 = "idle", "", 0.0   # idle -> countdown -> play -> done
    last_ts = 0

    def begin(player):
        nonlocal state, countdown_t0
        audio.restart()
        session.start_run(player, difficulty)
        state, countdown_t0 = "countdown", time.monotonic()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)  # mirror view
        ts = max(int(time.monotonic() * 1000), last_ts + 1)
        last_ts = ts
        result = landmarker.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB,
                     data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), ts)
        now = time.monotonic()

        if state == "idle":
            canvas = session.render_idle(frame, result, name, difficulty, now)
        else:
            if state == "play" and audio.finished():
                session.finish()
                state = "done"
            t = min(audio.time(), session.tl.duration - 1e-3)
            canvas = session.render(frame, result, t,
                                    playing=state == "play" and not audio.paused)
            ch, cw = canvas.shape[:2]
            px = session.split + (cw - session.split) // 2  # centre of the player panel
            if state == "countdown":
                # The 3-2-1 only runs while you are in the outline; stepping out resets it
                session.draw_outline(canvas, session.in_frame)
                if not session.in_frame:
                    countdown_t0 = now
                    put(canvas, "GET IN FRAME", (px, ch // 2 - 20), 1.6, (90, 90, 255), 3,
                        center=True)
                    put(canvas, "stand inside the outline   (SPACE to skip)",
                        (px, ch // 2 + 35), 0.7, (255, 255, 255), 1, center=True)
                else:
                    n = int(np.ceil(COUNTDOWN - (now - countdown_t0)))
                    put(canvas, str(max(n, 1)), (px, ch // 2), 5.0, (0, 215, 255), 8,
                        center=True)
                if now - countdown_t0 >= COUNTDOWN:
                    state, audio.paused = "play", False
            elif state == "play" and audio.paused:
                put(canvas, "PAUSED", (cw // 2, ch // 2), 2.5, (0, 215, 255), 5,
                    center=True)
            if state == "done" and now - session.result_t0 >= RESULT_TIMEOUT:
                audio.restart()
                state, name = "idle", ""
        cv2.imshow(window, canvas)

        key = cv2.waitKey(1) & 0xFF
        if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
            break
        if key == 255:
            continue
        if state == "idle":  # letters go into the name, so only ESC quits here
            if key == 27:
                break
            if key in (13, 10):
                begin(name.strip() or "Player")
            elif key == 9:  # TAB
                levels = list(DIFFICULTY)
                difficulty = levels[(levels.index(difficulty) + 1) % len(levels)]
            elif key in (8, 127):
                name = name[:-1]
            elif 32 <= key < 127 and len(name) < NAME_MAX:
                name += chr(key)
            continue
        if key in (ord("q"), 27):
            break
        if state == "done":
            session.result_t0 = time.monotonic()  # any key restarts the idle timer
            if key == ord("r"):
                begin(session.player)
            elif key == ord("n"):
                audio.restart()
                state, name = "idle", ""
        elif key == ord("r"):
            begin(session.player)
        elif key == ord(" "):
            if state == "play":
                audio.paused = not audio.paused
            elif state == "countdown":  # skip the in-frame check
                state, audio.paused = "play", False

    audio.close()
    cap.release()
    landmarker.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Pose Party")
    ap.add_argument("--dance", metavar="FOLDER",
                    help="dance along to an extract_dance.py output folder")
    ap.add_argument("--difficulty", choices=list(DIFFICULTY), default="medium",
                    help="how late you copy the dancer: " +
                    ", ".join(f"{k} {v:g}s" for k, v in DIFFICULTY.items()))
    ap.add_argument("--camera", default="0",
                    help="webcam index (or a video file, for testing)")
    args = ap.parse_args()
    if args.dance:
        dance_main(args.dance, int(args.camera) if args.camera.isdigit() else args.camera,
                   args.difficulty)
    else:
        main()
