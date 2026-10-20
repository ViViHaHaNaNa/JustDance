"""
Pose Party - a tiny Just Dance-style game with OpenCV + MediaPipe.

Copy the stick figure in the top-right corner before the timer runs out.
Hold the pose steady to lock in your best score for that round.

Setup:
    pip install -r requirements.txt
    python pose_party.py                              # random poses
    python pose_party.py --dance output/<folder>      # dance along to a song

Dance mode plays a timeline made by extract_dance.py: the dancer (mirrored)
on the left, you on the right. You are scored against the dancer's pose right
now (best joint-angle match within +-0.3 s). Key moves scroll along the bottom
bar toward the hit zone, one every 2-3 s, and get rated OK / GOOD / PERFECT as
they land; between moves, roughly matching the dancer earns small bonus points.
Start screen: type your name, ENTER = dance (a 3-2-1 count-in waits until you
stand inside the outline; SPACE skips that check). At the end you get a grade
and a photo of your best moment saved in photos/.
Result screen: R = replay, N = new player, Q = quit (20 s idle -> start screen).

Keys:
    Q / ESC  quit
    R        restart
    SPACE    judge the current pose right now (pause/resume in dance mode)
"""

import argparse
import json
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
WINDOW = 0.3         # you match the dancer if you hit their pose within +-this many seconds
BAR_H = 190          # height of the move bar under the two panels
BAR_SECONDS = 6.0    # how far ahead the bar looks
HIT_X = 120          # x of the hit zone on the bar
APPROACH = 3.0       # the ring closes in and the 3-2-1 runs for this long before a move
MOVE_FADE = 0.5      # a landed move slides off the left of the bar for this long
MOVE_BEFORE = 0.2    # a move is rated on your best frames from this long before...
MOVE_AFTER = 0.1     # ...to this long after it lands
MOVE_FEEDBACK = 0.8  # how long OK / GOOD / PERFECT stays up
# Standing still shouldn't score: over the last second, if the dancer moved this much
# (average range of their joint angles, degrees) and you moved much less, the match is
# scaled down. A static pose happens to resemble some frame of almost any dance.
STILL_WINDOW = 1.0
STILL_DANCER_MIN = 10.0
STILL_FLOOR = 0.4    # a completely still player keeps this fraction of their match
BONUS_MATCH = 0.5    # between moves, matching the dancer at least this well earns bonus
BONUS_EVERY = 0.25   # ...one bonus every this many seconds of matching
BONUS_POINTS = 10
BONUS_CLEAR = 0.5    # no bonus this close to a move (those are rated instead)
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
        self.aspect = src_w / src_h
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
        # Shoulder width / torso length: collapses to a line when the dancer turns side-on
        pts2 = self.img * [self.aspect, 1.0]
        ix = {k: n for n, k in enumerate(self.joints)}
        torso = np.linalg.norm((pts2[:, ix["Lshoulder"]] + pts2[:, ix["Rshoulder"]]) / 2
                               - (pts2[:, ix["Lhip"]] + pts2[:, ix["Rhip"]]) / 2, axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            self.shoulder_ratio = (np.linalg.norm(pts2[:, ix["Lshoulder"]]
                                                  - pts2[:, ix["Rshoulder"]], axis=1) / torso)
        self.shoulder_ratio[~self.tracked] = np.nan
        self.moves = load_moves(self, folder)

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


def ipt(p):
    return tuple(int(v) for v in p)


def draw_skeleton(img, pts, color_of, thick, head=True):
    for a, b in BONES:
        if a in pts and b in pts:
            cv2.line(img, ipt(pts[a]), ipt(pts[b]), color_of(a, b), thick, cv2.LINE_AA)
    if head and "head" in pts and "Lshoulder" in pts and "Rshoulder" in pts:
        r = max(8, int(np.linalg.norm(pts["Lshoulder"] - pts["Rshoulder"]) * 0.3))
        cv2.circle(img, ipt(pts["head"]), r, color_of("head", "head"), thick, cv2.LINE_AA)


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


TURN_RATIO = 0.24    # shoulders this narrow (shoulder width / torso length) = turned side-on


def pose_extent(p):
    """How far the wrists reach from the body, in torso lengths (0 = arms tucked in)."""
    need = ("Lshoulder", "Rshoulder", "Lhip", "Rhip", "Lwrist", "Rwrist")
    if not all(k in p for k in need):
        return 0.0
    sh = (p["Lshoulder"] + p["Rshoulder"]) / 2
    torso = np.linalg.norm(sh - (p["Lhip"] + p["Rhip"]) / 2)
    if torso < 1e-6:
        return 0.0
    return float(max(np.linalg.norm(p[w] - sh) for w in ("Lwrist", "Rwrist")) / torso)


def classify_pose(p, angles=None, return_all=False, min_clarity=0.35):
    """(label, clarity 0..1) for a pose given as screen-side joint points
    (mirrored, y down), or (None, 0) when it is just standing around.

    Labels use the player's own sides: in the mirrored view, "Left" is the
    left of the screen, which is your left arm when you copy it. Turning around
    is handled separately, from how narrow the shoulders get (see pick_key_moves).
    """
    need = ("Lshoulder", "Rshoulder", "Lhip", "Rhip", "Lwrist", "Rwrist")
    if not all(k in p for k in need):
        return None, 0.0
    sh, hp = (p["Lshoulder"] + p["Rshoulder"]) / 2, (p["Lhip"] + p["Rhip"]) / 2
    torso = np.linalg.norm(sh - hp)
    if torso < 1e-6:
        return None, 0.0
    ramp = lambda x, lo, hi: float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))

    arm = {}
    for side, out_sign in (("L", -1), ("R", 1)):
        w, s_ = p[side + "wrist"], p[side + "shoulder"]
        dy = (w[1] - s_[1]) / torso                     # < 0: above the shoulder
        out = out_sign * (w[0] - s_[0]) / torso         # > 0: away from the body
        elbow_out = (out_sign * (p[side + "elbow"][0] - s_[0]) / torso
                     if side + "elbow" in p else 0.0)
        hip_dy = abs(w[1] - p[side + "hip"][1]) / torso
        arm[side] = {
            "up": ramp(-dy, 0.35, 0.8),                 # hand clearly above the shoulder
            "out": min(ramp(out, 0.5, 0.9), 1 - ramp(abs(dy), 0.4, 0.8)),
            "hip": min(1 - ramp(hip_dy, 0.2, 0.45), ramp(elbow_out, 0.25, 0.5)),
        }
    L, R = arm["L"], arm["R"]
    cands = {
        "Arms up": min(L["up"], R["up"]),
        "Arms out": min(L["out"], R["out"]),
        "Hips": min(L["hip"], R["hip"]),
        "Left up": L["up"] * (1 - R["up"]) * 0.9,
        "Right up": R["up"] * (1 - L["up"]) * 0.9,
        "Left out": L["out"] * (1 - R["out"]) * 0.9,
        "Right out": R["out"] * (1 - L["out"]) * 0.9,
    }
    wd = np.linalg.norm(p["Lwrist"] - p["Rwrist"]) / torso
    mid_y = (((p["Lwrist"] + p["Rwrist"]) / 2)[1] - sh[1]) / torso
    # Hands brought together in front of the body: easy to confuse with half a dozen
    # other gestures, so it only wins when nothing clearer is going on
    cands["Clap"] = min(1 - ramp(wd, 0.2, 0.45), ramp(mid_y, -0.1, 0.2),
                        1 - ramp(mid_y, 0.7, 1.0)) * 0.6
    lean = (sh[0] - hp[0]) / torso  # < 0: leaning to the screen-left
    cands["Lean left" if lean < 0 else "Lean right"] = ramp(abs(lean), 0.15, 0.35) * 0.85
    if angles and "Lknee" in angles and "Rknee" in angles:
        cands["Squat"] = ramp(150 - (angles["Lknee"] + angles["Rknee"]) / 2, 15, 45) * 0.8
    if return_all:
        return cands
    label = max(cands, key=cands.get)
    return (label, cands[label]) if cands[label] > min_clarity else (None, 0.0)


def pick_key_moves(tl, first=2.0, gap=(1.9, 3.1), max_gap=4.5):
    """Key moves for the bar: clear, distinct poses on the beat, every 2-3 s.

    Each move is {"t", "label", "kind", "pts"}; pts is the pose for the pictogram.
    Candidates are beats where the dancer is tracked throughout. If the shoulders
    close up to a line near the beat the dancer is turning, and it's a "Turn" (arm
    positions are meaningless mid-turn, which is what used to turn these into "Left
    up" or "Clap"). Otherwise the pose has to read as something nameable; we take the
    clearest frame within 0.1 s of the beat, and prefer big, held poses over arms
    tucked in or mid-swing.
    """
    times = tl.beats if len(tl.beats) else np.arange(0.0, tl.duration, 0.1)
    tracked_t = np.flatnonzero(tl.tracked)
    if not len(tracked_t):
        return []
    span = int(WINDOW * tl.fps)
    near = max(1, int(0.1 * tl.fps))
    step = max(1, int(0.03 * tl.fps))
    speed = np.nan_to_num(np.abs(np.diff(tl.angles, axis=0)), nan=0.0).mean(axis=1) * tl.fps

    cands = []
    for b in times:
        i = tl.index(b)
        lo, hi = max(i - span, 0), i + span + 1
        if b > tl.duration - 0.5 or not tl.tracked[lo:hi].all():
            continue
        win = tl.shoulder_ratio[lo:hi]
        if np.isfinite(win).any() and np.nanmin(win) < TURN_RATIO:
            j = lo + int(np.nanargmin(win))
            cands.append((float(b), "Turn", 0.9, j))
            continue
        best = None
        for j in range(max(i - near, 0), i + near + 1, step):
            label, clarity = classify_pose(tl.points_at(j, tl.aspect, 1.0),
                                           tl.angles_at(j), min_clarity=0.2)
            if label and (best is None or clarity > best[1]):
                best = (label, clarity, j)
        if best is None:
            continue
        label, clarity, j = best
        extent = pose_extent(tl.points_at(j, tl.aspect, 1.0))
        motion = speed[max(i - span // 2, 0):i + span // 2 + 1].mean()
        score = clarity + 0.3 * min(extent / 1.5, 1.0) - 0.5 * min(motion / 300.0, 1.0)
        cands.append((float(b), label, score, j))

    moves, last = [], tracked_t[0] / tl.fps + first - gap[1]
    while True:
        window = gap[1]
        pool = []
        while not pool and window <= max_gap:
            pool = [c for c in cands if last + gap[0] <= c[0] <= last + window]
            window += 0.5
        if not pool:
            later = [c for c in cands if c[0] > last + max_gap]
            if not later:
                break
            last = later[0][0] - gap[1]  # nothing nameable for a while: skip ahead
            continue
        prev = moves[-1]["label"] if moves else None
        t, label, score, j = max(pool, key=lambda c: c[2] + (0.3 if c[1] != prev else 0.0))
        moves.append({"t": t, "label": label, "kind": "turn" if label == "Turn" else "pose",
                      "pts": tl.points_at(j, tl.aspect, 1.0)})
        last = t
    return moves


def load_moves(tl, folder):
    """The song's key moves: moves.json if the folder has one (so a person can fix
    or add moves by hand), else picked automatically. moves.json is a list of
    {"t": seconds, "label": "Arms up"}; a label of "Turn" makes it a turn."""
    path = os.path.join(folder, "moves.json")
    if not os.path.exists(path):
        return pick_key_moves(tl)
    with open(path, encoding="utf-8") as f:
        raw = sorted(json.load(f), key=lambda e: e["t"])
    moves = []
    for e in raw:
        t, label = float(e["t"]), str(e["label"])
        i = tl.index(t)
        turn = label.lower() == "turn"
        if turn:  # the pictogram is the narrowest moment around it
            lo, hi = tl.index(t - WINDOW), tl.index(t + WINDOW) + 1
            win = tl.shoulder_ratio[lo:hi]
            if np.isfinite(win).any():
                i = lo + int(np.nanargmin(win))
        moves.append({"t": t, "label": label, "kind": "turn" if turn else "pose",
                      "pts": tl.points_at(i, tl.aspect, 1.0)})
    return moves


def export_moves(folder):
    """Write the automatically picked moves to moves.json to edit by hand."""
    path = os.path.join(folder, "moves.json")
    if os.path.exists(path):
        raise SystemExit(f"{path} already exists; delete it first to export fresh moves.")
    moves = pick_key_moves(Timeline(folder))
    with open(path, "w", encoding="utf-8") as f:
        json.dump([{"t": round(m["t"], 2), "label": m["label"]} for m in moves], f, indent=1)
    print(f"Wrote {len(moves)} moves to {path}. Edit labels or times, delete lines you "
          f"don't like, and the game will use that list.")


PHOTO_DIR = "photos"
RESULT_TIMEOUT = 20.0   # seconds on the result screen before going back to idle
NAME_MAX = 12
FX_TIME = 0.4           # how long the hit effect lasts
# Grade = 70% average accuracy + 30% how much of the song the best streak covered
GRADES = [("S", 0.60, (0, 215, 255)), ("A", 0.46, (90, 230, 90)),
          ("B", 0.36, (255, 190, 70)), ("C", 0.00, (110, 110, 255))]


def grade_for(avg_match, best_combo, judged):
    streak = best_combo / judged if judged else 0.0
    value = 0.7 * avg_match + 0.3 * streak
    return next((g, c) for g, th, c in GRADES if value >= th)


TURN_BEFORE, TURN_AFTER = 0.5, 0.4   # a turn is rated on your narrowest shoulders in this window
LEADERBOARD_SIZE = 20


class DanceGame:
    """Score, combo and ratings. Key moves are rated when they land; between
    moves, roughly matching the dancer earns small bonus points."""

    def __init__(self, moves=()):
        self.move_times = np.array([m["t"] for m in moves], float)
        self.move_turn = [m.get("kind") == "turn" for m in moves]
        self.reset()

    def reset(self):
        self.score = self.bonus = self.combo = self.best_combo = self.gained = 0
        self.counts = {r[0]: 0 for r in RATINGS}
        self.matches = []  # match 0..1 of every move that has landed
        self.samples = deque()  # (song time, match or None, shoulder ratio or None)
        self.move_i = 0
        self.last_move = None  # (time, is_turn) of the move that landed last
        self.last_t, self.bonus_clock, self.bonus_t = None, 0.0, -1e9
        self.rating, self.rating_t = None, -1e9
        self.done = False

    @property
    def accuracy(self):
        return float(np.mean(self.matches)) if self.matches else 0.0

    def grade(self):
        return grade_for(self.accuracy, self.best_combo, len(self.matches))

    def update(self, t, live, ratio=None):
        """live = match 0..1 this frame, or None if it isn't scored (dancer lost,
        or your arms aren't visible). ratio = your shoulder width / torso length
        (for turns), None if you aren't in view. Returns True if a move landed."""
        self.samples.append((t, live, ratio))
        dt = 0.0 if self.last_t is None else min(t - self.last_t, 0.1)
        self.last_t = t

        landed = False
        while self.move_i < len(self.move_times):
            m, turn = self.move_times[self.move_i], self.move_turn[self.move_i]
            if t < m + (TURN_AFTER if turn else MOVE_AFTER):
                break
            self.move_i += 1
            if turn:  # rated on how narrow your shoulders got: turned side-on = 0.3 or less
                rs = [r for tt, _, r in self.samples
                      if m - TURN_BEFORE <= tt <= m + TURN_AFTER and r is not None]
                value = float(np.clip((0.6 - min(rs)) / 0.3, 0.0, 1.0)) if rs else 0.0
            else:
                # Unscored frames count as 0 here: hiding from the camera can't dodge a move
                near = [v or 0.0 for tt, v, _ in self.samples
                        if m - MOVE_BEFORE <= tt <= m + MOVE_AFTER]
                smooth = (np.convolve(near, np.ones(5) / 5, "valid") if len(near) >= 5
                          else [np.mean(near)] if near else [0.0])
                value = float(np.max(smooth))
            self._judge(value, t)
            self.last_move = (float(m), turn)
            landed = True
        while self.samples and self.samples[0][0] < t - 1.5:
            self.samples.popleft()

        # Bonus between moves: keep matching the dancer, collect points as you go
        k = np.searchsorted(self.move_times, t)
        gap = min((abs(self.move_times[j] - t) for j in (k - 1, k)
                   if 0 <= j < len(self.move_times)), default=1e9)
        if live is not None and live >= BONUS_MATCH and gap > BONUS_CLEAR:
            self.bonus_clock += dt
            if self.bonus_clock >= BONUS_EVERY:
                self.bonus_clock -= BONUS_EVERY
                self.score += BONUS_POINTS
                self.bonus += BONUS_POINTS
                self.bonus_t = t
        else:
            self.bonus_clock = 0.0
        return landed

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


# --------------------------------------------------------------------------
# Leaderboard: one leaderboard.json per song folder
# --------------------------------------------------------------------------
def load_board(folder):
    try:
        with open(os.path.join(folder, "leaderboard.json"), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return []


def add_to_board(folder, entry):
    """Add a score; returns its rank (1 = top) or None if it didn't make the board."""
    board = load_board(folder)
    board.append(entry)
    board.sort(key=lambda e: (-e["score"], e["date"]))
    rank = board.index(entry) + 1
    try:
        with open(os.path.join(folder, "leaderboard.json"), "w", encoding="utf-8") as f:
            json.dump(board[:LEADERBOARD_SIZE], f, indent=1)
    except OSError:
        pass
    return rank if rank <= LEADERBOARD_SIZE else None


def put_right(img, text, x, y, scale=0.6, color=(255, 255, 255), thick=1):
    """put() with the text ending at x instead of starting there."""
    (tw, _), _ = cv2.getTextSize(text, FONT, scale, thick)
    put(img, text, (x - tw, y), scale, color, thick)


def draw_board(img, x, y, board, mark=None, rows=5, width=420, scale=0.6):
    """Top `rows` scores at (x, y); `mark` = 0-based rank to highlight (it is added
    as an extra line if it fell below the visible rows)."""
    gold = (0, 215, 255)
    put(img, "LEADERBOARD", (x, y), 0.7, gold, 2)
    if not board:
        put(img, "no scores yet", (x, y + 40), scale, (170, 170, 170), 1)
        return
    shown = list(range(min(rows, len(board))))
    if mark is not None and mark >= rows:
        shown.append(mark)
    for r, k in enumerate(shown):
        e, yy = board[k], y + 38 + r * int(scale * 56)
        hl = k == mark
        if hl:
            cv2.rectangle(img, (x - 10, yy - int(scale * 36)), (x + width, yy + int(scale * 12)),
                          (80, 55, 40), -1)
        color = gold if hl else (235, 235, 235)
        put(img, f"{k + 1}.", (x, yy), scale, color, 1)
        put(img, str(e["name"])[:NAME_MAX].upper(), (x + 50, yy), scale, color, 2 if hl else 1)
        put_right(img, str(e["score"]), x + width - 50, yy, scale, color, 1)
        put_right(img, str(e.get("grade", "")), x + width - 8, yy, scale, color, 2)


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


# Clean, iconic pictograms for the standard move labels, like the game's own: a
# mannequin with the arms set to the pose, instead of the raw (noisy) keypoints.
# Units are torso lengths from the shoulder line, x right, y down; arm entries are
# (elbow, wrist) for the screen-LEFT arm and get mirrored for the right one.
ICON_ARMS = {
    "down": ((-0.10, 0.24), (-0.14, 0.48)),
    "up": ((-0.17, -0.26), (-0.12, -0.52)),
    "out": ((-0.25, 0.0), (-0.50, 0.0)),
    "hip": ((-0.30, 0.20), (-0.11, 0.48)),
    "clap": ((-0.20, 0.22), (-0.03, 0.14)),
}
# label -> (left arm, right arm, lean degrees (+ = to the screen-right), legs, narrow)
ICONS = {
    "Arms up": ("up", "up"), "Arms out": ("out", "out"), "Hips": ("hip", "hip"),
    "Clap": ("clap", "clap"),
    "Left up": ("up", "down"), "Right up": ("down", "up"),
    "Left out": ("out", "down"), "Right out": ("down", "out"),
    "Lean left": ("down", "down", -26), "Lean right": ("down", "down", 26),
    "Squat": ("down", "down", 0, True),
    "Turn": ("clap", "clap", 0, False, True),
}


def draw_icon(img, label, box, arm_color=(0, 215, 255), body_color=(240, 240, 240)):
    """Draw the standard pictogram for `label` into box=(x0,y0,x1,y1).
    Returns False if the label has no standard icon (custom moves)."""
    spec = ICONS.get(label)
    if spec is None:
        return False
    left, right = spec[0], spec[1]
    lean = np.radians(spec[2]) if len(spec) > 2 else 0.0
    legs = len(spec) > 3 and spec[3]
    narrow = len(spec) > 4 and spec[4]

    sw = 0.05 if narrow else 0.17             # half shoulder width (side-on when turning)
    shoulders = {"L": np.array([-sw, 0.0]), "R": np.array([sw, 0.0])}
    hips = {"L": np.array([-0.09 if not narrow else -0.03, 0.5]),
            "R": np.array([0.09 if not narrow else 0.03, 0.5])}
    pts = {"headc": np.array([0.0, -0.22]), "sL": shoulders["L"], "sR": shoulders["R"]}
    for side, name in (("L", left), ("R", right)):
        elbow, wrist = (np.array(a) for a in ICON_ARMS[name])
        if side == "R":  # the arm table is for the left arm: mirror it
            elbow, wrist = elbow * [-1, 1], wrist * [-1, 1]
        if narrow:  # side-on: arms swing forward, close to the body
            elbow, wrist = elbow * [0.3, 1], wrist * [0.3, 1]
        pts["e" + side], pts["w" + side] = shoulders[side] + elbow, shoulders[side] + wrist

    # Leaning tips everything above the hips around the hip centre
    pivot = np.array([0.0, 0.5])
    c, s = np.cos(lean), np.sin(lean)
    rot = lambda p: pivot + np.array([c * (p[0] - pivot[0]) - s * (p[1] - pivot[1]),
                                      s * (p[0] - pivot[0]) + c * (p[1] - pivot[1])])
    for k in list(pts):
        pts[k] = rot(pts[k])

    if legs:   # squat: knees out and forward, shorter stance
        knees = {"L": np.array([-0.30, 0.82]), "R": np.array([0.30, 0.82])}
        ankles = {"L": np.array([-0.17, 1.15]), "R": np.array([0.17, 1.15])}
        bottom = 1.2
    else:      # standing: thighs only, so the arms get the room
        knees = {"L": hips["L"] + [-0.01, 0.38], "R": hips["R"] + [0.01, 0.38]}
        ankles = None
        bottom = 0.95
    x0, y0, x1, y1 = box
    # One scale for every icon (so they look alike in size): sized so hands-up and
    # arms-out both fit
    top = -0.68
    scale = min((x1 - x0) / 1.25, (y1 - y0) / (bottom - top))
    mid = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
    centre = np.array([0.0, (top + bottom) / 2])
    P = lambda p: tuple(int(v) for v in (np.asarray(p) - centre) * scale + mid)
    thick = max(3, int(scale * 0.075))

    line = lambda a, b, col, t=thick: cv2.line(img, P(a), P(b), col, t, cv2.LINE_AA)
    for side in "LR":
        line(hips[side], knees[side], body_color)
        if ankles:
            line(knees[side], ankles[side], body_color)
    line(rot(shoulders["L"]), rot(shoulders["R"]), body_color)
    line(rot(shoulders["L"]), hips["L"], body_color)
    line(rot(shoulders["R"]), hips["R"], body_color)
    line(hips["L"], hips["R"], body_color)
    cv2.circle(img, P(pts["headc"]), int(scale * 0.11), body_color, max(2, thick - 1), cv2.LINE_AA)
    for side in "LR":
        sh = pts["s" + side]
        line(sh, pts["e" + side], arm_color, thick + 1)
        line(pts["e" + side], pts["w" + side], arm_color, thick + 1)
        cv2.circle(img, P(pts["w" + side]), max(3, int(thick * 0.9)), arm_color, -1, cv2.LINE_AA)
    return True


def draw_pictogram(img, pts, box, legs=False, arm_color=(0, 215, 255),
                   body_color=(235, 235, 235)):
    """Clean stick-figure pictogram of a pose, fitted into box=(x0,y0,x1,y1).
    Arms are drawn in colour so the move reads at a glance. Only the upper
    body is shown (so it fills the box) unless legs=True."""
    pts = {k: v for k, v in pts.items() if legs or not ("knee" in k or "ankle" in k)}
    if len(pts) < 5:
        return
    x0, y0, x1, y1 = box
    sw = (np.linalg.norm(pts["Lshoulder"] - pts["Rshoulder"])
          if "Lshoulder" in pts and "Rshoulder" in pts else 0.1)
    head_r = 0.32 * sw
    arr = np.array(list(pts.values()))
    mn, mx = arr.min(0), arr.max(0)
    mn[1] -= head_r * 1.2  # room for the head circle
    span = np.maximum(mx - mn, 1e-3)
    scale = min((x1 - x0) / span[0], (y1 - y0) / span[1])
    P = {k: (v - (mn + mx) / 2) * scale + np.array([(x0 + x1) / 2, (y0 + y1) / 2])
         for k, v in pts.items()}
    for a, b in BONES:
        if a in P and b in P:
            arm = "elbow" in a + b or "wrist" in a + b
            cv2.line(img, ipt(P[a]), ipt(P[b]), arm_color if arm else body_color,
                     4 if arm else 3, cv2.LINE_AA)
    if "head" in P:
        cv2.circle(img, ipt(P["head"]), max(4, int(head_r * scale)), body_color, 3,
                   cv2.LINE_AA)


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


def angle_range(history, keys=None):
    """Mean over joint angles of (max - min) across a list of {name: degrees} dicts."""
    spans = []
    for k in (keys or {k for h in history for k in h}):
        vals = [h[k] for h in history if k in h]
        if len(vals) >= 3:
            spans.append(max(vals) - min(vals))
    return float(np.mean(spans)) if spans else None


def still_factor(player_hist, tl, t):
    """1.0 if you're moving about as much as the dancer was over the last second,
    down to STILL_FLOOR if the dancer was moving and you were frozen."""
    lo, hi = tl.index(t - STILL_WINDOW), tl.index(t)
    if hi - lo < 3 or len(player_hist) < 5:
        return 1.0
    seg = tl.angles[lo:hi + 1]
    seg = seg[:, np.isfinite(seg).sum(axis=0) >= 3]  # angles the dancer actually has here
    dancer = float((np.nanmax(seg, axis=0) - np.nanmin(seg, axis=0)).mean()) if seg.size else 0.0
    mine = angle_range(player_hist)
    if dancer < STILL_DANCER_MIN or mine is None:
        return 1.0
    return float(np.clip(STILL_FLOOR + mine / dancer, STILL_FLOOR, 1.0))


def shoulder_ratio_of(p, vis):
    """Shoulder width / torso length of a player's points, None if not all visible."""
    keys = ("Lshoulder", "Rshoulder", "Lhip", "Rhip")
    if any(vis[k] < 0.5 for k in keys):
        return None
    torso = np.linalg.norm((p["Lshoulder"] + p["Rshoulder"]) / 2 - (p["Lhip"] + p["Rhip"]) / 2)
    return float(np.linalg.norm(p["Lshoulder"] - p["Rshoulder"]) / torso) if torso > 1e-6 else None


class DanceSession:
    """Everything for one song except the webcam, window and audio device."""

    def __init__(self, folder, title="Dance"):
        self.folder, self.title, self.multi_song = folder, title, False
        self.tl = Timeline(folder)
        self.video = VideoPanel(os.path.join(folder, "video_720p.mp4"))
        self.game = DanceGame(self.tl.moves)
        self.rng = np.random.default_rng()
        self.split = 0
        self.start_run("PLAYER")

    def start_run(self, player):
        """Fresh score, effects and best-move photo for this player."""
        self.player = player
        self.game.reset()
        self.pts = self.world = None
        self.in_frame = False
        self.fx, self.pops = [], []
        self.buf = deque(maxlen=45)     # recent frames: (t, match, shoulder ratio, reach, frame)
        self.best_peak, self.best_value, self.best_frame = -1.0, 0.0, None
        self.bonus_seen = -1e9
        self.angle_hist = deque()       # (t, your joint angles) over the last STILL_WINDOW
        self.photo = self.photo_path = self.grade = self.rank = None
        self.board = []
        self.result_t0 = 0.0

    def finish(self):
        """Song over: grade it, save the score, stamp and save the best-move photo."""
        g = self.game
        g.done = True
        self.grade = g.grade()
        self.rank = add_to_board(self.folder, {
            "name": self.player, "score": g.score, "grade": self.grade[0],
            "accuracy": round(g.accuracy, 3), "combo": g.best_combo,
            "date": time.strftime("%Y-%m-%d %H:%M:%S")})
        self.board = load_board(self.folder)
        if self.best_frame is not None:
            self.photo = photo_card(self.best_frame, self.player, g.score,
                                    self.grade, self.best_value)
            self.photo_path = save_photo(self.photo, self.player)
            print(f"Photo: {self.photo_path or 'could not be saved'}")
        self.result_t0 = time.monotonic()

    def render(self, cam, result, t, playing):
        """cam: mirrored webcam frame, result: its PoseLandmarker result,
        t: song time from the audio clock. Returns the full game frame: the
        dancer and you side by side, and the move bar underneath.

        You are scored against the dancer's pose right now, best match within
        +-WINDOW of t.
        """
        tl, h = self.tl, PANEL_H
        i = tl.index(t)
        tracked = bool(tl.tracked[i])
        raw = cam

        # Left: the dancer, flipped, with the extracted skeleton
        vid = cv2.flip(self.video.frame_at(t), 1)
        vw = int(h * vid.shape[1] / vid.shape[0])
        vid = cv2.resize(vid, (vw, h))
        dancer = tl.points_at(i, vw, h)
        line = (255, 255, 255) if tracked else (120, 120, 120)
        draw_skeleton(vid, dancer, lambda a, b: (0, 0, 0), 6)
        draw_skeleton(vid, dancer, lambda a, b: line, 3)

        # Right: you, each limb coloured by how far it is from the dancer's
        cw = int(h * cam.shape[1] / cam.shape[0])
        cam = cv2.resize(cam, (cw, h))
        player_angles, vis, has_player = {}, {}, bool(result.pose_landmarks)
        ratio = reach = None
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
            ratio, reach = shoulder_ratio_of(self.pts, vis), pose_extent(self.pts)
        else:
            self.pts = self.world = None
            self.in_frame = False

        # Not in frame = 0, but an arm the camera can't see right now (mid-turn)
        # just isn't scored rather than counting as a miss
        arms_seen = all(k in player_angles for k in REQUIRED)
        live, best_j = ((None, None) if has_player and not arms_seen
                        else dance_score(player_angles, tl, t))
        if playing and player_angles:
            self.angle_hist.append((t, player_angles))
        while self.angle_hist and self.angle_hist[0][0] < t - STILL_WINDOW:
            self.angle_hist.popleft()
        if live is not None and playing:  # standing still doesn't score
            live *= still_factor([a for _, a in self.angle_hist], tl, t)

        if has_player:
            visible = {k: p for k, p in self.pts.items() if vis[k] > 0.5}
            errs = {}
            if best_j is not None:  # error against the frame that scored best
                target = tl.angles_at(best_j)
                errs = {k: abs(v - target[k]) for k, v in player_angles.items()
                        if k in target}
            draw_skeleton(cam, visible, limb_color(errs), 5, head=False)
            for p in visible.values():
                cv2.circle(cam, ipt(p), 5, (255, 255, 255), -1, cv2.LINE_AA)

        if playing:
            self.buf.append((t, live, ratio, reach or 0.0, raw))
            if self.game.update(t, live, ratio):
                self._add_fx(t, (cw, h))
                self._pick_photo()
            if self.game.bonus_t != self.bonus_seen:  # a bonus tick just landed
                self.bonus_seen = self.game.bonus_t
                where = self.pts["head"] if self.pts else np.array([cw / 2, h / 2])
                self.pops.append({"t": t, "pos": where})

        self.fx = [f for f in self.fx if draw_fx(cam, f, t)]
        self.pops = [q for q in self.pops if t - q["t"] < 0.7]
        for q in self.pops:  # "+10" floats up off your head
            put(cam, f"+{BONUS_POINTS}", (int(q["pos"][0]) + 60, int(q["pos"][1] - 20 - 70 * (t - q["t"]))),
                0.9, (90, 230, 90), 2, center=True)

        self.split = vw
        panels = np.hstack([vid, cam])
        self.draw_hud(panels, vw, t, live, tracked, has_player)
        canvas = np.vstack([panels, self.draw_bar(panels.shape[1], t)])
        if self.game.done:
            self.draw_result(canvas)
        return canvas

    def _pick_photo(self):
        """A move just landed: keep a frame from it if it's the best move so far.

        The photo comes from the moves themselves, not the whole song, because the
        best-matching frames overall are usually you and the dancer both standing
        still. Within the move, prefer the frame where you're best matched and your
        arms are out (or, for a turn, your shoulders are narrowest)."""
        g = self.game
        m, turn = g.last_move
        lo, hi = ((m - TURN_BEFORE, m + TURN_AFTER) if turn
                  else (m - MOVE_BEFORE, m + MOVE_AFTER))
        frames = [e for e in self.buf if lo <= e[0] <= hi and e[2] is not None]
        if not frames:
            return
        reach_bonus = lambda e: 0.15 * min(e[3] / 1.5, 1.0)
        pick = (min(frames, key=lambda e: e[2]) if turn
                else max(frames, key=lambda e: (e[1] or 0.0) + reach_bonus(e)))
        value = g.matches[-1]
        score = value + (0.0 if turn else reach_bonus(pick))
        if score > self.best_peak:
            self.best_peak, self.best_value, self.best_frame = score, value, pick[4].copy()

    def draw_bar(self, w, t):
        """Bottom bar: upcoming key moves scroll left toward the hit zone."""
        tl, g = self.tl, self.game
        times = g.move_times
        bar = np.linspace((34, 26, 26), (16, 12, 12), BAR_H).astype(np.uint8)[:, None, :]
        bar = np.repeat(bar, w, axis=1)
        cy, r0 = 86, 54
        px = (w - HIT_X - 80) / BAR_SECONDS
        gold = (0, 215, 255)

        nxt = int(np.searchsorted(times, t, side="right"))  # first move still to come
        dt_next = times[nxt] - t if nxt < len(times) else 1e9
        pulse = max(0.0, 1 - dt_next / 0.4)

        # Hit zone, lit up as the next move arrives and flashing the rating on landing
        cv2.line(bar, (HIT_X, 0), (HIT_X, BAR_H - 14), (90, 80, 80), 1, cv2.LINE_AA)
        age = t - g.rating_t
        if g.rating and age < MOVE_FEEDBACK:
            k = 1 - age / MOVE_FEEDBACK
            color = next(c for n, *_, c in RATINGS if n == g.rating)
            layer = bar.copy()
            cv2.circle(layer, (HIT_X, cy), r0 + 14, color, -1, cv2.LINE_AA)
            cv2.addWeighted(layer, 0.55 * k, bar, 1 - 0.55 * k, 0, bar)
            pop = 1 + 0.4 * max(0.0, 1 - age / 0.15)
            put(bar, g.rating, (HIT_X + 190, cy), 1.4 * pop, color, 3, center=True)
        cv2.circle(bar, (HIT_X, cy), r0 + 14, (255, 255, 255) if pulse < 0.5 else gold,
                   3 + int(3 * pulse), cv2.LINE_AA)

        lo = int(np.searchsorted(times, t - MOVE_FADE))
        hi = int(np.searchsorted(times, t + BAR_SECONDS))
        for idx in range(lo, hi):
            dt = times[idx] - t
            x = int(HIT_X + dt * px)
            alpha = 1.0 if dt >= 0 else max(0.0, 1 + dt / MOVE_FADE)
            target = bar if alpha >= 1.0 else bar.copy()
            self._draw_move(target, tl.moves[idx], x, cy, r0, dt, idx == nxt)
            if alpha < 1.0:
                cv2.addWeighted(target, alpha, bar, 1 - alpha, 0, bar)

        cv2.rectangle(bar, (10, BAR_H - 10), (w - 10, BAR_H - 4), (60, 55, 55), -1)
        cv2.rectangle(bar, (10, BAR_H - 10),
                      (10 + int((w - 20) * min(t / tl.duration, 1.0)), BAR_H - 4), gold, -1)
        return bar

    def _draw_move(self, bar, mv, x, cy, r0, dt, is_next):
        gold = (0, 215, 255)
        cv2.circle(bar, (x, cy), r0, (52, 40, 40), -1, cv2.LINE_AA)
        cv2.circle(bar, (x, cy), r0, (190, 190, 190), 2, cv2.LINE_AA)
        box = (x - int(r0 * 0.74), cy - int(r0 * 0.82), x + int(r0 * 0.74), cy + int(r0 * 0.82))
        if not draw_icon(bar, mv["label"], box):  # a custom label: show the real pose
            draw_pictogram(bar, mv["pts"], box, legs=False)
        if mv.get("kind") == "turn":  # a circular arrow around the figure
            cv2.ellipse(bar, (x, cy), (r0 - 9, r0 - 9), 0, 40, 330, gold, 3, cv2.LINE_AA)
            a = np.radians(40)
            tip = np.array([x + (r0 - 9) * np.cos(a), cy + (r0 - 9) * np.sin(a)])
            for da in (0.55, -0.35):
                end = tip + 12 * np.array([np.cos(a + np.pi / 2 + da), np.sin(a + np.pi / 2 + da)])
                cv2.line(bar, ipt(tip), ipt(end), gold, 3, cv2.LINE_AA)
        if 0 < dt <= APPROACH:  # the ring closes in on the move as it arrives
            k = dt / APPROACH
            col = tuple(int(a + (b - a) * (1 - k)) for a, b in zip((130, 130, 130), gold))
            cv2.circle(bar, (x, cy), int(r0 * (1 + 0.6 * k)), col, 2 + int(2 * (1 - k)),
                       cv2.LINE_AA)
        put(bar, mv["label"], (x, cy + r0 + 19), 0.65, (255, 255, 255), 1, center=True)

        if is_next and 0 < dt <= APPROACH:  # 3-2-1, flashing as each second begins
            n = int(np.ceil(dt))
            flash = max(0.0, 1 - (n - dt) / 0.45)
            # The number sits in a badge on the corner so the icon stays readable;
            # the disc also glows gold for a moment each second
            if flash > 0:
                layer = bar.copy()
                cv2.circle(layer, (x, cy), r0, gold, -1, cv2.LINE_AA)
                cv2.addWeighted(layer, 0.4 * flash, bar, 1 - 0.4 * flash, 0, bar)
            bx, by, br = x + int(r0 * 0.8), cy - int(r0 * 0.8), 20 + int(10 * flash)
            cv2.circle(bar, (bx, by), br, gold, -1, cv2.LINE_AA)
            cv2.circle(bar, (bx, by), br, (255, 255, 255), 2, cv2.LINE_AA)
            put(bar, str(n), (bx, by), 1.0 + 0.5 * flash, (30, 20, 20), 3, center=True)

    def draw_outline(self, img, ok):
        """'Get in frame' guide on the player panel: green once you fit it."""
        h, w = img.shape[:2]
        h = PANEL_H
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

    def draw_hud(self, img, split, t, live, tracked, has_player):
        g, h, w = self.game, img.shape[0], img.shape[1]
        cx = split + (w - split) // 2  # centre of the player panel

        cv2.line(img, (split, 0), (split, h), (0, 0, 0), 3)
        put(img, "DANCER", (15, 35), 0.8, (0, 215, 255))
        if not tracked:
            put(img, "dancer lost - not scored", (15, h - 40), 0.7, (90, 90, 255), 1)

        shade(img, split + 10, 10, split + 330, 140)
        put(img, self.player.upper(), (split + 25, 30), 0.6, (200, 200, 200), 1)
        put(img, f"SCORE {g.score}", (split + 25, 62), 1.0)
        put(img, f"COMBO x{g.combo}", (split + 25, 92), 0.7, (0, 215, 255))
        put_right(img, f"BONUS {g.bonus}", split + 315, 92, 0.6, (90, 230, 90), 1)
        m = live or 0.0
        bar(img, split + 25, 106, 290, 22, m,
            score_color(m) if live is not None else (90, 90, 90))
        tick = split + 25 + int(290 * BONUS_MATCH)  # stay right of this for bonus points
        cv2.line(img, (tick, 104), (tick, 130), (255, 255, 255), 2)
        put(img, "--" if live is None else f"{int(m * 100)}%", (split + 70, 124), 0.6,
            (255, 255, 255), 1)

        age = t - g.rating_t
        if g.rating and age < MOVE_FEEDBACK:
            color = next(c for n, *_, c in RATINGS if n == g.rating)
            rx = (split + 340 + w) // 2  # top of the panel, clear of the score box
            pop = 1 + 0.35 * max(0.0, 1 - age / 0.2)  # lands big, settles
            put(img, g.rating, (rx, 55), 1.8 * pop, color, 4, center=True)
            if g.gained:
                put(img, f"+{g.gained}", (rx, 110), 1.0, color, 2, center=True)
        if not has_player:
            put(img, "Step back so I can see you!", (cx, h - 70), 0.9, (90, 90, 255),
                center=True)

    def draw_result(self, img):
        g, w = self.game, img.shape[1]
        h = PANEL_H  # lay out within the two panels; the bar below just goes dark
        letter, gcolor = self.grade
        gold = (0, 215, 255)
        shade(img, 0, 0, w, img.shape[0], 0.88)
        put(img, "SHOW'S OVER!", (w // 2, 40), 1.5, gold, 3, center=True)
        if self.rank == 1:
            put(img, "NEW HIGH SCORE!", (w // 2, 80), 0.9, (90, 230, 90), 2, center=True)

        # Left: the best-move photo
        x0, y0, ph = 40, 105, 270
        if self.photo is not None:
            pw = int(ph * self.photo.shape[1] / self.photo.shape[0])
            img[y0:y0 + ph, x0:x0 + pw] = cv2.resize(self.photo, (pw, ph))
            cv2.rectangle(img, (x0 - 3, y0 - 3), (x0 + pw + 3, y0 + ph + 3),
                          (255, 255, 255), 3)
            caption = f"saved to {self.photo_path}" if self.photo_path else "best move"
            put(img, caption, (x0 + pw // 2, y0 + ph + 28), 0.55, (200, 200, 200), 1,
                center=True)
        else:
            pw = int(ph * 16 / 9)
            put(img, "No photo - I never saw you!", (x0 + pw // 2, h // 2), 0.9,
                (150, 150, 150), center=True)

        # Middle: grade and stats
        mx = x0 + pw + 60
        put(img, letter, (mx + 75, 235), 6.0, gcolor, 12, center=True)
        sx = mx + 185
        put(img, self.title, (sx, 120), 0.55, (200, 200, 200), 1)
        put(img, self.player.upper(), (sx, 155), 1.0, (255, 255, 255))
        put(img, f"Score  {g.score}", (sx, 205), 1.2, gold)
        put(img, f"Move accuracy  {int(g.accuracy * 100)}%", (sx, 245), 0.75)
        put(img, f"Best combo  x{g.best_combo}", (sx, 280), 0.75)
        put(img, f"Bonus  {g.bonus}", (sx, 315), 0.75, (90, 230, 90))
        put(img, "  ".join(f"{n} {g.counts[n]}" for n, *_ in RATINGS),
            (sx, 355), 0.5, (200, 200, 200), 1)

        # Right: this song's leaderboard, with this run highlighted
        bx = max(w - 440, sx + 330)
        draw_board(img, bx, 120, self.board, None if self.rank is None else self.rank - 1,
                   rows=6, width=min(420, w - bx - 30))

        left = max(0, int(np.ceil(RESULT_TIMEOUT - (time.monotonic() - self.result_t0))))
        keys = "R  play again        N  new player        "
        keys += "M  songs        " if self.multi_song else ""
        put(img, keys + "Q  quit", (w // 2, h - 75), 1.0, (255, 255, 255), 2, center=True)
        put(img, f"back to the {'song menu' if self.multi_song else 'start screen'} in {left}s",
            (w // 2, h - 35), 0.7, (160, 160, 160), 1, center=True)

    def render_idle(self, cam, result, name, now, can_go_back):
        """Name screen for the chosen song: type a name, ENTER to dance."""
        h = PANEL_H
        vh, vw_src = self.video.frame.shape[:2]
        vw = int(h * vw_src / vh)

        left = np.linspace((60, 25, 30), (25, 10, 15), h).astype(np.uint8)[:, None, :]
        left = np.repeat(left, vw, axis=1)
        put(left, "POSE PARTY", (vw // 2, 90), 2.0, (0, 215, 255), 4, center=True)
        put(left, self.title, (vw // 2, 150), 0.9, (255, 255, 255), 2, center=True)
        put(left, "Copy the dancer. Hit the moves.", (vw // 2, 190), 0.65,
            (200, 200, 200), 1, center=True)
        put(left, "Your name", (vw // 2, 262), 0.7, (180, 180, 180), 1, center=True)
        cv2.rectangle(left, (vw // 2 - 200, 282), (vw // 2 + 200, 342), (255, 255, 255), 2)
        cursor = "_" if int(now * 2) % 2 == 0 else " "
        put(left, (name + cursor) if name else cursor, (vw // 2, 313), 1.1,
            (255, 255, 255), 2, center=True)
        put(left, "ENTER  start", (vw // 2, 405), 1.0, (90, 230, 90), 2, center=True)
        put(left, "ESC  " + ("choose another song" if can_go_back else "quit"),
            (vw // 2, 450), 0.7, (160, 160, 160), 1, center=True)
        return stack_with_footer(left, webcam_panel(cam, result), "Moves scroll in from "
                                 "the right. Match the dancer when one reaches the circle.")


def webcam_panel(cam, result):
    """Your webcam at panel height, with a skeleton once you're in view."""
    h = PANEL_H
    cw = int(h * cam.shape[1] / cam.shape[0])
    right = cv2.resize(cam, (cw, h))
    seen = bool(result.pose_landmarks)
    if seen:
        pts, vis = landmarks_to_points(result.pose_landmarks[0], cw, h)
        draw_skeleton(right, {k: p for k, p in pts.items() if vis[k] > 0.5},
                      lambda a, b: (90, 230, 90), 4)
    put(right, "I can see you!" if seen else "Step back so I can see you!",
        (cw // 2, h - 40), 0.9, (90, 230, 90) if seen else (90, 90, 255), center=True)
    return right


def stack_with_footer(left, right, text):
    """Two panels plus a footer the height of the move bar, so menu screens keep
    the same overall shape as the game."""
    panels = np.hstack([left, right])
    footer = np.zeros((BAR_H, panels.shape[1], 3), np.uint8)
    footer[:] = (20, 14, 14)
    put(footer, text, (panels.shape[1] // 2, BAR_H // 2), 0.8, (170, 170, 170), 1,
        center=True)
    return np.vstack([panels, footer])


# --------------------------------------------------------------------------
# Song menu
# --------------------------------------------------------------------------
SONG_DIR = "output"  # one sub-folder per song, made by extract_dance.py
MENU_ROWS = 5


def find_songs(root):
    """[{"title", "folder", "seconds"}] for every extracted song under root."""
    songs = []
    for name in sorted(os.listdir(root)) if os.path.isdir(root) else []:
        d = os.path.join(root, name)
        if not all(os.path.exists(os.path.join(d, f))
                   for f in ("timeline.npz", "audio.wav", "video_720p.mp4")):
            continue
        title = name.replace("_", " ")
        try:
            with open(os.path.join(d, "song.json"), encoding="utf-8") as f:
                title = json.load(f).get("title") or title
        except (OSError, ValueError):
            pass
        with wave.open(os.path.join(d, "audio.wav")) as wf:
            seconds = wf.getnframes() / wf.getframerate()
        songs.append({"title": title, "folder": d, "seconds": seconds})
    return songs


def render_menu(songs, sel, board):
    """Song list on the left, the selected song's leaderboard on the right."""
    h, vw = PANEL_H, 960
    left = np.linspace((60, 25, 30), (25, 10, 15), h).astype(np.uint8)[:, None, :]
    left = np.repeat(left, vw, axis=1)
    put(left, "POSE PARTY", (vw // 2, 80), 2.2, (0, 215, 255), 4, center=True)
    put(left, "Choose a song", (vw // 2, 135), 0.8, (220, 220, 220), 1, center=True)

    first = int(np.clip(sel - MENU_ROWS // 2, 0, max(0, len(songs) - MENU_ROWS)))
    for r, idx in enumerate(range(first, min(len(songs), first + MENU_ROWS))):
        y = 200 + r * 58
        song, chosen = songs[idx], idx == sel
        if chosen:
            cv2.rectangle(left, (50, y - 26), (vw - 50, y + 26), (80, 55, 40), -1)
            cv2.rectangle(left, (50, y - 26), (vw - 50, y + 26), (0, 215, 255), 2)
        color = (255, 255, 255) if chosen else (150, 150, 150)
        put(left, song["title"][:34], (75, y + 9), 0.8, color, 2 if chosen else 1)
        mins = f"{int(song['seconds'] // 60)}:{int(song['seconds'] % 60):02d}"
        put(left, mins, (vw - 130, y + 9), 0.7, color, 1)
    if len(songs) > MENU_ROWS:
        put(left, f"{sel + 1} / {len(songs)}", (vw - 110, 135), 0.6, (150, 150, 150), 1,
            center=True)
    put(left, "UP / DOWN  choose        ENTER  play        ESC  quit", (vw // 2, h - 40),
        0.65, (160, 160, 160), 1, center=True)

    right = np.linspace((45, 20, 25), (20, 8, 12), h).astype(np.uint8)[:, None, :]
    right = np.repeat(right, vw, axis=1)
    put(right, songs[sel]["title"][:30], (60, 80), 0.9, (255, 255, 255), 2)
    draw_board(right, 60, 140, board, rows=8, width=vw - 120, scale=0.7)
    return stack_with_footer(left, right, "Add songs with extract_dance.py: each one gets "
                             "its own folder in output/")


UP_KEYS = {2490368, 65362, 63232, ord("w")}      # arrow codes: Windows, Linux, macOS
DOWN_KEYS = {2621440, 65364, 63233, ord("s"), 9}  # TAB too


def dance_main(folder=None, camera=0, songs_dir=SONG_DIR):
    songs = ([{"title": "Dance", "folder": folder, "seconds": 0.0}] if folder
             else find_songs(songs_dir))
    if not songs:
        raise SystemExit(f"No songs found in {songs_dir}/. Make one with extract_dance.py.")
    ensure_model()
    landmarker = make_landmarker()

    cap = cv2.VideoCapture(camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        raise SystemExit("Couldn't open the webcam. Try --camera 1 if you have several.")

    window = "Pose Party - Dance"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 1600, 620)

    menu = not folder                    # a folder on the command line skips the menu
    state = "menu" if menu else "load"   # menu -> load -> idle -> countdown -> play -> done
    sel, name, countdown_t0, last_ts = 0, "", 0.0, 0
    session = audio = None
    loaded = board_for = None
    board = []

    def load_song():
        nonlocal session, audio, loaded
        song = songs[sel]
        if loaded != song["folder"]:
            if audio:
                audio.close()
            session = DanceSession(song["folder"], song["title"])
            audio = AudioClock(os.path.join(song["folder"], "audio.wav"))
            loaded = song["folder"]
        session.multi_song = menu
        audio.restart()

    def begin(player):
        nonlocal state, countdown_t0
        audio.restart()
        session.start_run(player)
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

        if state == "load":  # first frame after choosing: load (a second or two)
            load_song()
            state = "idle"
        if state == "menu":
            if board_for != sel:  # reread the board when you land on a song
                board, board_for = load_board(songs[sel]["folder"]), sel
            canvas = render_menu(songs, sel, board)
        elif state == "idle":
            canvas = session.render_idle(frame, result, name, now, menu)
        else:
            if state == "play" and audio.finished():
                session.finish()
                state = "done"
            t = min(audio.time(), session.tl.duration - 1e-3)
            canvas = session.render(frame, result, t,
                                    playing=state == "play" and not audio.paused)
            ch, cw = PANEL_H, canvas.shape[1]
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
                state, name, board_for = ("menu" if menu else "idle"), "", None
        cv2.imshow(window, canvas)

        raw = cv2.waitKeyEx(1)
        if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
            break
        if raw == -1:
            continue
        key = raw & 0xFF if raw not in UP_KEYS | DOWN_KEYS else raw
        if state == "menu":
            if key in UP_KEYS:
                sel = (sel - 1) % len(songs)
            elif key in DOWN_KEYS:
                sel = (sel + 1) % len(songs)
            elif key in (13, 10):
                state = "load"
            elif key in (27, ord("q")):
                break
            continue
        if state == "idle":  # letters go into the name, so only ESC leaves
            if key == 27:
                if not menu:
                    break
                state, name, board_for = "menu", "", None
            elif key in (13, 10):
                begin(name.strip() or "Player")
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
            elif key == ord("m") and menu:
                audio.restart()
                state, name, board_for = "menu", "", None
        elif key == ord("r"):
            begin(session.player)
        elif key == ord(" "):
            if state == "play":
                audio.paused = not audio.paused
            elif state == "countdown":  # skip the in-frame check
                state, audio.paused = "play", False

    if audio:
        audio.close()
    cap.release()
    landmarker.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Pose Party")
    ap.add_argument("--dance", nargs="?", const="", metavar="FOLDER",
                    help="dance along to a song: with a folder made by extract_dance.py "
                    "play that song, with nothing open the song menu")
    ap.add_argument("--songs", default=SONG_DIR, metavar="DIR",
                    help="where the song menu looks for songs (default: %(default)s)")
    ap.add_argument("--export-moves", metavar="FOLDER",
                    help="write a song's automatically picked key moves to moves.json "
                    "so you can fix labels/times by hand, then exit")
    ap.add_argument("--camera", default="0",
                    help="webcam index (or a video file, for testing)")
    args = ap.parse_args()
    if args.export_moves:
        export_moves(args.export_moves)
    elif args.dance is not None:
        dance_main(args.dance or None,
                   int(args.camera) if args.camera.isdigit() else args.camera, args.songs)
    else:
        main()
