# Pose Party

A Just Dance-style game built on OpenCV and MediaPipe. A dancer in a video moves
on the left of the screen; you copy them on the right in front of your webcam.
Every beat you're rated OK / GOOD / PERFECT by comparing your joint angles to
the dancer's, and at the end you get a grade (S/A/B/C) and a photo of your best
moment.

There are two parts:

| File | What it does |
|---|---|
| `extract_dance.py` | Offline step. Runs pose estimation on a dance video and writes a timeline (joint angles, 3D/2D joints, tracked/lost flags), the audio, and a 720p copy of the video. |
| `pose_party.py` | The game. `--dance <folder>` plays a song from an extracted timeline. With no arguments it runs the older random-pose mode. |

## Setup

Developed on Python 3.13 (Windows 11); other versions are untested. You need a webcam.

```bash
pip install -r requirements.txt
```

ffmpeg is needed by `extract_dance.py`. It uses the one on your `PATH` if there
is one, otherwise the binary that `imageio-ffmpeg` ships with.

The pose models are downloaded on first use into the working directory
(`pose_landmarker_lite.task` for the game, `pose_landmarker_heavy.task` for
extraction). Run everything from the repo root.

## Quick start

No dance video is included (they're large and not ours to redistribute), so bring
your own: a single dancer, fairly static camera, whole body visible.

1. Extract a timeline. Try a 30 s clip first, with `--start`/`--end` around a chorus:

   ```bash
   python extract_dance.py path/to/dance.mp4 --start 48 --end 78
   python extract_dance.py path/to/dance.mp4          # whole video
   ```

   This writes `output/<video name>[_start-end]/` containing `timeline.npz`,
   `audio.wav`, `video_720p.mp4` and `preview.mp4` (the original with the
   extracted skeleton drawn on top; watch it to check the tracking). At the end it
   prints the tracked percentage and the longest stretches where the dancer was
   lost.

   Extraction runs at roughly 7 fps on a laptop CPU, so a 3 minute video takes
   about 20 minutes.

2. Play:

   ```bash
   python pose_party.py --dance output/<folder> --difficulty medium
   ```

   The first run also tracks the song's beats and caches them as `beats.npy` in
   that folder.

## Controls

| Where | Keys |
|---|---|
| Start screen | type your name, **TAB** difficulty, **ENTER** start, **ESC** quit |
| Count-in | stand inside the outline (the 3-2-1 only runs while you're in it), **SPACE** skips that check |
| During the song | **SPACE** pause, **R** restart, **Q** / **ESC** quit |
| Result screen | **R** replay, **N** new player, **Q** quit. After 20 s with no key it returns to the start screen |

Photos of your best moment are saved to `photos/`.

## How scoring works

- You are scored against the pose the dancer struck some time *earlier*, so you
  get time to follow along. `--difficulty` sets that delay: easy 2 s, medium
  1.5 s, hard 0.3 s. The first few seconds of a song are therefore unscored.
- A pose is a set of 8 joint angles (elbows, shoulders, hips, knees). Positions
  are never compared, so it doesn't matter how tall you are or where you stand.
- Each frame takes the best match within ±0.3 s of the target pose. Frames the
  timeline marks as lost are not scored, and neither are frames where the camera
  can't see your arms.
- Every beat (from the audio, ~91 BPM for the test song) is rated on the frames
  around it. Combos and the final grade come from those ratings.
- Each limb of your skeleton is coloured by its own angle error, green to red.
  A white ghost shows the pose you're being scored against, a faded one shows
  the pose 0.7 s later, and the corner strip previews the next three.

## Tuning

Almost everything is a constant near the top of `pose_party.py` or of the dance
section (search for `DIFFICULTY`):

| Constant | Effect |
|---|---|
| `DIFFICULTY` | Delay in seconds per level |
| `ANGLE_TOLERANCE`, `WORST_TOLERANCE` | How forgiving a pose match is (higher is easier) |
| `RATINGS` | Match thresholds and points for PERFECT / GOOD / OK / MISS |
| `WINDOW` | Timing tolerance around the target pose |
| `LIMB_TOLERANCE` | Degrees off at which a limb turns fully red |
| `GRADES` | Thresholds for S/A/B/C |
| `SCORE_IN_3D` | Score with 3D world angles instead of 2D image angles (see below) |

## Known limitations and ideas

Things we know are rough, roughly in order of how much they'd help:

- **Not tested on a real game night.** The scoring was validated headless by
  feeding the dancer's own video in as the "webcam" (right delay scores S, wrong
  delay scores C), but difficulty, thresholds and grade cut-offs haven't been
  tuned with real players.
- **Scoring uses 2D angles.** MediaPipe's 3D world angles are noisier: even a
  perfect copy of the dancer scored about half as many PERFECTs. 2D is what the
  game uses by default (`SCORE_IN_3D = False`), which means limbs pointing
  toward or away from the camera are compared poorly.
- **Some angles drop out.** Knees and hips are often hidden (360 degree turns,
  limbs behind the body). Missing angles are skipped, not counted as misses, so
  hiding your arms avoids a MISS.
- **Beat tracking is simple.** It's a numpy-only onset + dynamic-programming
  tracker, and has a hard-coded +45 ms phase correction that was calibrated on one
  song. It might double/halve the tempo on other music.
- **Extraction assumes one dancer** and keeps whichever person is largest in
  frame. Cuts to close-ups or title cards show up as lost stretches.
- **Audio sync** uses the output device's reported latency. Bluetooth headphones
  can add more.
- Only developed and run on Windows. There are no automated tests yet.
- The old random-pose mode (`python pose_party.py` with no arguments) is still
  there but hasn't been touched much.
