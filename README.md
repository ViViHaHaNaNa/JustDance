# Pose Party

A Just Dance-style game built on OpenCV and MediaPipe. A dancer in a video moves
on the left of the screen; you copy them on the right in front of your webcam.
Key moves scroll along the bottom bar toward a hit zone; when one lands you're
rated OK / GOOD / PERFECT by comparing your joint angles to the dancer's, and in
between you earn small bonus points for staying roughly in sync. At the end you
get a grade (S/A/B/C) and a photo of your best moment.

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

   Each song also gets a `song.json` with the title shown in the menu. Set it
   with `--title "Song name"`.

   **Game footage (Just Dance etc.):** cut out the HUD and the game's own move
   pictograms with `--crop X0,Y0,X1,Y1` (pixels, or fractions of the frame), and
   skip menus and results screens with `--start`/`--end`. The crop is applied to
   pose detection, the 720p copy and the preview, so the panel in the game shows
   only that region:

   ```bash
   python extract_dance.py MovesLikeJagger.mp4 --start 25.9 --end 230.5        --crop 319,29,945,614 --title "Moves Like Jagger" --out output/moves_like_jagger
   ```

2. Play:

   ```bash
   python pose_party.py --dance                    # song menu: every song in output/
   python pose_party.py --dance output/<folder>    # straight into one song
   ```

   The first run also tracks the song's beats (key moves land on them) and caches
   them as `beats.npy` in that folder.

**Several songs at once:** list them in a JSON file (copy `songs.example.json`) and
run

```bash
python extract_dance.py --batch songs.json --jobs 2
```

Each song takes the same options as the command line (`video`, `title`, `start`,
`end`, `crop`, `out`); put options shared by all of them under `"defaults"` (game
footage from the same capture usually shares one `crop`). Songs run `--jobs` at a
time in parallel (2 is a safe start on a laptop), logs go in
`output/_batch_logs/`, and a summary at the end lists each song's tracked
percentage and flags any under 90%. Check a song's `preview.mp4` before trusting it.
**Adding songs later:** append them to `songs.json` and run with `--skip-existing` so
the ones you already extracted aren't redone, or pick songs by title with
`--only "Sadi" "Beauty"`.

**Two dancers in the video:** the extractor tracks one person. It starts on the
biggest one and then sticks with whoever is closest to where they were in the last
frame, so it doesn't flip between dancers. Add `"follow": "left"` (or `"right"`, or
`--follow left` on the command line) to start on the leftmost/rightmost dancer, and
crop to that dancer's side if the other one wanders into frame.
If you leave out `title` it's made from the file name ("MovesLikeJagger.mp4" becomes
"Moves Like Jagger").

## Controls

| Where | Keys |
|---|---|
| Song menu | **UP** / **DOWN** (or **W** / **S** / **TAB**) choose, **ENTER** play, **ESC** quit |
| Name screen | type your name, **ENTER** start, **ESC** back to the menu |
| Count-in | stand inside the outline (the 3-2-1 only runs while you're in it), **SPACE** skips that check |
| During the song | **SPACE** pause, **R** restart, **Q** / **ESC** quit |
| Result screen | **R** replay, **N** new player, **M** song menu, **Q** quit. After 20 s with no key it returns to the song menu |

Photos of your best move are saved to `photos/`, and every song keeps its own
leaderboard in `leaderboard.json` inside its folder (shown on the result screen and
next to the song in the menu).

## How it plays

- The dancer performs the whole song live. You are scored against their pose
  *right now*: the best joint-angle match within ±0.3 s.
- A pose is a set of 8 joint angles (elbows, shoulders, hips, knees). Positions
  are never compared, so it doesn't matter how tall you are or where you stand.
  Frames the timeline marks as lost are not scored.
- **Key moves** are picked automatically from the timeline, one every 2-3 s: beats
  where the dancer is tracked, holds a pose that reads as something nameable, and
  isn't mid-swing. Each gets a pictogram and a 1-2 word label ("Arms
  up", "Lean left", ...). The pictograms are clean mannequin icons for the standard
  labels (`ICONS` / `draw_icon`); a custom label in `moves.json` falls back to the
  dancer's real pose at that moment. They scroll toward the hit zone on the left, a ring
  shrinks around each as it approaches, and the next one counts 3-2-1.
- **Turns:** when the dancer's shoulders close up to a line near a beat (they've
  turned side-on or are spinning) the move is a "Turn" with a circular arrow. Arm
  positions mean nothing mid-turn, so a turn is rated on whether *your* shoulders
  narrow too, i.e. whether you actually turn.
- When a move lands, your best frames around that moment are rated OK / GOOD /
  PERFECT (or MISS) and feed the combo. Not being visible to the camera counts as
  0 for a move.
- **Bonus points:** between moves, every 0.25 s that your match is above the white
  tick on the match bar you earn +10 (a green "+10" floats off your head and the
  BONUS counter goes up). That's how copying the dancer's other moves, the ones
  not on the bar, scores.
- **Photo:** taken from your best-rated *move*, picking the frame where you match
  best with your arms out. (The best-matching frames over the whole song are just
  you and the dancer both standing still, which makes a bad photo.)
- **Leaderboard:** each finished run is saved with name, score, grade and accuracy.
- **Standing still doesn't score.** A frozen pose happens to resemble some frame of
  almost any dance (a player doing nothing used to get a B), so if the dancer moved
  over the last second and you barely did, your match is scaled down (to 40% at
  worst). See `STILL_WINDOW`, `STILL_DANCER_MIN`, `STILL_FLOOR`.
- Each limb of your skeleton is coloured by its own angle error, green to red.
- The grade comes from the average rating of the moves and your longest streak.

## Tuning

Almost everything is a constant near the top of `pose_party.py` or of the dance
section (search for `PANEL_H`):

| Constant | Effect |
|---|---|
| `ANGLE_TOLERANCE`, `WORST_TOLERANCE` | How forgiving a pose match is (higher is easier) |
| `RATINGS` | Match thresholds and points for PERFECT / GOOD / OK / MISS |
| `WINDOW` | Timing tolerance around the dancer's pose |
| `BONUS_MATCH`, `BONUS_EVERY`, `BONUS_POINTS` | Bonus between moves |
| `BAR_SECONDS`, `APPROACH` | How far ahead the bar looks, and when the ring/3-2-1 starts |
| `LIMB_TOLERANCE` | Degrees off at which a limb turns fully red |
| `TURN_RATIO` | How narrow the shoulders must get for a move to count as a turn |
| `GRADES` | Thresholds for S/A/B/C |
| `SCORE_IN_3D` | Score with 3D world angles instead of 2D image angles (see below) |

Which poses get picked and what they're called lives in `classify_pose` and
`pick_key_moves`.

**Fixing a song's moves by hand:** `python pose_party.py --export-moves output/<folder>`
writes the automatic picks to `moves.json` in that folder: a list of
`{"t": seconds, "label": "Arms up"}`. Edit labels and times, delete moves you don't
like, add your own (a label of "Turn" makes a turn), and the game uses that list
instead. Delete the file to go back to automatic.

## Known limitations and ideas

Things we know are rough, roughly in order of how much they'd help:

- **Not tested on a real game night.** The scoring was only validated headless by
  feeding the dancer's own video in as the "webcam" (a live copy scores well, one
  that is 1.5 s late scores badly). Thresholds and grade cut-offs haven't been
  tuned with real players.
- **Move labels are heuristics.** `classify_pose` names a pose from arm height and
  reach, lean, and knee bend, and was tuned on two dancers. Subtle choreography
  (hands brought in toward the body, one leg raised) still gets named "Clap" or
  "Squat", and some stretches of a song end up with no moves. Spins are only
  detected when the shoulders go side-on, and a full turn to face away looks like
  facing the camera again (the timeline renames left/right per frame). Use
  `moves.json` (above) to fix these by hand; better pose names would help a lot.
- **Scoring uses 2D angles.** MediaPipe's 3D world angles are noisier: even a
  perfect copy of the dancer scored about half as many PERFECTs. 2D is what the
  game uses by default (`SCORE_IN_3D = False`), which means limbs pointing
  toward or away from the camera are compared poorly.
- **Some angles drop out.** Knees and hips are often hidden (360 degree turns,
  limbs behind the body). Missing angles are skipped rather than counted as
  misses, so hiding your arms between moves just earns no bonus; a landing move
  does count it as 0.
- **Beat tracking is simple.** It's a numpy-only onset + dynamic-programming
  tracker, and has a hard-coded +45 ms phase correction that was calibrated on one
  song. It might double/halve the tempo on other music, which only changes where
  moves can land.
- **Extraction assumes one dancer** and keeps whichever person is largest in
  frame. Cuts to close-ups or title cards show up as lost stretches.
- **Audio sync** uses the output device's reported latency. Bluetooth headphones
  can add more.
- Only developed and run on Windows. There are no automated tests yet.
- The old random-pose mode (`python pose_party.py` with no arguments) is still
  there but hasn't been touched much.
