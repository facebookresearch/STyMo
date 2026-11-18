# STyMo — Project Status

_Last updated: 2026-11-18_

**Latest result:** Dance and Uneven Floor in all 10 styles (20 videos, 472 MB) are in `render_results/dance_unevenfloor/`, named `<clip>__<style>.mp4`. Waiting on Hagar's review.

## Thesis: dance style transfer (hiphop ↔ ballet), started 2026-11-12

Plan and every decision so far (questions, options, answers): [`docs/thesis-plan.md`](docs/thesis-plan.md).

| # | Step | Status |
|---|------|--------|
| T1 | Decide direction (Q1–Q9) | ✅ Done 2026-11-17 |
| T2 | SMPL skeleton support (add ground `Root` joint, metres) | ✅ Done (demo-level) |
| T3 | Advisor demo: train on Step Up pair (man = hiphop, woman = modern ballet), stylize longer hiphop | ✅ Videos rendered; waiting on Hagar's review |
| T4 | Ballet data: AMASS path from Hagar + search for public ballet datasets | ⬜ To do |
| T5 | Neutraliser: heuristic baseline, then learned | ⬜ To do |

### Advisor demo (2026-11-17/18): Step Up hiphop → modern ballet

**Videos:** `render_results/stepup_demo/` (1080p, 60 fps). Green = hiphop input, blue = output at default strength, purple = output amplified (dynamic 2 / static 1.5).

| Video | Input | Length |
|---|---|---|
| `train_pair_man__hiphop_to_ballet.mp4` | The man's training clip itself (should come out as the woman) | 1.6 s |
| `stepup_man_0059_0102__hiphop_to_ballet.mp4` | Man, film 0:59–1:02 | 2.3 s |
| `stepup_0058_0102__hiphop_to_ballet.mp4` | Man, close-up (legs mostly out of frame, so leg motion is guessed) | 4 s |
| `finedance_094_hiphop__hiphop_to_ballet__fixroot.mp4` | FineDance hiphop "chiefRocka", seconds 20–65, root locked to the input path | 45 s |
| `finedance_094_hiphop__hiphop_to_ballet.mp4` | Same, free root: the output drifts ~2.5 m off and leaves the frame | 45 s |

**Human-body versions** (SMPL mesh instead of skeleton, 30 fps): `render_results/stepup_demo_mesh/`, same clips, named `<clip>__mesh.mp4`. `train_pair_man__mesh.mp4` adds the real woman in orange for comparison. Made with `headless/render_smpl_mesh.py` (gvhmr env, GVHMR's renderer; average SMPL body shape).

**Finding: as expected, STyMo does not learn hiphop → ballet from this pair.** Average joint-rotation difference:

| Comparison | Total | Posture (constant) | Dynamics (time-varying) |
|---|---|---|---|
| Real man vs real woman (what the model must bridge) | 26.4° | 17.2° | 22.4° |
| Model on the man's training clip vs real woman | 23.8° | 12.3° | 24.4° |
| Model output vs its input: 2.3 s / 4 s / 45 s clips | 24.1° / 23.0° / 30.5° | 15.4° / 14.0° / 16.0° | 18.9° / 17.0° / 25.3° |

- On its own training clip the output moves only ~10% closer to the woman (for `angry` it came within 2.6°). Posture moves part of the way (17° → 12°), dynamics not at all (22° → 24°).
- Why: the two dancers do different steps, so the pair has no frame-by-frame correspondence for STyMo to learn. It changes the motion a lot (~24–30°), but not into ballet.
- Over long inputs the predicted root path drifts (2.5 m in 20 s); `--fix_root` in `headless/stymo_headless.py` fixes it.
- This is the step 1 proof of concept and motivates the neutral bridge (T5): paired, aligned data is what STyMo needs.

**New tools** (in `headless/`): `smpl_to_stymo.py` (SMPL-22 BVH → STyMo layout, `--check` round trip: 2 µm), `finedance_to_bvh.py` (FineDance 315-dim → SMPL BVH, run in `gvhmr` env). SMPL entry added to `src/body_parts.py`. Model: `motion/stepup/ballet/results/Transformer-demo/` (96 s training).

## Earlier phase: reproducing the paper's results

We are first checking that STyMo (SIGGRAPH 2026) runs and produces the results the paper describes, before changing anything.

| # | Step | Status |
|---|------|--------|
| 1 | Environment with the "old" Python the requirements were pinned for (3.12) | ✅ Done |
| 2 | Train style models (`angry`, `robot`) | ✅ Done |
| 3 | Run the app without a display, render results to files | ✅ Done |
| 4 | Check that the output is really stylized (not a copy of the input) | ✅ Done: it is ([Findings](#findings)) |
| 5 | Review results across more clips and styles | ⏳ 20 videos rendered; full viewing still to do |
| 6 | Repeat with a new Python (3.15) and compare to 3.12 | ⏸ On hold until Hagar says go |

## Where to see the results

**All styles, one folder** (Dance and Uneven Floor, 20 videos):
`/mnt/raid/home/hagar_dolev/projects/STyMo/render_results/dance_unevenfloor/`

```bash
scp -r hagar_dolev@mass-01:/mnt/raid/home/hagar_dolev/projects/STyMo/render_results/dance_unevenfloor .
```

Scenes (`scene.blend`) and raw motion (`.bvh`) for each video are in `headless/<clip>/` under the style's results folder. Other results are on the server (`mass-01`) under:
`/mnt/raid/home/hagar_dolev/projects/STyMo/motion/mocha/<style>/results/Transformer-py312/`

**Robot style** (`<style>` = `robot`):

| What | Path (relative to the robot folder) | Notes |
|------|-------------------------------------|-------|
| **Dance → robot** video + scene | `headless/Dance_Neutral_AverageJoe/video.mp4`, `scene.blend` | Green = neutral, blue = robot. Clear stiff-arm posture |

**Angry style** (`<style>` = `angry`):

| What | Path (relative to the angry folder) | Notes |
|------|-------------------------------------|-------|
| **Walk** video + scene | `headless/Loco_Walk_Neutral_AverageJoe/video.mp4`, `scene.blend` | Strongest style. Green = neutral, blue = angry |
| **Dance** video + scene | `headless/Dance_Neutral_AverageJoe/video.mp4`, `scene.blend` | GUI defaults. Weak style (see findings) |
| **Dance, amplified** | `headless_amplified/Dance_Neutral_AverageJoe/video.mp4`, `scene.blend` | Blue = defaults, purple = dynamic 2 / static 1.5 |
| Stylized motion (raw) | `headless*/<clip>/pred_dyn<D>_sta<S>.bvh` | Opens in Blender, Maya, etc. |
| Trained model | `model.pt` | + `tensorboard/` training logs |

- **Videos** (`.mp4`): 1080p, 60 fps, with a color legend at the top. They play anywhere.
- **Scenes** (`.blend`): open in Blender 4.5+ on your computer, press Space to play, and orbit or scrub freely.
- **Copy to your computer**: `scp -r hagar_dolev@mass-01:/mnt/raid/home/hagar_dolev/projects/STyMo/motion/mocha/<style>/results/Transformer-py312/headless* .`
- Note: `results/` is git-ignored, so results exist only on the server.

## Findings

**The pipeline works and the model learned the style.** The `angry` style is learned from one 200-frame (3.3 s) neutral→angry pair. Average change per joint compared with the neutral input:

| Clip | Change vs neutral | Posture (constant) / dynamics (time-varying) |
|------|------------------|------------------------------------|
| Real angry (training ground truth) | 15.0° | 10.2° / 11.6° |
| Model on its own training clip | 14.8° | 10.2° / 11.1° |
| Model on **Walk** | 14.2° | 10.8° / 9.8° |
| Model on **Dance** | 8.5° | 7.3° / 3.7° |
| Model on **Dance**, amplified (dyn 2 / static 1.5) | 13.7° | 11.0° / 7.2° |

- On its training clip, the prediction matches the real angry motion to within 2.6° per joint on average, so training works.
- The style transfers well to motion like the training clip (Walk).
- **Dance looks "the same as the input" at default settings.** Dance is far from the 3.3 s example, so mostly the posture transfers and only a little of the angry dynamics. This is expected for a few-shot method: it changes *how* a motion is done, not *what* is done. Turning up the sliders helps (amplified row), but one frame reaches a 155° joint change, so watch for over-exaggeration.

**Robot style on Dance transfers almost fully.** Learned from one 163-frame (2.7 s) neutral→robot pair; the model was trained with Dance as its only input clip (`--name py312`, 144 s).

| Clip | Change vs neutral | Posture / dynamics |
|------|------------------|--------------------|
| Real robot (training ground truth) | 17.1° | 15.1° / 7.6° |
| Model on **Dance** | 15.2° | 14.4° / 3.9° |

- Robot is mostly a *posture* style (stiff, angled arms: left forearm about 52°, left arm about 43°), and posture is the part STyMo carries over to unfamiliar motion. That's why Dance → robot looks much stronger than Dance → angry, whose character is mostly in the dynamics.

**All styles on Dance and Uneven Floor** (GUI defaults: dynamic 1, static 1). Average change per joint compared with the neutral input, shown as total (posture / dynamics):

| Style | Dance | Uneven Floor |
|-------|-------|--------------|
| `sad` | 10.3° (9.6 / 3.9) | 10.1° (8.3 / 5.6) |
| `anim_keyframed` | 9.4° (9.0 / 1.4) | 10.2° (9.7 / 3.1) |
| `happy` | 10.9° (10.6 / 2.5) | 12.0° (10.2 / 6.0) |
| `angry` | 8.5° (7.3 / 3.7) | 15.0° (11.4 / 10.2) |
| `robot` | 15.2° (14.4 / 3.9) | 17.6° (16.7 / 5.1) |
| `scared` | 15.1° (14.9 / 2.6) | 19.3° (18.5 / 6.5) |
| `clown` | 19.5° (18.3 / 6.6) | 20.7° (18.6 / 9.3) |
| `clownangry` | 26.0° (24.6 / 9.0) | 28.5° (26.2 / 10.8) |
| `clownangryOne` | 28.8° (28.4 / 5.4) | 27.3° (24.7 / 12.5) |
| `zombiescared` | 28.2° (27.3 / 4.3) | 30.4° (27.8 / 9.6) |

- Every style changes the motion, and dynamics carry over more on Uneven Floor than on Dance in all 10 styles. Walking on Uneven Floor is presumably closer to the training examples.
- These numbers measure how much the motion changed, not how good it looks. The largest ones (`clownangry`, `clownangryOne`, `zombiescared`) need checking by eye for over-exaggeration: in the sampled Dance frames, `clownangry` and `clownangryOne` lift a leg high while the source stands still.
- Uneven Floor renders on the flat checkerboard, because no terrain is drawn in these scenes.

- Ways to get more style on unusual motion: add a short example of that motion (neutral + angry) to the training pair (training takes about 3.5 min), or edit the static poses with `src/extract_average_pose.py`.

## Setup (how to run)

- **Conda env `stymo`**: Python 3.12.16, torch 2.5.1+cu121, everything else pinned from `requirements.txt` (that file is UTF-16: convert it with `iconv -f utf-16 -t utf-8` before `pip install -r`).
- **GPU**: the shell sets `CUDA_VISIBLE_DEVICES=-1`, which hides the GPUs. Use `export CUDA_VISIBLE_DEVICES=0`.
- **Blender**: portable 4.5.15 LTS in `~/opt/blender-4.5.15-linux-x64`, on `PATH` only inside the `stymo` env.
- **No display on the server**, so the interactive app (`src/stylizer_app.py`) can't open its windows. The tools in `headless/` (not committed) do the same work and write files instead:
  - `stymo_headless.py`: runs the app's real prediction code with GUI default values and records what it would draw.
  - `blender_replay.py`: rebuilds that scene in background Blender, saves a `.blend` and renders an `.mp4`.
  - `compare_motion.py`: measures how much two BVH files differ (the numbers in Findings).

```bash
cd /mnt/raid/home/hagar_dolev/projects/STyMo
conda activate stymo
export CUDA_VISIBLE_DEVICES=0

# Train (about 3.5 min on an RTX 3090)
python src/stylizer_train.py motion/mocha/angry/ motion/mocha/neutral/ --epochs 100 --name py312

# Stylize clips (--variant dynamic,static = the two main GUI sliders; repeat to compare side by side)
R=motion/mocha/angry/results/Transformer-py312
python headless/stymo_headless.py $R --clips Walk --variant 1,1 --variant 2,1.5

# Render: .blend takes seconds; .mp4 about 0.25 s per frame (a 28 s clip takes about 6.5 min)
PMS=$(python -c "import pymotion.render.blender as b;print(b.pymotion_blender_script)")
D=$R/headless/Loco_Walk_Neutral_AverageJoe
blender -b --factory-startup -P headless/blender_replay.py -- $D --pymotion_script $PMS --blend $D/scene.blend --mp4 $D/video.mp4

# Measure the style
python headless/compare_motion.py motion/mocha/neutral/Loco_Walk_Neutral_AverageJoe.bvh $D/pred_dyn1_sta1.bvh
```

## Open questions / next steps

- Watch the 20 videos, especially the three strongest styles.
- Render other input clips? (a batch of 18 renders took about 80 min on GPU 0, 3 at a time)
- Compare against the paper's figures and video.
- Later (on hold): `stymo315` env with Python 3.15. The pinned numpy/scipy/scikit-learn and the cu121 torch build have no 3.15 wheels, so newer versions will be needed.

## Log

- **2026-11-12 / 11-17**: Thesis planning. Found Step Up BVHs (GVHMR, SMPL-22, cm, 30 fps), FineDance (hiphop yes, ballet no), AMASS not on this machine. Chose SMPL + added Root joint over retargeting. Decision log in `docs/thesis-plan.md`. Started the advisor demo.

- **2026-10-29**: Created `stymo` env (Python 3.12). Trained `angry` (≈3.5 min, losses fall normally; they print as 0.0000 only because of rounding). Installed portable Blender. Built the headless pipeline. Rendered Dance.
- **2026-11-03**: Dance looked the same as the input. Investigated: the run is correct and the style is weak on Dance by nature of the method. Rendered Walk and an amplified Dance for comparison. Started this file.
- **2026-11-04**: Trained `robot` style (Dance as input) on GPU 0 and rendered Dance → robot. The style transfers well because robot is mostly posture.
- **2026-11-05**: Started the batch of Dance and Uneven Floor in all styles. Confirmed Blender renders use GPU 0 (GPUs 1–3 stay idle during a render). Input clips only affect the gating model, which is off at defaults, so `robot` was not retrained for Uneven Floor. Batch finished at 13:59 with no failures: training and stylizing 12:03–12:41, rendering 12:41–13:59. All 20 videos have the right frame count (Dance 1689, Uneven Floor 2896), 1080p, 60 fps; the end of each decodes. One frame of each was checked by eye; the videos have not been watched in full.
- **2026-11-17/18**: Advisor demo. Converted Step Up + FineDance to SMPL-with-Root, trained on the Step Up pair (96 s), stylized 4 inputs, rendered to `render_results/stepup_demo/`. Result: no real hiphop→ballet transfer (see Advisor demo). Long input drifted; fixed with `--fix_root`. Neutralisation research in `docs/research/neutralize-dance.md`.
- **2026-11-18**: Rendered the demo clips as human bodies (SMPL mesh) in `render_results/stepup_demo_mesh/` with new `headless/render_smpl_mesh.py`.
