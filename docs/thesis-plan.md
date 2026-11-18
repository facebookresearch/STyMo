# Thesis plan: dance style transfer (hiphop <-> ballet) on STyMo

_Status (2026-11-17): Q1-Q9 answered (Q5-Q9 below). Grilling paused; current focus is a few-hour advisor demo (next section). Remaining grilling items are under "Not yet asked"._

Hagar prefers to see numbered options in the text and answers by index in plain text, not through the dialog.

## Advisor demo plan (few hours)

**Goal:** show the advisor STyMo trained on the Step Up pair (man = hiphop source, woman = modern ballet target) and applied to a hiphop sequence longer than the 1.6 s pair, as side-by-side video. Expected outcome: mostly posture transfers, little ballet quality (consistent with STATUS.md findings). That result is itself the step 1 proof of concept.

**Key fact about "longer":** STyMo does not generate new choreography. It restyles whatever input motion it gets, frame for frame, at any length. So a longer ballet-ish dance requires a longer hiphop input. On this machine the candidates are:
- Step Up: solo man clip 2.33 s (`stepup_man_0059_0102`, film 0:59-1:02) and `stepup_0058_0102` 3.97 s (subject to check). The full film is not on the machine, only these short cut videos.
- FineDance hiphop: clean mocap, minutes long, 18 Hiphop clips. Needs a conversion to BVH (~1 h).
- Default: use both. The 2.3 s and 4 s film clips as "in-film" results, plus a 30-60 s FineDance hiphop excerpt as the "long choreography". If Hagar has the full film, a longer GVHMR extraction of the man is a better in-film input (later).

**Verified:** the duo man BVH equals the last 49 frames of the solo man BVH (offset 21, exact match), so the duo pair is time-aligned with the woman (film 1:00-1:02). Training pair = duo folder as is.

### Steps

0. **Save progress (first thing after leaving plan mode):** copy this decision log into the repo as `docs/thesis-plan.md`; add a "Thesis: dance style transfer" phase + log entry to `STATUS.md`; add a memory entry (thesis direction + "show numbered options in text"). Commit only if Hagar asks.
1. **SMPL skeleton support (route A, minimal):**
   - `headless/smpl_to_stymo.py` (new): read SMPL-22 BVH with pymotion (`pymotion.io.bvh`), add a `Root` joint (pelvis x/z on the floor, y = 0, yaw-only heading from the hip line, lightly smoothed), re-express pelvis relative to Root, shift floor to lowest foot = 0, cm -> m. Write BVH. Inverse not needed for the demo.
   - `src/body_parts.py`: add one SMPL entry (Root, pelvis, spine1-3, neck, head / legs / arms) and `["left_foot","right_foot"]` to the feet list, following the existing `*_joint` entries.
   - Check: FK positions of converted vs original within ~1e-4 m; `get_body_parts_indices` / `get_feet_indices` return expected joints.
2. **FineDance hiphop -> BVH** (`headless/finedance_to_bvh.py`, run in the `gvhmr` env): pick one Hiphop clip from `label_json/`, take a 30-60 s excerpt, decode 6D as rows (pytorch3d convention; verify by rendering one frame, knees must bend forward), axis-angle, write with GVHMR `tools/demo/export_bvh.py` `build_bvh`/`write_hierarchy` (fps 30). Then step 1's converter.
3. **Train:** style folder `motion/stepup/ballet/{source,target}/stepup_0100_0102.bvh` (man, woman converted), inputs in `motion/stepup/hiphop/`. `python src/stylizer_train.py motion/stepup/ballet/ motion/stepup/hiphop/ --epochs 100 --name demo` on GPU 0. Risk: 49 frames -> ~98 at 60 fps is near the ±32-frame window and 80/20 split; fallback `--window_frames 16` or train/val on the same frames (overfitting is fine for this demo).
4. **Stylize:** `headless/stymo_headless.py` on each input with `--variant 1,1 --variant 2,1.5` (default and amplified).
5. **Render:** `headless/blender_replay.py` side by side (hiphop input vs ballet output), plus the woman's real clip for reference. ~0.25 s/frame: 60 s clip at 60 fps ≈ 15 min. Check that the pymotion Blender script draws the SMPL skeleton correctly.
6. **Numbers (optional if time):** `headless/compare_motion.py` posture/dynamics change for each output.

**Deliverable:** 3-4 mp4s in `render_results/stepup_demo/` + a short section in STATUS.md with what to tell the advisor.

**Time budget:** steps 0-1 ~1.5 h, step 2 ~1 h, steps 3-5 ~1 h of compute (mostly rendering), slack 30 min. If time runs short, drop step 2 and demo on the two film clips only.

### Verification
- Converter FK round-trip check passes; skeleton loads in STyMo without "No target body part found".
- Training losses fall; model on its own training input reproduces the woman's clip closely (sanity, as done for `angry`).
- Videos have the right frame count and visibly show both characters on the floor at the right scale.

## Context

Hagar's thesis: take a dance given as 3D human motion and transfer it between styles (ballet <-> hiphop, inspired by Step Up). Starting point is the STyMo repo (SIGGRAPH 2026), which learns a style from a ~2-4 s aligned pair of clips.

- Step 1: test STyMo on a hiphop/ballet pair taken from the film; expected to fail.
- Step 2: build a neutraliser for dance, first heuristic (baseline), then learned.

## Facts found

- Step Up clips: `~/projects/GVHMR/outputs/demo/stepup_*`, 5 BVH files from GVHMR video pose estimation. SMPL 22 joints, ZXY, centimetres, 29.97 fps. Named man/woman, not hiphop/ballet.
- Only matching pair: `stepup_duo_0100_0102/` (man + woman, 49 frames = 1.63 s each). Confirmed same time window: duo man = solo man frames 21-69. Man = hiphop, woman = modern ballet (Hagar, 2026-11-17).
- STyMo needs: `source/` + `target/` with identical filenames, aligned frame by frame, same skeleton as the inference input. Joint names hard-coded (`src/body_parts.py`, `src/features.py:24-50`). No retargeting, no time alignment. Resamples to 60 fps itself; `--scale 0.01` for centimetres.
- AMASS (2026-11-18): `/mnt/raid/home/ofir_abramovich/datasets/AMASS/AMASS/extracted/<subset>/` (another user's folder, readable). SMPL-X npz (`poses` (T,165), `trans`, `betas`, 120 fps in DanceDB). 24 subsets. Dance: DanceDB 171 files / 7.5 GB (20 are `*_stagei.npz` shape-only files), plus a few in MoSh, SFU, SOMA, HUMAN4D, KIT.
- DanceDB has no ballet. It has many Greek folk dances, salsa, bachata, flamenco, 1 hiphop (2.2 min), 1 musical theatre, and an emotion set: performers Vasso, Andria, Elena each dance ~20-35 s clips in ~13 emotions (happy, sad, angry, tired...) plus **Neutral** (Vasso v2-v4, Elena v2). If each performer repeats one routine, that is a ready-made (neutral, styled) set; lengths differ, so time alignment would be needed. Not yet checked by eye.
- FineDance: `~/databases/finedance`, extracted, 203 motions as (frames, 315) = 3 + 52 joints x 6D (probably SMPL-H), 30 fps assumed, music included, labels in `label_json/`. Hiphop 18 clips / 49 min (Street without Jazz: 61 clips / 124 min). No ballet; "Classic" is Chinese classical (56 clips / 157 min). No shared choreography across styles, no neutral performances.
- Existing FineDance converters: `~/projects/salad/scripts/convert_finedance.py`, `salad/utils/conversion/`.
- STATUS.md finding: STyMo carries over posture far more than dynamics on dance.

## Decision log

### Q1 Transfer goal: what must survive the transfer?
1. Same steps, new quality (choreography, timing, floor path kept)
2. Steps may be replaced by the target style's vocabulary
3. Start with same steps, aim for replaced

**Answer: 3**, with a change: timing and length of a movement may be altered. Step replacement is the next research question (or the fallback if same-steps is too easy or not novel enough).

### Q2 What does "smooth out" mean?
1. Blend between styles (a dial from hiphop to ballet)
2. Smooth transition in time within one dance
3. Clean noisy motion
4. Neutralise the style

**Answer: 4.** Option 1 kept as a later research goal.

### Q3 What is the film pair?
1. Two different phrases
2. Same phrase, both styles
3. Similar, not aligned

**Answer: 2 or 3**; Hagar may re-time the extracted movements. (To confirm.)

### Q4 Step 1 verdict
1. By eye plus numbers (`headless/compare_motion.py`)
2. By eye only
3. Style classifier

**Answer: all, staged.** 2 first (proof it does not work), 1 as intermediate, 3 for the paper.

### Q5 Skeleton: SMPL everywhere, or retarget to STyMo's 25 joints?
1. SMPL everywhere (add SMPL names to STyMo)
2. Retarget dance data to the MoCha skeleton

**Answer (2026-11-17): 1, Route A (SMPL plus an added Root joint).** Investigation findings:

- The real gap is not joint names or count. Every skeleton STyMo supports has a ground-level `Root` joint (yaw only, y = 0) above the pelvis, with the pelvis at index 1 carrying its own position channels. SMPL-22 has no such joint. `src/features.py:91-97` raises on this, and it guards a set of index-0/index-1 assumptions (`motion_loss.py:41-49,121-156`, `utils_inference.py:152-196,276-288`, `stylizer_train.py:625-628`) that would otherwise give wrong results silently.
- Units must be metres. With centimetres and no `--scale 0.01`, foot contacts are almost never detected (threshold 0.25 m/s, `stylizer_train.py:705`) and nothing errors. `--scale` is also applied twice between training and the app (`stylizer_train.py:599-647` vs `stylizer_app.py:216,264`) and in `extract_average_pose.py`; writing metres into the files avoids all of it.
- Model sizes follow joint count, so each skeleton trains from scratch (minutes). No shipped checkpoints. Body parts only group joints for a regulariser and the app sliders; they do not shape the network.
- SMPL `left_foot`/`right_foot` are equivalent to MoCha's toe joints for contact detection.
- No retargeting tool exists on this machine (none in pymotion, Blender add-ons, or sibling projects).
- Route A (recommended by the investigation), "SMPL plus a Root joint": a converter script (adds `Root`, re-expresses pelvis, floor to 0, cm to m; ~150-250 lines) + one SMPL entry in `src/body_parts.py` (~30 lines) + FineDance-to-BVH via GVHMR's `tools/demo/export_bvh.py` (~80-120 lines). About 1.5-3 days with checks. No change to model code. Costs: STyMo's 11 MoCha style pairs are unusable unless those short pairs are retargeted MoCha->SMPL (2-3 more days; this direction merges joints, which is well-defined).
- Route B, retarget all dance data to MoCha-25: ~300-500 lines, 4-8 days plus tuning. Must invent an extra spine joint, `Neck1` and bridge a Z-up vs Y-up rest pose for every long dance clip; typical artifacts are foot sliding, kinked torso, wrist/shoulder flips. Only upside: the 10 trained MoCha styles run on dance immediately.
- Checks for route A: converter round trip (FK positions within 1e-5 m), feature round trip without training, identity-pair training, contact fraction per foot, headless render.
- Side finding, unverified: `salad/utils/conversion/smplh_fk.py:71` may decode FineDance's 6D rotations with the wrong layout (columns vs rows), which would invert every local rotation. Confirm by rendering one clip before trusting salad's FineDance positions.
- Not determined: whether GVHMR's floor is at y = 0, contact thresholds suited to dance, how MoCha's `Root` yaw was smoothed.
- Follow-up question this raises: where do the paired style examples for dance come from, the MoCha pairs or new SMPL data? (Ties into Q7 and Q9.)

### Q6 Step 1 pair: duo pair as is (man = hiphop source, woman = ballet target)?
1. Yes, duo pair as is
2. Re-time first
3. Different clips

**Answer: 1** (first).

### Q7 Role of neutral
1. Neutral is the bridge (hiphop -> neutral -> ballet)
2. Neutral only for making training pairs; final model direct
3. Neutraliser is the contribution; transfer is future work

**Answer: 1** for now; options 2 and 3 kept as future research.

### Q8 Film clips
1. Demo and qualitative test only
2. Core training data
3. Both: mocap for training, film as a second test set

**Answer: 3** for now; to be clarified further.

### Q9 Ballet data source
1. AMASS first (needs the real path)
2. Chinese classical as stand-in
3. Find a public ballet dataset
4. Extract from video with GVHMR

**Answer: 1 + 3.** Hagar will provide AMASS; meanwhile research more ballet data sources.

### Q10 What does "neutral" mean for a dance? (asked 2026-11-17)
1. Posture only
2. Posture and dynamics (also remove sharp/percussive quality)
3. Posture, dynamics and timing

Recommended: 2. **Open.** Background: `docs/research/neutralize-dance.md`.

### Q11 Neutral relative to what?
1. Everyday non-dance movement (AMASS-like)
2. The average of the dance styles ("generic dance")

Recommended: 2. **Open.**

### Q12 Download Bandai Namco motion dataset part 1? (has a `normal` style and a `dance` motion; CC BY-NC 4.0; skeleton undocumented)
1. Yes, download and check the dance files and skeleton
2. Not now, decide after the advisor meeting

Recommended: 1. **Open.**

**Update 2026-11-18:** `docs/research/neutralize-movement.md` checked the repository: dataset-1 has only `dance-long_normal_001` and `dance-short_normal_001`, no multi-style `dance`. Recommendation is now 2 (low value).

### Q13 Check whether DanceDB's emotion clips are one routine in several emotions? (asked 2026-11-18)
1. Yes, render Vasso's Neutral next to two of her emotion clips (~20 min)
2. Later, after the advisor meeting

Recommended: 1. **Open.**

## Ballet lead in AMASS

CMU subject 05 (20 trials, ballet vocabulary: pirouette, arabesque, jeté) per `docs/research/ballet-datasets.md`, now reachable in the AMASS copy at `.../AMASS/extracted/CMU/CMU/05/` (all 20 trials present, verified 2026-11-18). Content not yet viewed.

## Not yet asked (blocked on the above)

- What "neutral" means for a dance, operationally (blocked on Q7)
- Heuristic neutraliser design; learned model or loss (blocked on the definition)
- How timing/length changes are allowed and measured (Q1 follow-up)
- Use of music/beat
- Evaluation metrics and classifier data (blocked on Q9)
- Thesis scope, timeline, paper target
