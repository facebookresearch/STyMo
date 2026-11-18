# Research: public datasets with ballet motion

_2026-11-17. Scope: which public datasets contain classical ballet as 3D motion (or as video we could lift to 3D), and how usable each is for the thesis. Target format is SMPL-22 at 30 fps with an added Root joint (Q5 in [`../thesis-plan.md`](../thesis-plan.md)). Ballet source decision is Q9 (AMASS first + find a public ballet dataset). Nothing was downloaded except small label/caption files (AIOZ-GDANCE labels, HumanML3D captions, Motion-X++ caption zips) and file listings. "Unverified" marks claims I could not trace to the primary source._

## Short answer

There is no large, clean, solo classical-ballet mocap dataset in SMPL. Ballet exists only in small pieces spread over several datasets. Ranked for this thesis:

1. **CMU mocap subject 05 (inside AMASS CMU).** 20 trials of one dancer whose descriptions are ballet vocabulary: pirouette, arabesque, jeté en tournant, glissade, rond de jambe, pointe tendue, coupé dessous. Labelled "modern dance" by CMU, but the steps are ballet. Optical mocap, 120 fps, and through AMASS it is already SMPL-H / SMPL-X, so it needs only the Q5 converter plus resampling to 30 fps. Raw CMU data is free for all uses. No music. Best first target once Hagar provides AMASS.
2. **DD100 "Ballet" (Pas de Deux), from Duolando (ICLR 2024).** Real classical ballet, optical mocap at 120 fps, SMPL-X, with music. It is a duet, so lifts and partner support are in the motion. Ballet is 1 of 10 genres in about 117 minutes total, so roughly 10 to 12 minutes of ballet (estimate, not stated). No hiphop. Best "real ballet with music" source.
3. **ImperialDance (AAAI 2024).** The only dataset found with both **Ballet and Hip-Hop** captured in the same studio, same skeleton, with music. Five ballet choreographies, each danced 100 times by dancers of three skill levels. Weak points: 3D joint positions only (21 joints, OptiTrack, 100 fps, no rotations), clips cut to 10 s, and the choreographies are short and repeated. Good for a style classifier and for evaluation. Needs IK to reach SMPL.
4. **Dance Revolution dataset (ICLR 2021).** Ballet, Hiphop and Japanese Pop as 1-minute video clips, about 12 hours total. Only 2D OpenPose keypoints are released. Usable as a list of ballet and hiphop videos to run through GVHMR, the same pipeline as the Step Up clips. Clip counts and video availability are unverified.
5. **University of Cyprus Dance Motion Capture Database (DanceDB).** Only 3 entries tagged "ballet", all one dancer, 2 distinct pieces. Direct BVH/C3D/FBX download. The ballet entries are **not** in the AMASS DanceDB release. Small, but a clean extra test clip.
6. **AIST++ "Ballet Jazz".** Not classical ballet: AIST lists it under "New School" street styles. SMPL, 60 fps, with music, and it sits next to two hiphop genres in the same dataset. Useful as a "balletic street jazz" midpoint, not as the ballet target.

Everything else checked (FineDance, PopDanceSet, Motorica, PhantomDance, AIOZ-GDANCE, InterDance, MDD, Motion-X / Motion-X++, OpenDance5D, SoulDance, DanceCamera3D, Chang-E, Bandai Namco) has no verified classical ballet. Two 2026 or recent sets do have ballet but are not usable yet: DanceFlow (DanceCrafter, 2026; release unclear, MHR format) and the Transflower SMVibe subset (never released).

All usable options are non-commercial research licences except raw CMU. That is fine for a thesis.

## Comparison table

| Dataset | Ballet? | Ballet amount | Hiphop in same set? | Capture | Format / joints | fps | Music | Licence / access | Link checked |
|---|---|---|---|---|---|---|---|---|---|
| CMU mocap subject 05 (via AMASS CMU) | Yes, ballet steps (labelled "modern dance") | 1 dancer, 20 trials; length not checked | No (CMU has 1 breakdance subject, 85) | Optical (Vicon) | ASF/AMC, C3D; AMASS gives SMPL-H/SMPL-X | 120 | No | Raw CMU: "free for all uses". AMASS: registration, non-commercial | CMU site: yes. AMASS: login needed |
| DD100 (Duolando) | Yes, "Pas-de-Deux ballet" | 1 of 10 genres; ~117 min total; ballet minutes not stated | No | Optical, 20 cameras, 53 markers + inertial gloves | SMPL-X (.npy per dancer) | 120 | Yes | NTU S-Lab License 1.0 (non-commercial, text not re-read). Google Drive zip | Drive page reachable; sign-in prompt shown |
| ImperialDance (DanceMVP) | Yes, "Ballet" | 5 choreographies (~15 to 25 s each), 100 repeats per level, 3 levels | **Yes**, Hip-Hop (2 choreographies) + K-pop, Jazz, Urban | Optical (OptiTrack) | 3D positions of 21 joints, array (100, 3, 1000, 21) | 100 | Yes (YouTube / Bilibili IDs, not audio files) | No licence file. Google Drive folder | Folder lists b_ch0..b_ch4, h_ch0..h_ch1 |
| Dance Revolution | Yes, "Ballet" | 1-min clips; reported 136 ballet clips (unverified) | **Yes**, "Hiphop" (reported 298 clips) | Monocular video | 2D OpenPose BODY_25 keypoints only | 15 | Yes | Code MIT; data on Baidu | Not checked (Baidu) |
| UCY DanceDB | Yes, tag "ballet" (under Contemporary) | 1 dancer (Ioanna Tacha), 2 pieces ("Believer", "Piano"), 3 entries | One "Hip-Hop" piece (Fanie Alexandrou), in AMASS | Optical (PhaseSpace Impulse X2) | BVH, C3D, FBX, MP4. Ballet not in AMASS | not stated | Video only | UCY non-commercial licence, no redistribution | Table and file links resolve (needs the GEANT intermediate cert) |
| AIST++ | "Ballet Jazz" only, not classical | ~1/10 of 1,408 sequences, ~30 min (estimate) | **Yes**, Middle Hip-hop and LA-style Hip-hop | Multi-view video, triangulated, SMPL fit | SMPL-24 | 60 | Yes | Annotations CC BY 4.0; AIST Dance DB terms for video/music | Site reachable |
| FineDance | No ("Classic" = Chinese classical) | 0 | Yes | Optical (Vicon) | SMPL-H 52 joints | 30 | Yes | Own terms | Local copy |
| DanceFlow (DanceCrafter, 2026) | Yes, named in text | Not stated | "Breaking" only | 36 h video reconstruction + 5 h mocap | MHR | not stated | Yes | Not stated; release unclear | Repo exists; no data link found |
| Transflower SMVibe | "Freestyle, Ballet" | 0.67 h for the whole subset | No | VR trackers | not released | n/a | n/a | Project page: "will release soon" | Not released |

## Per-dataset notes

### CMU mocap subject 05 (and the rest of AMASS)
- CMU motion list, subject 5 "(modern dance)", trials 2 to 20. Descriptions include "expressive arms, pirouette", "sideways arabesque, turn step", "quasi-cou-de-pied, raised leg above hip-height, jete en tourant", "rond de jambe in the air, jete, turn", "glissade devant, glissade derriere, attitude/arabesque", "arms held high, pointe tendue a terre", "coupe dessous, grand jete en tourant". All at 120 fps. Trial 1 is a walk.
- Subject 49 "(modern dance, gymnastics)" has a few more balletic trials (side arabesque, curl inwards), at 120 fps.
- The ballet label is in the CMU descriptions, not in AMASS itself. AMASS keeps the CMU file names (`CMU/05/05_xx_poses.npz`), so the subset is easy to pick out.
- HumanML3D captions confirm it: of 43 HumanML3D clips whose captions use "ballet", "ballerina", "pirouette" or "arabesque", 18 come from `CMU/05`. The rest are single clips spread over CMU, BMLmovi, MPI_Limits, KIT and SFU (SFU `0018` is Chinese and Xinjiang dance, not ballet).
- CMU licence: "This dataset of motions is free for all uses" and "may be copied, modified, or redistributed without permission". The AMASS SMPL fits are under the AMASS licence (non-commercial research, no redistribution).
- Caveat: one dancer, short trials, likely trained but not necessarily a professional ballet dancer (not stated). Pointe work is not mentioned. CMU marker set has few foot markers, so turns on the ball of the foot may be smoothed. Not checked.
- Other AMASS sub-datasets: DanceDB (20 subjects, 153 sequences, ~3.5 h per the AMASS news page) has no ballet files; its content is Greek/Cypriot folk, salsa/bachata/reggaeton, one hiphop, and contemporary dances performed with emotions. HUMAN4D has "Dancing" takes (style not stated). SFU has Cha Cha, dance turns, Chinese and Xinjiang dance. File names were checked on a third-party Hugging Face mirror of AMASS, so treat the file lists as unverified until the real AMASS download is listed.

### DD100 (Duolando, ICLR 2024)
- Paper: "five Latin dances (Cha Cha, Rumba, Samba, Paso Doble, and Jive), four Modern (a.k.a. Standard) dances (Foxtrot, Tango, Quick-step, and Waltz), and Pas-de-Deux ballet". Five pairs of professional dancers.
- Capture: 20 optical cameras at 120 fps, 53 body markers; SMPL-X fitted with the SOMA pipeline; hands from inertial gloves mapped to MANO.
- Size: ~117 min in the abstract, 115.4 min in Section 2, 1.95 h in Table 1. About 80% of sequences recorded twice. Ballet minutes not stated; files are named `[GENRE]_[CHOREOGRAPH]_[TAKE]_00/01.npy` (00 = follower/lady, 01 = leader/man), so counting is easy after download.
- Caveats: duet motion with holds and lifts, so a single dancer's track contains supported poses that a solo dancer cannot do. Some music was shortened and slowed. Video not released.
- For us: SMPL-X body joints 0-21 match SMPL-22, so the Q5 converter applies after dropping hands, jaw and eyes.

### ImperialDance (DanceMVP, AAAI 2024)
- README: "five dance genres: Ballet, K-pop, Jazz, Hip-Hop, and Urban". OptiTrack, "21 skeletal body joints in 3D space in each frame at 100fps". "All the dance sequences were normalized in 10 seconds." Array (100, 3, 1000, 21) per file. 69,300 s total, 5 subjects, levels Expert, Intermediate, Beginner.
- Ballet: 5 choreographies (b_ch0 to b_ch4), each 15 to 25 s of music per the README. Hiphop: 2 choreographies (h_ch0, h_ch1).
- Joint names: RShin, RToe, RThigh, LThigh, LFoot, RShoulder, LHand, RHand, LFArm, LShoulder, RFoot, Neck, LShin, Chest, Head, LToe, Ab, RUArm, LUArm, Hip, RFArm. Positions only, so it needs IK to SMPL before STyMo can use it.
- Value: same studio, same skeleton, ballet and hiphop, plus skill levels. Fits a style classifier (Q4 option 3) and the evaluation plan. Too short and repetitive to be the main training source.
- No licence file in the repo (GitHub reports none). Music is given only as YouTube/Bilibili IDs with time ranges.

### Dance Revolution (ICLR 2021)
- Paper: "1-minute video clips, totaling about 12 hours", three styles: "Ballet", "Hiphop" and "Japanese Pop". README: released pose data in `ballet_1min/`, `hiphop_1min/`, and pop folders, plus audio, at 15 FPS.
- Per-style clip counts (136 ballet, 298 hiphop, 356 pop) come from a search summary of the paper's table; my PDF text extraction did not reach the table. Unverified.
- Released motion is 2D keypoints, not 3D. To use it we would need the source videos and GVHMR. Whether video IDs are released was not checked.

### University of Cyprus Dance Motion Capture Database
- The public performance table has 3 rows tagged "ballet", all dancer Ioanna Tacha, captured 2015-10-03 with PhaseSpace Impulse X2. Files: `Ballet_Ioanna_Tacha_Believer.bvh` (14.85 MB), `Ballet_Ioanna_Tacha_Piano.bvh` (22.76 MB, listed twice as "version 2"), plus C3D, FBX and MP4. The ballet rows have no AMASS link.
- The dance type column says "Contemporary Dance" for these rows; only the tag says ballet. Style should be checked by watching the MP4.
- Other: 134 contemporary performances (mostly emotion-labelled), 19 "other" (Flamenco, Belly Dance, RnB, Capoeira, Hip-Hop, Musical Theater, Zumba, Bachata, Reggaeton, Salsa, Pastirske, Kolo), plus Greek/Cypriot folk.
- Licence: non-commercial research, education or art; no redistribution; must cite Aristidou et al. 2019.
- The site's TLS chain is incomplete, so browsers and tools may refuse it. Fetching works once the HARICA "GEANT TLS RSA 1" intermediate is added.

### AIST++
- Paper: 10 genres, "Old School: Break, Pop, Lock, and Waack. New School: Middle Hip-hop, LA-style Hip-hop, House, Krump, Street Jazz, and Ballet Jazz". 1,408 sequences, 5.2 h, 30 dancers, SMPL at 60 fps.
- The AIST Dance DB page groups Ballet Jazz under New School, "dance styles that have appeared since the 1990s". It is a street jazz style with ballet influence, not classical ballet.
- Reconstruction: multi-view video, 2D keypoints, triangulation, then SMPL fit with a fixed mean body shape. No 3D ground truth.
- 85% "basic" short routines, 15% "advanced".

### Datasets checked with no classical ballet
- **FineDance**: local `label_json` has Classic (ShenYun 40, HanTang 13, DunHuang 2, Kun 1), Folk, Mix, Street (Hiphop 18, Popping 21, Jazz 19, Breaking 13, Urban 6, Locking 3). No ballet.
- **PopDanceSet (POPDG, CVPR 2024)**: 19 popular genres from Bilibili (CPOP, KPOP, house named), HybrIK monocular SMPL. No ballet in the paper or in the source URL list (zero "ballet" hits).
- **Motorica Dance Dataset**: Hiphop 84 min, Krumping, Popping, Locking, Jazz, Charleston, Tapping, Casual. BVH, 120 fps, OptiTrack. No ballet. Non-commercial.
- **PhantomDance**: animator-made, SMPL 24 joints, 30 fps JSON, 9.5 h. "13 genres, ranging from Urban Dance to Chinese Classical Dance". Ballet not mentioned.
- **AIOZ-GDANCE**: label file counted: Commercial 643, Aerobic 383, Zumba 359, Rumba 98, Bollywood 90, Samba 45, Irish 6. No ballet, no hiphop.
- **InterDance**: 15 genres on the project page (Rumba, Cha Cha, Waltz, Jazz, Shenyun, Jive, Tai, Uighur, Hantang, Hiphop, Dunhuang, Kpop, Urban, Miao, Samba). No ballet. Data "coming".
- **MDD (2025)**: 15 ballroom, Latin and social partner dances. No ballet or hiphop.
- **Motion-X / Motion-X++**: paper shows a "dance ballet" example in one figure only. Released Motion-X++ caption zips for `music` (instrument playing) and `perform` have zero "ballet" hits. AIST is included, so "Ballet Jazz" comes along.
- **OpenDance5D (2025)**: 101.68 h, 14 genres, WHAM monocular SMPL at 25 fps. Genre names only in a figure; ballet not mentioned in the text. Unverified.
- **SoulDance (ICCV 2025)**: 12.5 h mocap, 15 genres, names not in text. Repo says release is on hold for legal reasons (2025-09). Unverified.
- **DanceCamera3D / DCM**: anime MMD dances, "4 music genres". No ballet found.
- **Chang-E**: Dunhuang dance only, ~40 min, SMPL, licence form signed by institution staff.
- **Bandai Namco, Dancer101, the AIST++-based pro/amateur set**: not re-researched. Bandai Namco styles are emotions and characters, not genres. The pro/amateur set is built from AIST++, so at most Ballet Jazz.

## How this fits the plan

- **Training pairs (STyMo):** none of these sets has the same choreography in hiphop and ballet, so pairs still have to come from the neutraliser route (Q7, `neutralize-dance.md`).
- **Ballet target data:** CMU 05 now (once AMASS is on disk), then DD100 ballet for more and for music.
- **Style classifier and evaluation:** ImperialDance (ballet vs hiphop, same capture) plus AIST++ hiphop and FineDance hiphop.
- **Film-like test set:** Dance Revolution videos through GVHMR match the Step Up pipeline (monocular, SMPL-22, 30 fps), so they test the same failure modes.

## Open questions for Hagar

1. Does ballet vocabulary danced as "modern dance" (CMU 05) count as ballet for the thesis, or must the target be classical ballet performed as such?
2. Is duet ballet (DD100) acceptable, using one partner's track and cutting out lifts and supported turns?
3. Should we download DD100 and ImperialDance now? Both are Google Drive links; DD100 sizes not shown. This also confirms the DD100 ballet minutes.
4. Is it worth adding IK (positions to SMPL rotations) for ImperialDance, or keep it only for the classifier, which can work on positions?
5. Video route: should we look for the Dance Revolution source video IDs and run GVHMR on a few ballet clips, like Step Up?
6. AMASS path is still needed. Which AMASS version is it (SMPL-H or SMPL-X fits)? That decides the CMU 05 converter input.

## Sources

- [CMU Graphics Lab Motion Capture Database, motion list (subject 5, 49)](http://mocap.cs.cmu.edu/search.php?subjectnumber=%25&motion=%25), [home page ("free for all uses")](http://mocap.cs.cmu.edu/), [FAQ (redistribution)](http://mocap.cs.cmu.edu/faqs.php)
- [AMASS paper (arXiv 1904.03278)](https://arxiv.org/abs/1904.03278), [AMASS news (DanceDB release)](https://amass.is.tue.mpg.de/news.html), [AMASS licence](https://amass.is.tue.mpg.de/license.html)
- [HumanML3D captions and index](https://github.com/EricGuo5513/HumanML3D) (used to locate ballet clips inside AMASS)
- Third-party AMASS mirror file listing, discovery only: [realdream-ai/AMASS on Hugging Face](https://huggingface.co/datasets/realdream-ai/AMASS/tree/main/raw/DanceDB)
- [Duolando / DD100 paper (arXiv 2403.18811)](https://arxiv.org/abs/2403.18811), [repo README](https://github.com/lisiyao21/Duolando), [project page (licence)](https://lisiyao21.github.io/projects/Duolando)
- [ImperialDance repo README](https://github.com/YunZhongNikki/ImperialDance-Dataset), [DanceMVP paper (AAAI 2024)](https://ojs.aaai.org/index.php/AAAI/article/view/28893)
- [Dance Revolution paper (arXiv 2006.06119)](https://arxiv.org/abs/2006.06119), [repo README](https://github.com/stonyhu/DanceRevolution)
- [UCY Dance Motion Capture Database, performances and licence](https://dancedb.cs.ucy.ac.cy/main/performances), [FAQ (formats)](https://dancedb.cs.ucy.ac.cy/main/faq)
- [AIST++ paper (arXiv 2101.08779)](https://arxiv.org/abs/2101.08779), [AIST++ facts and licence](https://google.github.io/aistplusplus_dataset/factsfigures.html), [AIST Dance DB structure (genre groups)](https://aistdancedb.ongaaccel.jp/database_structure/)
- [FineDance repo](https://github.com/li-ronghui/FineDance), [FineDance paper (arXiv 2212.03741)](https://arxiv.org/abs/2212.03741), local `~/databases/finedance/label_json`
- [POPDG / PopDanceSet repo](https://github.com/Luke-Luo1/POPDG), [paper (arXiv 2405.03178)](https://arxiv.org/abs/2405.03178)
- [Motorica Dance Dataset README](https://github.com/simonalexanderson/MotoricaDanceDataset)
- [PhantomDance repo](https://github.com/libuyu/PhantomDanceDataset), [DanceFormer paper (arXiv 2103.10206)](https://arxiv.org/abs/2103.10206)
- [AIOZ-GDANCE repo](https://github.com/aioz-ai/AIOZ-GDANCE), [Hugging Face dataset and labels](https://huggingface.co/datasets/aiozai/AIOZ-GDANCE)
- [InterDance paper (arXiv 2412.16982)](https://arxiv.org/abs/2412.16982), [project page](https://inter-dance.github.io/)
- [MDD paper (arXiv 2508.16911)](https://arxiv.org/abs/2508.16911)
- [Motion-X paper (arXiv 2307.00818)](https://arxiv.org/abs/2307.00818), [Motion-X repo](https://github.com/IDEA-Research/Motion-X), [Motion-X++ on Hugging Face](https://huggingface.co/datasets/YuhongZhang/Motion-Xplusplus)
- [OpenDance paper (arXiv 2506.07565)](https://arxiv.org/abs/2506.07565)
- [SoulDance paper (arXiv 2507.14915)](https://arxiv.org/abs/2507.14915), [repo README](https://github.com/xjli360/SoulDance-Official)
- [DanceCrafter / DanceFlow paper (arXiv 2604.18648)](https://arxiv.org/abs/2604.18648), [project page](https://faustrazor.github.io/)
- [Transflower paper (arXiv 2106.13871)](https://arxiv.org/abs/2106.13871), [project page](https://metagen.ai/transflower)
- [DanceCamera3D repo](https://github.com/Carmenw1203/DanceCamera3D-Official)
- [Chang-E project page](https://cislab.hkust-gz.edu.cn/projects/chang-e/)
