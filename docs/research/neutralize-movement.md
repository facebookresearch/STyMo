# Research: neutralising human movement (methods, implementation details, what "neutral" means)

_2026-11-17. Scope: second pass on Q7 in [`../thesis-plan.md`](../thesis-plan.md) (neutral as the bridge: hiphop → neutral → ballet). Builds on [`neutralize-dance.md`](neutralize-dance.md), which listed the option families (A1–A6 heuristics, B1–B5 learned) and the data. This file does not repeat that survey. It adds: (1) methods that explicitly output neutral motion, not covered there; (2) concrete equations and parameters for A1–A6 from the primary papers and code; (3) how the literature defines "neutral" in measurable terms; (4) a check of the Bandai Namco `dance` claim, which turned out to be wrong._

## Short answer

- **Correction to `neutralize-dance.md`:** the Bandai Namco dataset-1 has **no** multi-style dance. Listing the repository tree (GitHub API, 175 BVH files in dataset-1) shows only `dance-long_normal_001` and `dance-short_normal_001`. The README's content list says `dance`, but no file with that content exists. So there is still no public same-choreography dance in several styles plus neutral. [Bandai tree](https://api.github.com/repos/BandaiNamcoResearchInc/Bandai-Namco-Research-Motiondataset/git/trees/master?recursive=1), [dataset-1 README](https://github.com/BandaiNamcoResearchInc/Bandai-Namco-Research-Motiondataset/blob/master/dataset/Bandai-Namco-Research-Motiondataset-1/README.md)
- **The only published method built to output neutral motion from expressive motion** that I found is **Crenn et al.** (ICMI 2017, IEEE Access 2020). It minimises `Cost = α·Neutral + β·Data + γ·Penalty` in a two-level PCA space. `Neutral` is a weighted mean of 3 posture and 5 temporal features, with weights fitted so that dataset-neutral clips score near zero. Its goal is emotion recognition, not animation, and no code is linked. [Crenn 2020](https://doi.org/10.1109/ACCESS.2020.3038473)
- **The most usable recent recipe for "make a neutral twin" is Hu et al. 2024** (arXiv 2405.06646). Noise the styled clip to step 950 of 1000 (DDPM), then denoise with a text-to-motion prior and a prompt where the style words are replaced by "neutrally". Root x/z velocity is inpainted at every step to keep the path. This beat KNN+DTW retrieval on motion quality (FMD 5.87 vs 21.76) but was worse on content and style accuracy. No code. [Hu et al. 2024](https://arxiv.org/html/2405.06646v2)
- **Yumer & Mitra 2016 already ran style → neutral** as an evaluation (100 clips, against Hsu 2005 and Xia 2015), then re-stylised the outputs and measured the round-trip error. That round trip is a ready-made evaluation protocol for our neutraliser. [Yumer & Mitra 2016](https://geometry.cs.ucl.ac.uk/projects/2016/spectral-style-transfer/paper_docs/SpectralStyleTransfer.pdf)
- **"Drop the style code" does not work for AdaIN models.** Aberman et al. set AdaIN to identity (zero mean, unit variance) and "the network outputs pure noise". To get neutral from such a model you must feed a *neutral reference clip*. They also found neutral among the 3 styles closest to the centre of the style space. [Aberman 2020](https://arxiv.org/abs/2005.05751)
- **Measurable definitions of neutral exist, but they are all relative to a dataset:** (a) the dataset's neutral label; (b) EMOTE: every Effort and Shape factor at 0 on a −1…+1 scale; (c) Crenn: the centre of Russell's arousal–valence circle, scored by 8 kinematic features; (d) Aristidou 2017: 37 LMA features (speeds, accelerations, jerk, volumes) on 30 Hz dance mocap. None defines a neutral *dance*. The thesis has to choose and justify a definition. [EMOTE](https://doi.org/10.1145/344779.352172), [Aristidou 2017](https://andreasaristidou.com/publications/papers/Emotion_Control_of_Unstructured_Dance_Movements.pdf)
- **Low-pass filtering alone (A1) is a weak neutraliser.** Hsu et al. 2005 note that frequency-band approaches assume style is in the high frequencies, but differences such as normal vs limp vs crouch "occur in all bands of the spectrum". Posture (A3) and timing (A5) terms are needed too. [Hsu 2005](https://people.csail.mit.edu/ehsu/work/sig05stf/stf-sig2005.pdf)

## Correction: Bandai Namco has no multi-style dance

Checked by listing the repository tree through the GitHub API (no download). Content × style for dataset-1:

| Content | Styles present |
|---|---|
| walk, walk-back, walk-left, walk-right, run, dash, bow, bye, byebye, guide | all 15 (`active, angry, childish, chimpira, feminine, giant, happy, masculinity, musical, normal, not-confident, old, proud, sad, tired`) |
| call, kick, punch, respond, slash | `normal` only |
| dance-long, dance-short | `normal` only (one file each) |
| `dance` | **no files** |

- The README lists a content called `dance` and omits `punch`. The file names and `cfg/content_label.txt` list `dance-long` and `dance-short` instead, so `dance` in the README is a summary label, not a separate content. [README](https://github.com/BandaiNamcoResearchInc/Bandai-Namco-Research-Motiondataset/blob/master/dataset/Bandai-Namco-Research-Motiondataset-1/README.md), [content_label.txt](https://github.com/BandaiNamcoResearchInc/Bandai-Namco-Research-Motiondataset/blob/master/dataset/Bandai-Namco-Research-Motiondataset-1/cfg/content_label.txt)
- The file names use `childish`, while the README says `children`.
- README also says `respond` is not normal-only, but the files have only `respond_normal`.
- Dataset-2 (2,902 BVH) has 10 contents (walk, run, turns, raise hands, wave) in 7 styles incl. `normal`. No dance.
- What is still useful: two normal-only dance clips as examples of a "neutral dance", and 10 contents with aligned `normal` ↔ style pairs for testing a neutraliser on non-dance data.

## 1. Methods that remove style or output neutral

"Explicit neutral" means the method can be asked for neutral output and the paper shows it. "Code" was checked on GitHub where a repo exists.

| Method | How "neutral" is defined | Explicit neutral output? | Data | Representation, fps | Code, licence |
|---|---|---|---|---|---|
| Amaya, Bruderlin, Calvert 1996, *Emotion from Motion* | A "neutral" capture of the same movement | Direction is neutral → emotional, via speed and spatial-amplitude transforms | Own mocap (OPTOTRAK) | Joint trajectories | None found. [GI 1996](https://graphicsinterface.org/proceedings/gi1996/gi1996-26/) |
| Hsu, Pulli, Popović 2005, *Style Translation* | Any input style; "normal" is one of 9 locomotion styles | Yes in principle (any style pair, learned from aligned examples) | Own mocap, 120 fps, 35 DOF; styles normal, limp, crouch, sneak, flap, proud, march, catwalk, inverse | Joint angles, differential form, LTI model | Data zip on project page; no licence stated. [project](https://people.csail.mit.edu/ehsu/work/sig05stf/) |
| Holden, Saito, Komura 2016 | Not defined; style = Gram matrix of hidden units | No (transfer only) | CMU + others, 60 fps | Joint positions (local frame) | Not checked. [paper](https://www.theorangeduck.com/media/uploads/motionsynthesis.pdf) |
| Yumer & Mitra 2016 | Xia et al. neutral class | **Yes**: converted 100 stylised clips to neutral in the evaluation | Xia 2015 (8 styles incl. neutral; 6 actions), 120 fps | Per-DOF signals, root ground-plane DOF excluded | None found. [paper](https://geometry.cs.ucl.ac.uk/projects/2016/spectral-style-transfer/paper_docs/SpectralStyleTransfer.pdf) |
| Crenn et al. 2017 / 2020 | Score from 8 features; neutral = near centre of arousal–valence circle; weights fitted so dataset-neutral → 0 | **Yes**, its whole purpose | Emilya, MPI, UCLIC, "SIGGRAPH" (Xia-extended) | 90-D posture → 7-D PCA per frame → 13-D PCA per clip; quaternion DFT | No code link in paper. Paper CC BY 4.0. [2020](https://doi.org/10.1109/ACCESS.2020.3038473), [2017](https://doi.org/10.1145/3136755.3136763) |
| Smith, Cao, Neff, Wang 2019 | Xia neutral | No: neutral → 7 styles only; "neutral versions of the motion were not produced" | Xia 2015, 120 Hz, augmented to ~550k tuples | Local joint positions + separate timing network | Not checked. [paper](https://www.cs.ucdavis.edu/~neff/papers/EfficientNNStyleTransfer.pdf) |
| Aberman et al. 2020 | Xia / BFA neutral class | Only with a neutral reference clip; identity AdaIN gives noise | Xia (8 styles), BFA (16 styles) | Content: quaternions; style: 3D positions or 2D projections | [deep-motion-editing](https://github.com/DeepMotionEditing/deep-motion-editing), BSD-2-Clause |
| Motion Puzzle (Jang et al. 2022) | Xia neutral as a classifier class | No. Nulling the style net gives "content only", not shown as neutral | Trained on CMU, tested on Xia | 21 joints, 2-axis rotation + positions + velocity, 60 fps, 120-frame clips | [motion_puzzle](https://github.com/DK-Jang/motion_puzzle), MIT |
| GenMoStyle (Guo et al., ICLR 2024) | Not defined; the word "neutral" does not appear | No; can sample from an unconditional style prior | Aberman (16 styles), Xia, CMU, 30 fps | 260-D pose, 21 joints | [GenMoStyle-code](https://github.com/Murrol/GenMoStyle-code), MIT; training code "delayed", README points to OpenReview supplement |
| MCM-LDM (Song et al., CVPR 2024) | Not defined | Not shown. In code the style embedding is zeroed for classifier-free guidance, so a "no style" sample is possible (untested, my reading of the code) | HumanML3D | HumanML3D 263-D, SMPL 22 joints, 20 fps | [MCM-LDM](https://github.com/XingliangJin/MCM-LDM), **no licence file** |
| Hu et al. 2024 (arXiv 2405.06646) | Text prompt with style words replaced by "neutrally"; HumanML3D treated as "predominantly neutral" | **Yes**: generates a neutral twin for each style example | HumanML3D prior; Xia retargeted to SMPL; Bandai-2 for evaluation | Root vector + local positions + 6D rotations, HumanML3D frame rate | None. [arXiv](https://arxiv.org/html/2405.06646v2) |
| 100STYLE (Mason et al. 2022) | A style called "Neutral", not defined on the page | Dataset only | 100 styles, one actor, 60 fps, XSens BVH | 28 bones | [dataset page](https://www.ianxmason.com/100style/), CC BY 4.0 |
| AStF (Chen et al. 2025) | Not stated in abstract | No | Not stated in abstract | Adds skewness and kurtosis to AdaIN's mean/variance | [AStF](https://github.com/CHMimilanlan/AStF) (licence not checked) |
| STyMo (2026, arXiv 2609.04500) | Source is always called neutral | No: neutral → style only, reverse not evaluated | MoCha | Root-local positions + contacts in; rotation deltas out | This repo |

Not neutralisers (checked, no "neutral" or removal in abstract): LoRA-MDM "Dance Like a Chicken" (arXiv 2503.19557), MoSAIC (arXiv 2607.26304).

### Notes on the ones that matter most

**Crenn et al. 2020 (explicit neutral synthesis).**
- Cost: `Cost(M) = α·Neutral(M) + β·Data(M) + γ·Penalty(M)`, with α = 0.8, β = 0.6, γ = 0.5 "for all databases". Optimised with CMA-ES. [Crenn 2020, Sec. IV-A1](https://doi.org/10.1109/ACCESS.2020.3038473)
- `Data` keeps the result close to the input so the action stays the same. `Penalty` keeps bone lengths and stops foot sliding.
- Search space: PCA on postures (90-D → 7-D per frame), then PCA on the 7×n sequence (13 components, 95 % variance). This keeps outputs near real motion. Data-hungry steps are avoided.
- `Neutral(M) = Σ ω_j f_j / Σ ω_j`. Features are listed in section 3 below.
- The 2017 version used a simpler cost (distance plus joint accelerations). The 2020 paper says the older neutral motions were "robotic", and the new, more realistic ones raised recognition accuracy. [Crenn 2017](https://doi.org/10.1145/3136755.3136763)
- Relevance: this is option B4 in `neutralize-dance.md` with a hand-made score instead of a classifier. It is the nearest prior art for the thesis contribution and should be cited and compared.

**Hu et al. 2024 (diffusion neutral twin).**
- Steps: add G = 950 steps of noise (T = 1000, DDPM, not DDIM) to the styled clip. Denoise from G to 0 with the text-to-motion prior and a neutral prompt. At every step, overwrite the root x/z linear velocity with the input's. The paper says this inpainting "acts similarly to DTW". [Hu et al. 2024, Sec. 3.4, 4](https://arxiv.org/html/2405.06646v2)
- Prompt: style words are replaced by an AMR parser or ChatGPT 3.5. Example: "A person jumps in depression." → "A person jumps neutrally."
- Prior: 8-layer Transformer encoder, latent 512, trained like MDM. MDM is already in `~/projects/motion-diffusion-model`, so this is cheap to try (option B3).
- Measured: content alignment by foot-contact accuracy; neutrality by a CLIP-space motion–text discriminator. G was picked where the two curves "reach a balance around G=950".
- Caveat: at G = 950 almost all of the input is noise. What is kept is mainly the root path and the text. For dance, the prompt has to describe the steps, or the steps will be lost.

**Aberman et al. 2020 (what the code really does).**
- Paper: content encoder uses instance normalisation (IN) on temporal feature maps; decoder has "residual blocks with AdaIN"; an MLP maps the style code to per-channel scale and shift. [Aberman 2020](https://arxiv.org/abs/2005.05751)
- Released config is smaller than that description. Content encoder: one strided conv (128 → 144 channels, kernel 8, stride 2) with IN, plus one residual block with IN. Decoder: **one** AdaIN layer on 144 channels "before everything", then one bottleneck residual block without normalisation (`res_norm = 'none' # no adain in res`), then one upsampling conv to 31 × 4 quaternion channels. Style encoder: two convs (→ 96 → 144) and a global max pool; MLP 144 → 192 → 256 → AdaIN parameters. [config.py](https://github.com/DeepMotionEditing/deep-motion-editing/blob/master/style_transfer/config.py), [networks.py](https://github.com/DeepMotionEditing/deep-motion-editing/blob/master/style_transfer/networks.py)
- So for this model, style lives in 288 numbers (scale and shift of 144 channels), applied the same at every frame.

## 2. Implementation details for the heuristics A1–A6

Unless marked "my note", details come from the cited paper or code. STyMo itself trains at 60 fps by default (`src/stylizer_train.py`); HumanML3D-style SMPL-22 data is 20 fps ([HumanML3D README](https://github.com/EricGuo5513/HumanML3D)); the thesis data is SMPL-22 at 30 fps.

### A1. Temporal low-pass filter

- Evidence against using it alone: Hsu et al. 2005 describe the frequency-band approach of Pullen & Bregler (keep low frequencies, add high-frequency "texture") and say it "works well when style variations are contained in the high-frequency spectrum", but normal vs limp vs crouch differences "occur in all bands". I did not read Pullen & Bregler directly (PDF not reachable). [Hsu 2005, Sec. 2](https://people.csail.mit.edu/ehsu/work/sig05stf/stf-sig2005.pdf)
- Holden et al. 2016 report that a network without dropout and pooling reproduced the spine signal more closely but with "visible high frequency noise". Smoothness and fidelity trade off. [Holden 2016, Fig. 12](https://www.theorangeduck.com/media/uploads/motionsynthesis.pdf)
- My note: filter a continuous rotation form (6D, or quaternions after sign-flip to one hemisphere), not Euler angles, then re-normalise. Filter the root velocity separately, or not at all, to keep the floor path. Use a cutoff in Hz, not in frames, so 30 fps and 60 fps results are comparable.

### A2. Spectral magnitude replacement (Yumer & Mitra 2016)

- Signal: each DOF on its own. Root translation and rotation in the ground plane are left out. [Yumer & Mitra, Sec. 4](https://geometry.cs.ucl.ac.uk/projects/2016/spectral-style-transfer/paper_docs/SpectralStyleTransfer.pdf)
- Spectrum: single-sided DFT, `R[ω] = |F[ω]|` (magnitude), `A[ω] = ∠F[ω]` (phase). "The magnitude defines the existence and intensity of a motion at ω whereas the phase describes relative timing."
- Core transfer (Eq. 5): find `R'` close to `R + s·(R_s − R_r)` subject to `A' = A` and `Im(f'[t]) = 0` (real signal). `R_s`: source-style magnitude; `R_r`: reference-style magnitude of the *same action as the source*, in the input's style. `s[ω] = R[ω] / max_ω R[ω]` scales the change by the input's own energy. (Equation text was reconstructed from a damaged text extraction; check Eq. 5 in the PDF before implementing.)
- **For neutralising:** set the source to a *neutral* clip and the reference to the *same action in the input's style*. The added difference is then "neutral minus style". Both clips come from a database; the input need not match their action.
- Database match: the nearest action is found by "skeletal power", `P_i = sqrt((1/N) Σ_ω |R_i[ω]|²)` per DOF. This was found to separate actions and to be "less sensitive to style".
- Windowing (Sec. 3.2): Hann window, computed for every frame and centred on it. Window lengths **N = 17, 19, 21 frames at 120 fps** (about 0.14–0.18 s), all three kept in the database to absorb tempo differences.
- Generalised form (Eq. 8): adds a smoothness term `λ_s Σ |R' − R'_p|` (match the previous frame's window), a relaxed phase term `λ_a Σ |A' − A|`, and a parent–child phase term `λ_b Σ_j Σ |(A' − A'_j) − (A_s − A_s,j)|` over the joint's parent and children. Weights: `λ_s = 1`, `λ_a = λ_b = mean(R[ω] over ω)`.
- Solver: gradient descent per frame, initialised from the previous frame; a forward–backward pass for the first frames. Foot plants are detected with a nearest-neighbour table and cleaned up afterwards. About 50 fps on an 8-core CPU.
- My note for SMPL-22 at 30 fps: 0.14–0.18 s is only 4–5 frames, which gives 2–3 frequency bins. The windows must be set in seconds and probably made longer for dance, where one move lasts about a beat (0.4–0.6 s at 100–150 BPM). This is untested.

### A3. Pull posture towards a neutral pose

- Measurable posture targets from Crenn 2020: **body openness** (arms away from body, feet apart), **sagittal leaning** (forward/back), **body straightness** (bend of head, trunk, knees); per-frame values, then mean and std over the clip. [Crenn 2020, Sec. IV-A3](https://doi.org/10.1109/ACCESS.2020.3038473)
- Aristidou 2017 Body features (all distances, so position-based): foot–hip, hand–shoulder, hand–hand, hand–head, hand–hip, hip–ground, "hip–ground minus feet–hip", feet distance, hand–chest. [Aristidou 2017, Table 1](https://andreasaristidou.com/publications/papers/Emotion_Control_of_Unstructured_Dance_Movements.pdf)
- EMOTE Shape: torso Horizontal, Vertical and Sagittal parameters in [−1, +1] rotate key points along ellipses around the shoulder; "when no pose information is provided, the system assumes a neutral posture, where all the angles are 0". [EMOTE, Sec. 4.2.1](https://web.archive.org/web/20080704210415id_/http://www.cis.upenn.edu/~badler/siggraph00/emote.pdf)
- Amaya 1996 found (as reported in its abstract) that emotional transforms change "timing (speed) and range (spatial amplitude)". Not verified in the full text (PDF uses an unreadable font encoding). [Amaya 1996](https://graphicsinterface.org/proceedings/gi1996/gi1996-26/)

### A4. Per-channel statistics normalisation (AdaIN inverse)

- What AdaIN does in motion models: per-channel `y = γ·(x − μ(x))/σ(x) + β`, with μ and σ over time, γ and β from the style code. Aberman write the conv + norm as `x̃(τ) = β[x(τ) * k(τ) + b] + γ`, "temporally invariant". [Aberman 2020, Eq. 11](https://arxiv.org/abs/2005.05751)
- It runs on **learned features**, not on joints. In the released code it is one layer at the bottleneck (see section 1). Motion Puzzle applies it per body part ("BP-AdaIN") at every decoder level, with IN before every content STGCN layer "to remove style variations". GenMoStyle also erases "global statistic features (style)" with IN. [Motion Puzzle](https://ar5iv.labs.arxiv.org/html/2202.05274), [GenMoStyle](https://arxiv.org/html/2401.13505v2)
- Heuristic version on raw features (my note): per joint-channel c, over the clip or a window, `x'_c = σ_N,c · (x_c − μ_c)/σ_c + μ_N,c`, with μ_N, σ_N from a neutral corpus. On rotations, μ is the mean pose (overlaps with A3) and σ is the range of motion (overlaps with A6). On velocities, σ is the "energy" of the move.
- Higher-order variant: AStF adds skewness and kurtosis as style statistics. [AStF](https://arxiv.org/abs/2511.04192)
- Second-order variant: Holden 2016 Eq. 14–15, `Style(H) = s·‖G(Φ(S)) − G(H)‖² + c·‖Φ(C) − H‖²`, `G(H) = Σ_i H_i H_iᵀ / n` (Gram matrix over time of hidden units), c = 1.0, s = 0.01, H initialised from white noise and optimised. To neutralise: S = a neutral clip, C = the styled clip. [Holden 2016, Sec. 7.2](https://www.theorangeduck.com/media/uploads/motionsynthesis.pdf)

### A5. Time normalisation

- Global speed (Aberman 2020): warp time by `V_sty / V_con`, where V is the mean over frames of the maximum joint velocity. For neutralising: warp by `V_neutral / V_input`, with V_neutral from a neutral corpus. [Aberman 2020](https://arxiv.org/abs/2005.05751)
- Alignment: Hsu 2005 show DTW over-widens valleys and mismatches extrema between normal and crouched walks, and propose iterative motion warping (IMW). [Hsu 2005, Fig. 2](https://people.csail.mit.edu/ehsu/work/sig05stf/stf-sig2005.pdf)
- Hu et al. 2024 found the KNN+DTW baseline "compresses or stretches the neutral motions along the temporal axis", raising FMD. Root-velocity inpainting was their alternative. [Hu et al. 2024, Sec. 6.2](https://arxiv.org/html/2405.06646v2)
- Smith 2019 and Xia 2015 both model timing separately from pose (a "timing network"; "a local regression model predicts timing"). This supports treating A5 as its own component. [Smith 2019](https://www.cs.ucdavis.edu/~neff/papers/EfficientNNStyleTransfer.pdf), [Xia 2015 abstract](https://history.siggraph.org/?p=112760)

### A6. Range-of-motion clamping and joint locking

- Aristidou 2017 edit rule set: four rules move four control joints (pelvis, head, two hands); each rule's coefficient is fitted by least squares to reach target feature values; then full-body IK (FBBIK) fills in the rest and keeps bone lengths. Applied on keyframes (every 5th frame) and interpolated. [Aristidou 2017, Sec. 4](https://andreasaristidou.com/publications/papers/Emotion_Control_of_Unstructured_Dance_Movements.pdf)
- Crenn's Penalty term does the same job as a constraint: bone lengths "mostly constant", feet not slipping in contact. [Crenn 2020](https://doi.org/10.1109/ACCESS.2020.3038473)

### Rotations or positions?

| Heuristic | Source works on | Fit for SMPL-22, 30 fps |
|---|---|---|
| A1 low-pass | Any signal; Crenn uses quaternion spectra | Yes, on 6D or quaternions; root separate |
| A2 spectral | Per DOF (rotations in BVH data, representation not stated in the text I could read) | Yes, but windows must be re-sized in seconds |
| A3 posture | Positions (Crenn, Aristidou distances) or angles (EMOTE) | Yes; measure on positions, edit on rotations |
| A4 statistics | Learned features (Aberman, Motion Puzzle) | Raw version is my proposal, untested |
| A5 timing | Joint velocities (positions) | Yes |
| A6 clamping | Positions + IK (Aristidou) | Needs an IK for SMPL; STyMo already uses rotations out |

## 3. How "neutral" is defined operationally

Five families appear in the primary sources.

1. **Dataset label.** Neutral is whatever the performer did when told "neutral". Xia 2015 (8 styles: neutral, angry, childlike, depressed, proud, sexy, strutting/struggling, old; the label varies between papers), 100STYLE "Neutral", Bandai `normal`, Emilya "neutral". None of these pages defines it further. [Yumer & Mitra, Sec. 4](https://geometry.cs.ucl.ac.uk/projects/2016/spectral-style-transfer/paper_docs/SpectralStyleTransfer.pdf), [100STYLE](https://www.ianxmason.com/100style/)
2. **Text.** "A person … neutrally", with a general-purpose dataset (HumanML3D) assumed to be mostly neutral. [Hu et al. 2024](https://arxiv.org/html/2405.06646v2)
3. **Centre of a space.**
   - EMOTE: Effort (Space, Weight, Time, Flow) and Shape factors each in [−1, +1]; 0 = "neutral (no Effort Element)". Laban experts (CMAs) were asked to mark each factor as −1, +1 or neutral (0). [EMOTE](https://web.archive.org/web/20080704210415id_/http://www.cis.upenn.edu/~badler/siggraph00/emote.pdf)
   - Crenn: neutral sits "near the center" of Russell's arousal–valence circle. [Crenn 2020](https://doi.org/10.1109/ACCESS.2020.3038473)
   - Aberman: in learned style space, neutral is "among the top three styles" by mean distance to the others. The authors could not reach "a definitive conclusion" about why. [Aberman 2020](https://arxiv.org/abs/2005.05751)
4. **A score from kinematic features** (the most measurable).
   - Crenn 2020 neutrality function, `Neutral(M) = Σ ω_j f_j / Σ ω_j`, weights fitted on 25 % of the data so neutral clips score near 0 (Eq. 2–9):
     - Posture: body openness, sagittal leaning, body straightness (mean and std per clip).
     - Movement power: sum of spectral magnitudes of every joint's quaternion signal, all frequencies above 0.
     - Movement speed: `Σ_j (1/N) Σ_t |dP_j/dt|` over joint world positions.
     - Quantity of arm movement: cumulative distance travelled by arm joints.
     - Fluidity: mean acceleration of hands and feet, `Σ_j (1/N) Σ_t |d²P_j/dt²|`.
     - Regularity of arms: `Σ |R_left − R_right|` over the spectra of shoulder, elbow and hand quaternions (left–right asymmetry).
   - Aristidou 2017: 37 basic LMA features (Body: distances above; Effort: deceleration peaks, velocity and acceleration of pelvis, hands, feet, and jerk; Shape: volumes of 5 joints / all / upper / lower / left / right, torso height, hands level; Space: total distance, area per second, total volume). Min, max, mean, std give 121 features, normalised to [0, 1] and averaged over a **35-frame window at 30 Hz**. 31 were "effective and consistent" across 9 dancers. Data: contemporary dance, 480 Hz down to 30 Hz, 12 emotions. The paper does not define neutral; it maps features to arousal–valence by RBF regression. [Aristidou 2017, Sec. 3](https://andreasaristidou.com/publications/papers/Emotion_Control_of_Unstructured_Dance_Movements.pdf)
   - Roether et al. 2009: neutral and 4 emotional gaits of 25 people; perception relied on "overall speed" plus posture (mean flexion angles) and a few joint-specific dynamic features. [Roether 2009](https://doi.org/10.1167/9.6.15)
5. **Spectral residual.** Style = the difference in spectral magnitude between a clip and its neutral twin. Used by Yumer & Mitra for transfer and by Crenn (Eq. 10) as the expression feature for classification.

Could not verify: the computable Laban Effort formulas of Larboulette & Gibet 2015 (MOCO), which are often cited for Weight/Time/Space/Flow. The HAL full text is behind a bot check and no archive copy has the PDF. [HAL record](https://hal.science/hal-01196267)

### What this suggests for the thesis (my synthesis, not from a paper)

- A neutral dance can be defined as **"inside the neutral distribution on a fixed feature vector, while the content features stay unchanged"**. The feature vector can be Crenn's 8 features or a subset of Aristidou's 31. The neutral distribution can come from AMASS everyday motion or from Bandai `normal`.
- This splits Q2/Q7 into two measurable halves: a *style* vector that should move to the neutral range (posture openness, speed, acceleration, jerk, volume) and a *content* vector that should not move (root path, foot contacts, beat timing, key-pose order).
- Every A-heuristic then moves one style feature group, and the dose–response curves of `neutralize-dance.md` can be plotted directly in these features.

## Open questions for Hagar

1. Which "neutral" family do we commit to?
   1. Dataset label (needs a neutral corpus; AMASS or Bandai `normal`)
   2. EMOTE / LMA centre (0 on each Effort factor; needs an Effort estimator)
   3. Feature score in the style of Crenn (8 features, fitted weights)
   4. Text prompt "neutrally" with a diffusion prior (Hu et al.)
2. Should Crenn et al. be the main baseline for the learned/optimisation neutraliser (B4)? It is the only prior work that synthesises neutral motion on purpose.
3. Should we try Hu et al.'s recipe (G = 950, neutral prompt, root inpainting) on MDM, which is already installed, as a quick B3 experiment?
4. The Bandai `dance` pairs do not exist. Is it still worth downloading Bandai (CC BY-NC 4.0) for the `normal` ↔ style pairs on walk/run/gestures, to test the neutraliser where ground truth exists?
5. Do we adopt Yumer & Mitra's round trip (style → neutral → style, compare with the original) as one evaluation?

## Sources

- [Bandai Namco repo tree (GitHub API, listed 2026-11-17)](https://api.github.com/repos/BandaiNamcoResearchInc/Bandai-Namco-Research-Motiondataset/git/trees/master?recursive=1), [dataset-1 README](https://github.com/BandaiNamcoResearchInc/Bandai-Namco-Research-Motiondataset/blob/master/dataset/Bandai-Namco-Research-Motiondataset-1/README.md), [content_label.txt](https://github.com/BandaiNamcoResearchInc/Bandai-Namco-Research-Motiondataset/blob/master/dataset/Bandai-Namco-Research-Motiondataset-1/cfg/content_label.txt)
- [Amaya, Bruderlin, Calvert 1996, Emotion from Motion (Graphics Interface)](https://graphicsinterface.org/proceedings/gi1996/gi1996-26/)
- [Chi, Costa, Zhao, Badler 2000, The EMOTE Model for Effort and Shape (SIGGRAPH), archived author PDF](https://web.archive.org/web/20080704210415id_/http://www.cis.upenn.edu/~badler/siggraph00/emote.pdf), DOI 10.1145/344779.352172
- [Hsu, Pulli, Popović 2005, Style Translation for Human Motion (SIGGRAPH), paper](https://people.csail.mit.edu/ehsu/work/sig05stf/stf-sig2005.pdf), [project page with data](https://people.csail.mit.edu/ehsu/work/sig05stf/)
- [Xia, Wang, Chai, Hodgins 2015, Realtime Style Transfer for Unlabeled Heterogeneous Human Motion (abstract)](https://history.siggraph.org/?p=112760), DOI 10.1145/2766999
- [Holden, Saito, Komura 2016, A Deep Learning Framework for Character Motion Synthesis and Editing](https://www.theorangeduck.com/media/uploads/motionsynthesis.pdf)
- [Yumer & Mitra 2016, Spectral Style Transfer for Human Motion between Independent Actions (paper PDF)](https://geometry.cs.ucl.ac.uk/projects/2016/spectral-style-transfer/paper_docs/SpectralStyleTransfer.pdf)
- [Crenn, Meyer, Khan, Konik, Bouakaz 2017, Toward an Efficient Body Expression Recognition Based on the Synthesis of a Neutral Movement (ICMI)](https://doi.org/10.1145/3136755.3136763)
- [Crenn, Meyer, Konik, Khan, Bouakaz 2020, Generic Body Expression Recognition Based on Synthesis of Realistic Neutral Motion (IEEE Access, CC BY 4.0)](https://doi.org/10.1109/ACCESS.2020.3038473)
- [Aristidou et al. 2017, Emotion Control of Unstructured Dance Movements (SCA), author PDF](https://andreasaristidou.com/publications/papers/Emotion_Control_of_Unstructured_Dance_Movements.pdf). Note: the `cga2017.pdf` link in `neutralize-dance.md` now redirects to a group home page.
- [Smith, Cao, Neff, Wang 2019, Efficient Neural Networks for Real-time Motion Style Transfer](https://www.cs.ucdavis.edu/~neff/papers/EfficientNNStyleTransfer.pdf)
- [Aberman et al. 2020, Unpaired Motion Style Transfer from Video to Animation (arXiv 2005.05751)](https://arxiv.org/abs/2005.05751), [code: config.py](https://github.com/DeepMotionEditing/deep-motion-editing/blob/master/style_transfer/config.py), [networks.py](https://github.com/DeepMotionEditing/deep-motion-editing/blob/master/style_transfer/networks.py)
- [Jang et al. 2022, Motion Puzzle (arXiv 2202.05274)](https://ar5iv.labs.arxiv.org/html/2202.05274), [code](https://github.com/DK-Jang/motion_puzzle)
- [Mason, Starke, Komura 2022, Real-Time Style Modelling of Human Locomotion (arXiv 2201.04439)](https://arxiv.org/abs/2201.04439), [100STYLE dataset page](https://www.ianxmason.com/100style/)
- [Guo et al. 2024, Generative Human Motion Stylization in Latent Space (ICLR, arXiv 2401.13505)](https://arxiv.org/html/2401.13505v2), [code](https://github.com/Murrol/GenMoStyle-code)
- [Song et al. 2024, MCM-LDM (CVPR) code and README](https://github.com/XingliangJin/MCM-LDM), [mld.py conditioning code](https://github.com/XingliangJin/MCM-LDM/blob/main/mld/models/modeltype/mld.py)
- [Hu et al. 2024, Diffusion-based Human Motion Style Transfer with Semantic Guidance (arXiv 2405.06646)](https://arxiv.org/html/2405.06646v2)
- [Chen et al. 2025, AStF: Motion Style Transfer via Adaptive Statistics Fusor (arXiv 2511.04192)](https://arxiv.org/abs/2511.04192)
- [Sawdayee et al. 2025, Dance Like a Chicken / LoRA-MDM (arXiv 2503.19557)](https://arxiv.org/abs/2503.19557)
- [Amini & Desai 2026, MoSAIC (arXiv 2607.26304)](https://arxiv.org/abs/2607.26304)
- [STyMo (arXiv 2609.04500)](https://arxiv.org/html/2609.04500)
- [Roether et al. 2009, Critical features for the perception of emotion from gait (J. Vision)](https://doi.org/10.1167/9.6.15)
- [HumanML3D README (20 fps)](https://github.com/EricGuo5513/HumanML3D)
- Not verified (full text unreachable): [Larboulette & Gibet 2015, A Review of Computable Expressive Descriptors of Human Motion (MOCO)](https://hal.science/hal-01196267); Pullen & Bregler 2002 (cited only through Hsu 2005).
