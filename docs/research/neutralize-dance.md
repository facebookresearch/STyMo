# Research: options for neutralising ("smoothing out") dance style

_2026-11-17. Scope: ways to remove style from a dance clip so that it becomes a "neutral" version, aligned frame by frame with the original. Needed for Q7 in [`../thesis-plan.md`](../thesis-plan.md) (neutral as the bridge: hiphop → neutral → ballet)._

## Short answer

- I found no published method that targets "neutral dance" as its output. The literature splits motion into **content** (what is done) and **style** (how it is done). Neutralising is either "drop the style code" (learned) or "remove the frequency/posture components that carry style" (heuristic).
- The closest learned method is **VQ-Style** (arXiv 2602.02334, Feb 2026). From the full paper:
  - "Removal" is *content extraction*: decode with only the first (content) residual codebook and discard the later style layers, or interpolate the style codes to α = 0.
  - No reference clip is used. The paper does not formally define the neutral output, and admits its extracted motion does not exactly match the dataset's Neutral style.
  - Trained on 100STYLE and Aberman's data; no dance experiments; no code URL in the paper.
- **Correction (2026-11-17, see [`neutralize-movement.md`](neutralize-movement.md)):** a GitHub API listing of dataset-1 shows only `dance-long_normal_001` and `dance-short_normal_001`; there are no `dance` files in other styles. The claim below is wrong.
- **Data lead, checked on the dataset page:** the Bandai Namco dataset-1 has a `normal` (neutral) style among its 15 styles, and a `dance` content. `dance` is not in the list of contents that exist only in `normal`, so the same short dance appears to be performed in several styles *and* in normal: the only same-content, multi-style dance mocap with a neutral version found so far (BVH, 30 fps, CC BY-NC 4.0). `dance-long` and `dance-short` exist only in `normal`. Confirm by listing the files before relying on it.
- No public dataset I found has the same choreography danced in several styles plus a neutral version. That is why a heuristic neutraliser that **creates** aligned (neutral, styled) pairs is a real contribution, not only a baseline.

## Option families

### A. Heuristic neutralisers (baseline, no training)

Each heuristic targets one of the three things style is made of: posture, dynamics, timing. STyMo already splits style into static (posture) and dynamic parts, so these line up with its sliders.

| # | Heuristic | Removes | Literature anchor | Risk |
|---|---|---|---|---|
| A1 | Temporal low-pass filter on joint rotations (Gaussian / Savitzky–Golay; `savgol_filter` already used in `src/features.py`) | Sharp accents, hits, pops: hiphop's percussive quality | Yumer & Mitra 2016: style lives in the high-frequency spectral magnitude of the motion | Also blurs real choreography (fast footwork, jumps). Cutoff is a parameter to study. |
| A2 | Spectral magnitude replacement: per joint, sliding-window FFT; keep the phase (timing, content), replace the magnitude with that of a neutral reference | Same as A1, more targeted | Yumer & Mitra 2016 (spectral style transfer between *independent* actions, no correspondence needed) | Needs a neutral reference spectrum (e.g. AMASS everyday motion) |
| A3 | Pull posture towards a neutral pose: slerp each joint's rotation towards a reference pose by factor α | Posture: ballet turnout and extended lines, hiphop's low bent stance | Inverse of STyMo's static style (`src/extract_average_pose.py`) | Changes what the step is if α is large |
| A4 | Per-joint mean/variance normalisation over the clip (AdaIN-style on motion features) | Style statistics that are constant over time | Aberman et al. 2020 inject style exactly this way (temporally invariant AdaIN); removing it is the inverse | Simple, but blind to which joints carry style |
| A5 | Time normalisation: time-warp to an even beat, or flatten the speed profile within each movement | Dynamics and timing: suspensions, sudden stops | Classic DTW / time-warping work | Q1 allows timing to change, so this is in scope |
| A6 | Range-of-motion clamping and joint locking (less extension and turnout, quieter wrists, neck, spine) | Extremes typical of either style | Rule-based; can be phrased with Laban Effort qualities (Weight, Time, Flow, Space) | Hand-tuned per style |

The natural baseline is **A1 (or A2) + A3 + A5**: one heuristic per component of style. Each has one strength parameter, giving a dose–response curve to report.

### B. Learned neutralisers

| # | Approach | Data needed | Anchor |
|---|---|---|---|
| B1 | Content/style disentangling autoencoder; decode the content code with a "neutral" or zero style code | Unpaired clips with style labels; FineDance fits for hiphop | Aberman et al. 2020 ("Unpaired Motion Style Transfer from Video to Animation"); MoST (CVPR 2024, built for transfer between motions with *different* content, which dance needs); Hierarchical Style-based Networks (2020) |
| B2 | Residual-quantised codes: coarse levels = content, fine levels = style; neutralise by swapping or dropping fine codes | Labelled clips | VQ-Style (2026), lists style removal explicitly |
| B3 | Diffusion editing: invert with DDIM, regenerate without style conditioning (or with "neutral" / plain-text conditioning) | A pretrained motion diffusion model | SMooDi (ECCV 2024) does transfer through DDIM inversion. SMooDi and motion-diffusion-model are already in `~/projects/`, so this may be cheap to try. |
| B4 | Optimisation with a style loss: minimise a style classifier's confidence (towards uniform or "neutral") + content terms (root path, foot contacts, beat timing) + naturalness | A style classifier only | This is the "simple but sophisticated loss" idea; no paired data needed |
| B5 | Bootstrap: heuristic (A) makes aligned (neutral, styled) pairs → train a STyMo-like model neutral → style, and a neutraliser style → neutral | Output of A | The "baseline makes the data" idea from the first planning message |

B5 and B4 match the plan best: B5 uses STyMo as is, and B4 is a small, explainable contribution.

## Data that could define "neutral"

- **AMASS** (everyday motion): non-dance reference for A2 and A3, and the "neutral" class for B1 and B4. Hagar to supply the path.
- **Bandai Namco Research motion dataset** (arXiv 2306.08861, GitHub `BandaiNamcoResearchInc/Bandai-Namco-Research-Motiondataset`):
  - Dataset-1: 15 styles: normal, happy, sad, angry, proud, not-confident, masculinity, feminine, children, old, tired, active, musical, giant, chimpira. 175 items, 36,673 frames, 30 fps, BVH, CC BY-NC 4.0. Skeleton joint names are not stated on the page.
  - Dance contents: `dance` (several styles, apparently incl. normal), `dance-long` and `dance-short` (normal only). The normal-only long dances are ready-made "neutral dance" examples; `dance` gives aligned-content normal ↔ style pairs, the format STyMo trains on.
  - Its styles are emotions and characters, not dance genres. It defines what "neutral" looks like for a dance, not what ballet or hiphop look like.
- **STyMo's MoCha data**: has a neutral class, but no dance in other styles, and is on a different skeleton (see Q5).
- No dataset found with the same choreography in several dance styles. The nearest: Dancer101 (101 people, same routine, pairs identity rather than style); a professional/amateur paired set built from AIST++.

## How to evaluate a neutraliser

- **Style removed:** a hiphop/ballet/neutral classifier's accuracy on outputs drops to chance, or its "neutral" probability rises.
- **Content kept:** root trajectory error, foot-contact agreement, beat alignment (FineDance has music), key-pose distance.
- **Naturalness:** foot sliding, jerk, FID-style distance to real motion.
- **Dose–response:** sweep each heuristic's strength and plot style-removed against content-kept. A curve is a stronger thesis result than a single point.

## Open questions for Hagar

1. What does "neutral" mean operationally? Posture only, dynamics only, or timing too? This picks which A-heuristics make up the baseline.
2. Is neutral "plain everyday motion" (AMASS-like) or "the average of all dance styles"? Each gives a different reference for A2, A3 and B4.
3. Should we download Bandai Namco dataset-1 (non-commercial licence) to check the `dance` files and use `normal` as the neutral reference? A third skeleton would need converting like Q5.

## Sources

- [VQ-Style: Disentangling Style and Content in Motion with Residual Quantized Representations (arXiv 2602.02334)](https://arxiv.org/abs/2602.02334), [full text](https://arxiv.org/html/2602.02334)
- [Bandai Namco dataset-1 page (styles, contents)](https://github.com/BandaiNamcoResearchInc/Bandai-Namco-Research-Motiondataset/tree/master/dataset/Bandai-Namco-Research-Motiondataset-1)
- [Aberman et al. 2020, Unpaired Motion Style Transfer from Video to Animation (arXiv 2005.05751)](https://arxiv.org/pdf/2005.05751)
- [MoST: Motion Style Transformer between Diverse Action Contents (CVPR 2024, arXiv 2403.06225)](https://arxiv.org/html/2403.06225v2)
- [Hierarchical Style-based Networks for Motion Synthesis (arXiv 2008.10162)](https://arxiv.org/pdf/2008.10162)
- [Yumer & Mitra 2016, Spectral Style Transfer for Human Motion between Independent Actions](https://geometry.cs.ucl.ac.uk/projects/2016/spectral-style-transfer)
- [SMooDi: Stylized Motion Diffusion Model (ECCV 2024, arXiv 2407.12783)](https://arxiv.org/abs/2407.12783)
- [Bandai Namco motion dataset paper (arXiv 2306.08861)](https://arxiv.org/pdf/2306.08861) and [repository README](https://github.com/BandaiNamcoResearchInc/Bandai-Namco-Research-Motiondataset/blob/master/README.md)
- [LMA features for emotion in dance (SCA 2017)](https://www.ipab.inf.ed.ac.uk/cgvu/cga2017.pdf)
- [Identity-preserved human motion transfer / Dancer101 (arXiv 2204.06862)](https://arxiv.org/pdf/2204.06862)
- [Professional/amateur paired dance dataset (CVM)](https://cg.cs.tsinghua.edu.cn/cvmj/papers/CVM0292.pdf)
- [FineDance (ICCV 2023)](https://openaccess.thecvf.com/content/ICCV2023/papers/Li_FineDance_A_Fine-grained_Choreography_Dataset_for_3D_Full_Body_Dance_ICCV_2023_paper.pdf)
