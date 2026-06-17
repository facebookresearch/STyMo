# STyMo Command-Line & Application Reference

This document provides a comprehensive reference for the training script options, GUI application controls, and average pose extraction tool.

## Training Options

```bash
py .\src\stylizer_train.py <motion_path> <source_input> [OPTIONS]
```

**Required Arguments:**
- `motion_path`: Path to the style directory containing source/target folders (e.g., `.\motion\mocha\angry\`)
- `source_input`: Path to the BVH file(s) to use as inference input (this can be a `.bvh` file or a folder containing multiple of them)

**Key Options:**
- `--epochs`: Number of training epochs (default: 100)
- `--name`: Experiment name for saving results (default: "test")
- `--nn_confidence_th`: Threshold for gating model training. Set to a number (e.g., 2.5) to use that threshold multiplied by IQR for determining negative examples. Set to `interactive` to interactively select the threshold using a matplotlib visualization of the nearest-neighbor distance distribution (default: 2.5).
- `--feet_velocity_contact_th`: Threshold for feet velocity to determine foot contact (default: 0.25)
- `--scale`: Scale factor for input BVH units (1.0 for meters, 0.01 for cm, default: 1.0)
- `--target_frame_time`: Target framerate in seconds (default: 1/60 = 60fps)
- `--window_frames`: Prediction window size in frames (default: 32)
- `--fix_root`: Keep root joint fixed to source motion (only for inference)

### Output

Training saves results to `<motion_path>/results/Transformer-<name>/:`
- `model.pt`: Complete trained model with all parameters
- `tensorboard/`: Training logs for TensorBoard
- `input/`: Copy of input BVH files
- `source/` and `target/`: Copy of training data
- `sourceAvg/` and `targetAvg/`: Auto-extracted average poses per chunk

---

## Application Usage

### Interactive Controls

The GUI provides control over the stylization process:

**Dynamic Style (Encoder/Decoder)**
- **Encoder Output**: Scales the encoded output before it is used in the cross-attention module
- **Decoder Output**: Overall dynamic style intensity
- **Induce Action** (Collapsible, collapsed by default): Inject specific motion patterns
  - Click "Expand" to reveal controls (once a parameter is set, it is applied even when not expanded)
  - Index Start/End: Range of pre-learned actions to inject
  - Frequency: How often actions are triggered
  - Speed: Playback speed of injected actions
  - Strength: Blend amount of injected actions

**Static Style (Average)**
- **Output**: Overall static/posture style intensity

**Gating (Stylizability Gate)**
- **Dynamic Gate**: Controls how much the gating score affects the dynamic style (0 = ignore gating, 1 = full gating)
- **Static Gate**: Controls how much the gating score affects the static style (0 = ignore gating, 1 = full gating)
- When gating is enabled, out-of-distribution poses receive lower style intensity based on the gating model's prediction

**Advanced: Body Parts**
- Toggle "Body Parts" to control style per body region
- Separate sliders for: Arms, Legs, Spine, Head, Hands, etc.

**Postprocessing**
- **None/Fast/Normal/High**: Contact optimization quality

### Application Options

```bash
py .\src\stylizer_app.py <results_path> [OPTIONS]
```

**Arguments:**
- `results_path`: Path to the trained model results directory
- `--scale`: Scale factor for visualization (default: 1.0)
- `--inplace`: Render stylized output in-place rather than offset
- `--fix_root`: Fix root joint to source motion
- `--merge_output`: Process all BVH files in the input folder, merge predictions, and visualize the merged result (hides file selector)
- `--render_side`: Render skeletons twice, with a second version rotated 90 degrees to the side
- `--post_processing`: Default postprocessing quality - choices: none, fast, normal, high (default: fast)
- `--render_framerate`: Resample render to this framerate, e.g., 30.0 (default: None, uses original framerate)
- `--render_average`: Debug render the predicted average poses (shows the weighted blend of chunk average poses based on classifier probabilities)
- `--render_temp`: Render 2 additional delayed versions (0.5s and 1s) with colors blending towards white for temporal visualization

### Output

The application saves predictions to `<results_path>/app_results/`:
- `pred_<filename>.bvh`: Stylized prediction for each input file

---

## Extract Average Pose Tool

The `extract_average_pose.py` script extracts static pose pairs from input and prediction BVH files. These poses can be modified in Blender or any DCC tool before retraining, allowing fine-grained control over the static style component of the model.

### Usage

**Typical Usage (Interactive Mode):**
```bash
py .\src\extract_average_pose.py .\motion\mocha\clownangry\results\Transformer-newavg\ --interactive
```

Interactive mode provides:
1. File selection from available input BVH files
2. Single frame extraction (with automatic window selection)
3. Frame range extraction
4. Real-time feedback on extracted poses

**Command Line Mode:**
```bash
# Single frame (uses window from model's delta_indices)
py .\src\extract_average_pose.py <results_path> --frame 100

# Frame range
py .\src\extract_average_pose.py <results_path> --frame_start 50 --frame_end 150
```

### Options

- `results_path`: Path to the trained model results directory (must contain `input/` and `app_results/` folders)
- `--scale`: Scale factor for BVH files (default: 1.0)
- `--frame`: Single frame index to extract (uses window from delta_indices)
- `--frame_start` / `--frame_end`: Frame range for extraction
- `--interactive`: Run in interactive mode for file and frame selection
- `--output_index`: Manually specify output file index (auto-increments if not specified)

### Output

Creates source/target static pose pairs in the motion directory:
- `<motion_path>/sourceStatic/`: Averaged input poses
- `<motion_path>/targetStatic/`: Averaged prediction poses

Files are named with incrementing indices (0.bvh, 1.bvh, etc.). These BVH files can be opened and modified in Blender, Maya, or any other DCC tool to refine the static pose before retraining.
