# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Extract Average Pose Script

This script allows users to select frames from input and prediction BVH files
and compute averaged poses to save as source/target pairs for the static pose model.

The script creates sourceStatic/ and targetStatic/ folders two levels up from the results directory
(i.e., in the motion_path folder) containing the averaged poses from the selected frame ranges.
"""

import os
import argparse
import warnings
import torch

import numpy as np
import pymotion.rotations.quat as quat

from pymotion.io.bvh import BVH
from utils import resample_bvh


def load_bvh_file(file_path, scale, target_frame_time):
    """Load and preprocess a BVH file."""
    bvh = BVH()
    bvh.load(file_path)
    bvh.set_scale(scale)
    bvh = resample_bvh(bvh, target_frame_time)
    return bvh


def compute_average_pose(bvh, frame_start, frame_end):
    """
    Compute the average pose from a range of frames.

    Args:
        bvh: BVH object
        frame_start: Start frame index (inclusive)
        frame_end: End frame index (inclusive)

    Returns:
        avg_rotations: Averaged rotations [num_joints, 4]
        avg_positions: Averaged positions [num_joints, 3]
    """
    local_rotations, local_positions, _, _, _, _ = bvh.get_data()

    # Extract frame range
    rotations_range = local_rotations[frame_start : frame_end + 1]
    positions_range = local_positions[frame_start : frame_end + 1]

    # Average rotations using normalized mean (works well for small variations)
    # For quaternions, we use normalized mean which is a simple approximation
    # that works when quaternions are close to each other
    avg_rotations = quat.normalize(np.mean(rotations_range, axis=0))

    # Average positions (simple mean)
    avg_positions = np.mean(positions_range, axis=0)

    return avg_rotations, avg_positions


def create_single_frame_bvh(source_bvh, rotations, positions):
    """
    Create a BVH with a single frame from given rotations and positions.

    Args:
        source_bvh: Source BVH to copy structure from
        rotations: Joint rotations [num_joints, 4]
        positions: Joint positions [num_joints, 3]

    Returns:
        BVH object with single frame
    """
    new_bvh = BVH()
    new_bvh.copy(source_bvh)

    # Expand to single frame shape
    rotations = rotations[np.newaxis, ...]  # [1, num_joints, 4]
    positions = positions[np.newaxis, ...]  # [1, num_joints, 3]

    new_bvh.set_data(rotations, positions)
    return new_bvh


def get_frame_range_from_window(center_frame, delta_indices, num_frames):
    """
    Get the frame range based on a center frame and delta_indices (window).

    Args:
        center_frame: Center frame index
        delta_indices: List of delta indices defining the window
        num_frames: Total number of frames in the BVH

    Returns:
        frame_start: Start frame (clamped)
        frame_end: End frame (clamped)
    """
    min_delta = min(delta_indices)
    max_delta = max(delta_indices)

    frame_start = max(0, center_frame + min_delta)
    frame_end = min(num_frames - 1, center_frame + max_delta)

    return frame_start, frame_end


def extract_average_poses(
    results_path,
    scale,
    frame_selection,
    use_window,
    output_index=None,
    selected_file=None,
):
    """
    Main function to extract average poses from input and prediction BVH files.

    Args:
        results_path: Path to the results directory (e.g., motion/eval/angry/results/Transformer-base)
        scale: Scale factor for BVH files
        frame_selection: Either a single frame index, or a tuple (start, end) for a range
        use_window: If True, use delta_indices window around selected frame
        output_index: Optional index for the output file name (auto-increments if None)
        selected_file: Optional filename to process only a specific file (None = all files)
    """

    # Load model to get delta_indices and target_frame_time
    model_path = os.path.join(results_path, "model.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    saving_dict = torch.load(model_path, map_location="cpu", weights_only=False)
    delta_indices = saving_dict["delta_indices"]
    target_frame_time = saving_dict["target_frame_time"]

    print(
        f"Model loaded - window: [{min(delta_indices)}, {max(delta_indices)}], frame_time: {target_frame_time:.4f}"
    )

    # Find input and prediction folders
    input_folder = os.path.join(results_path, "input")
    app_results_folder = os.path.join(results_path, "app_results")

    if not os.path.exists(input_folder):
        raise FileNotFoundError(f"Input folder not found at {input_folder}")
    if not os.path.exists(app_results_folder):
        raise FileNotFoundError(
            f"App results folder not found at {app_results_folder}. "
            "Please run stylizer_app.py first to generate predictions."
        )

    # Create output directories (3 parent folders up from results_path)
    # results_path is typically: motion_path/results/Transformer-{name}
    # We want: motion_path/sourceStatic and motion_path/targetStatic
    parent_path = os.path.dirname(os.path.dirname(os.path.dirname(results_path)))
    source_static_dir = os.path.join(parent_path, "sourceStatic")
    target_static_dir = os.path.join(parent_path, "targetStatic")
    os.makedirs(source_static_dir, exist_ok=True)
    os.makedirs(target_static_dir, exist_ok=True)

    # Determine output index
    if output_index is None:
        existing_files = [
            f for f in os.listdir(source_static_dir) if f.endswith(".bvh")
        ]
        if existing_files:
            existing_indices = [int(os.path.splitext(f)[0]) for f in existing_files]
            output_index = max(existing_indices) + 1
        else:
            output_index = 0

    # Get list of input BVH files
    input_bvh_files = [f for f in os.listdir(input_folder) if f.endswith(".bvh")]
    if not input_bvh_files:
        raise ValueError(f"No BVH files found in {input_folder}")

    # Filter to selected file if specified
    if selected_file is not None:
        if selected_file not in input_bvh_files:
            raise ValueError(f"File {selected_file} not found in {input_folder}")
        input_bvh_files = [selected_file]

    # Process each input/prediction pair
    for input_filename in input_bvh_files:
        input_path = os.path.join(input_folder, input_filename)

        # Find corresponding prediction file
        # Prediction files are named pred_{original_name}.bvh
        base_name = os.path.splitext(input_filename)[0]
        pred_filename = f"pred_{base_name}.bvh"
        pred_path = os.path.join(app_results_folder, pred_filename)

        if not os.path.exists(pred_path):
            print(
                f"Warning: Prediction file not found for {input_filename}, skipping..."
            )
            continue

        # Load BVH files
        source_bvh = load_bvh_file(input_path, scale, target_frame_time)
        target_bvh = load_bvh_file(pred_path, scale, target_frame_time)
        num_frames = source_bvh.data["rotations"].shape[0]

        # Determine frame range
        if isinstance(frame_selection, tuple):
            frame_start, frame_end = frame_selection
        else:
            # Single frame selected
            center_frame = frame_selection
            if use_window:
                frame_start, frame_end = get_frame_range_from_window(
                    center_frame, delta_indices, num_frames
                )
            else:
                frame_start = center_frame
                frame_end = center_frame

        # Clamp to valid range
        frame_start = max(0, min(frame_start, num_frames - 1))
        frame_end = max(0, min(frame_end, num_frames - 1))

        # Compute average poses
        source_avg_rots, source_avg_pos = compute_average_pose(
            source_bvh, frame_start, frame_end
        )
        target_avg_rots, target_avg_pos = compute_average_pose(
            target_bvh, frame_start, frame_end
        )

        # Create single-frame BVH files
        source_static_bvh = create_single_frame_bvh(
            source_bvh, source_avg_rots, source_avg_pos
        )
        target_static_bvh = create_single_frame_bvh(
            target_bvh, target_avg_rots, target_avg_pos
        )

        # Normalize root to origin for static poses
        source_static_rots, source_static_pos, _, _, _, _ = source_static_bvh.get_data()
        target_static_rots, target_static_pos, _, _, _, _ = target_static_bvh.get_data()

        # Set root to identity rotation and zero position
        source_static_rots[0, 0] = np.array([1.0, 0.0, 0.0, 0.0])
        source_static_pos[0, 0] = np.array([0.0, 0.0, 0.0])
        target_static_rots[0, 0] = np.array([1.0, 0.0, 0.0, 0.0])
        target_static_pos[0, 0] = np.array([0.0, 0.0, 0.0])

        source_static_bvh.set_data(source_static_rots, source_static_pos)
        target_static_bvh.set_data(target_static_rots, target_static_pos)

        # Save BVH files
        output_name = f"{output_index}.bvh"
        source_static_bvh.save(os.path.join(source_static_dir, output_name))
        target_static_bvh.save(os.path.join(target_static_dir, output_name))

        print(
            f"Saved {input_filename} frames [{frame_start}-{frame_end}] -> {output_name}"
        )

        output_index += 1

    print(f"\n\033[92mExtraction complete!\033[0m")
    print(f"  sourceStatic: {source_static_dir}")
    print(f"  targetStatic: {target_static_dir}")


def interactive_mode(results_path, scale):
    """
    Interactive mode for selecting files and frames.
    """

    # Load model to get delta_indices and target_frame_time
    model_path = os.path.join(results_path, "model.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    saving_dict = torch.load(model_path, map_location="cpu", weights_only=False)
    delta_indices = saving_dict["delta_indices"]
    target_frame_time = saving_dict["target_frame_time"]

    # Find input folder
    input_folder = os.path.join(results_path, "input")
    input_bvh_files = sorted(
        [f for f in os.listdir(input_folder) if f.endswith(".bvh")]
    )

    if not input_bvh_files:
        raise ValueError(f"No BVH files found in {input_folder}")

    print("\n=== Interactive Frame Selection ===")
    print(f"Window size (delta_indices): {len(delta_indices)} frames")
    print(f"Window range: [{min(delta_indices)}, {max(delta_indices)}]")

    # File selection state
    selected_file = None
    num_frames = None

    while True:
        print("\n" + "=" * 50)
        if selected_file:
            print(f"Selected file: {selected_file} ({num_frames} frames)")
        else:
            print("No file selected")
        print("=" * 50)

        print("\nOptions:")
        print("  1. Select/change file")
        print("  2. Select single frame (with window)")
        print("  3. Select frame range")
        print("  4. Exit")

        choice = input("\nEnter choice (1-4): ").strip()

        if choice == "1":
            # File selection
            print("\nAvailable files:")
            for i, filename in enumerate(input_bvh_files):
                # Load file to show frame count
                bvh_path = os.path.join(input_folder, filename)
                bvh = load_bvh_file(bvh_path, scale, target_frame_time)
                file_frames = bvh.data["rotations"].shape[0]
                marker = " *" if filename == selected_file else ""
                print(f"  {i + 1}. {filename} ({file_frames} frames){marker}")

            try:
                file_choice = input(
                    f"\nEnter file number (1-{len(input_bvh_files)}): "
                ).strip()
                file_idx = int(file_choice) - 1
                if 0 <= file_idx < len(input_bvh_files):
                    selected_file = input_bvh_files[file_idx]
                    # Load selected file to get frame count
                    bvh_path = os.path.join(input_folder, selected_file)
                    bvh = load_bvh_file(bvh_path, scale, target_frame_time)
                    num_frames = bvh.data["rotations"].shape[0]
                    print(f"\nSelected: {selected_file} ({num_frames} frames)")
                else:
                    print("Invalid file number")
            except ValueError:
                print("Invalid input. Please enter a number.")

        elif choice == "2":
            # Single frame with window
            if selected_file is None:
                print("\nPlease select a file first (option 1)")
                continue

            try:
                frame = int(input(f"Enter frame index (0-{num_frames-1}): ").strip())
                if 0 <= frame < num_frames:
                    extract_average_poses(
                        results_path,
                        scale,
                        frame,
                        use_window=True,
                        selected_file=selected_file,
                    )
                else:
                    print(f"Frame must be between 0 and {num_frames-1}")
            except ValueError:
                print("Invalid input. Please enter a number.")

        elif choice == "3":
            # Frame range
            if selected_file is None:
                print("\nPlease select a file first (option 1)")
                continue

            try:
                start = int(input(f"Enter start frame (0-{num_frames-1}): ").strip())
                end = int(input(f"Enter end frame ({start}-{num_frames-1}): ").strip())
                if 0 <= start <= end < num_frames:
                    extract_average_poses(
                        results_path,
                        scale,
                        (start, end),
                        use_window=False,
                        selected_file=selected_file,
                    )
                else:
                    print("Invalid frame range")
            except ValueError:
                print("Invalid input. Please enter numbers.")

        elif choice == "4":
            print("Exiting...")
            break

        else:
            print("Invalid choice. Please enter 1-4.")


if __name__ == "__main__":
    warnings.filterwarnings(
        "ignore", category=FutureWarning, message=".*torch.load.*weights_only.*"
    )

    parser = argparse.ArgumentParser(
        description="Extract average poses from input and prediction BVH files."
    )
    parser.add_argument(
        "results_path",
        type=str,
        help="Path to the results directory (must contain 'input' and 'app_results' folders)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Scale factor for BVH files (default: 1.0)",
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=None,
        help="Single frame index to extract. Uses window from delta_indices.",
    )
    parser.add_argument(
        "--frame_start",
        type=int,
        default=None,
        help="Start frame index for range extraction.",
    )
    parser.add_argument(
        "--frame_end",
        type=int,
        default=None,
        help="End frame index for range extraction.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode for frame selection.",
    )
    parser.add_argument(
        "--output_index",
        type=int,
        default=None,
        help="Output file index (auto-increments if not specified).",
    )

    args = parser.parse_args()

    if args.interactive:
        interactive_mode(args.results_path, args.scale)
    elif args.frame is not None:
        # Single frame with window
        extract_average_poses(
            args.results_path,
            args.scale,
            args.frame,
            use_window=True,
            output_index=args.output_index,
        )
    elif args.frame_start is not None and args.frame_end is not None:
        # Frame range
        extract_average_poses(
            args.results_path,
            args.scale,
            (args.frame_start, args.frame_end),
            use_window=False,
            output_index=args.output_index,
        )
    else:
        print(
            "Error: Please specify --frame, --frame_start/--frame_end, or --interactive"
        )
        print("\nExamples:")
        print("  python extract_average_pose.py /path/to/results --frame 100")
        print(
            "  python extract_average_pose.py /path/to/results --frame_start 50 --frame_end 150"
        )
        print("  python extract_average_pose.py /path/to/results --interactive")
        parser.print_help()
