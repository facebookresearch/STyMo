# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
import pymotion

import numpy as np
import matplotlib.pyplot as plt
import pymotion.rotations.ortho6d as ortho6d

from sklearn.neighbors import NearestNeighbors
from standardize import Standardize
from features import (
    get_body_parts_indices,
    get_source_kinematics_full_body,
    get_full_body_target_kinematics,
    get_feet_indices,
    get_chunks,
)
from utils import get_bvh_data, resample_bvh
from utils_inference import get_global_positions
from pymotion.io.bvh import BVH
from features import get_continuous_delta_rotations


def read_bvh_files(motion_path, source_input, scale, target_frame_time):
    # Read .bvh files
    (
        source_bvhs,
        train_source_bvhs,
        train_target_bvhs,
        source_bvh_paths,
        _,
        _,
        _,
        _,
        _,
        end_joints,
    ) = get_bvh_data(
        motion_path,
        source_input,
        scale,
        target_frame_time,
        extra_end_joints_indices=[0],
    )
    body_parts_indices = get_body_parts_indices(source_bvhs[0])
    feet_indices = get_feet_indices(source_bvhs[0])
    parents = source_bvhs[0].data["parents"]
    return (
        source_bvhs,
        train_source_bvhs,
        train_target_bvhs,
        source_bvh_paths,
        end_joints,
        body_parts_indices,
        feet_indices,
        parents,
    )


def preprocess_paired_data(
    train_source_bvhs,
    train_target_bvhs,
    delta_indices,
    feet_velocity_contact_th,
    batch_size,
    device,
):
    x_k = []  # source kinematics
    x_k_contacts = []  # source kinematics contact information
    x_p = []  # previous target kinematics
    x_p_no_aug = []  # x_p without data augmentation
    Y_k = []  # target kinematics
    Y_s_pos = []  # source positions
    Y_s_rots = []  # source rotations
    average_delta_pelvis_positions = []
    average_delta_rotations = []
    average_pelvis_positions = []
    average_rotations = []

    for i in range(len(train_source_bvhs)):
        train_source_bvh = train_source_bvhs[i]
        train_target_bvh = train_target_bvhs[i]

        # Data augmentation (temporal dimension)
        factors = np.arange(0.7, 1.5, 0.1)
        for factor in factors:
            train_source_bvh_aug = BVH()
            train_source_bvh_aug.copy(train_source_bvh)
            frame_time = train_source_bvh.data["frame_time"]
            train_source_bvh_aug = resample_bvh(
                train_source_bvh_aug, frame_time * factor
            )
            train_target_bvh_aug = BVH()
            train_target_bvh_aug.copy(train_target_bvh)
            train_target_bvh_aug = resample_bvh(
                train_target_bvh_aug, frame_time * factor
            )

            num_frames = train_source_bvh_aug.data["rotations"].shape[0]
            # TODO: explore with different chunk_size, but for now what makes more sense is one
            # chunk per paired animation (because it usually has a meaning, like 1st animation
            # is walking, 2nd is running, etc.)
            chunk_start, chunk_end = get_chunks(
                chunk_size=num_frames, num_frames=num_frames
            )

            if abs(factor - 1.0) < 0.0001:
                print(f"Paired training: {train_source_bvh_aug.data['filename']} ({num_frames} frames)")

            x_k_i, x_k_contacts_i = get_source_kinematics_full_body(
                train_source_bvh_aug,
                delta_indices,
                feet_velocity_contact_th,
            )
            x_k.append(x_k_i)
            x_k_contacts.append(x_k_contacts_i)
            (
                Y_k_i,
                average_delta_rotations_i,
                average_delta_pelvis_positions_i,
                Y_s_pos_i,
                Y_s_rots_i,
                average_rotations_i,
                average_pelvis_positions_i,
            ) = get_full_body_target_kinematics(
                train_source_bvh_aug,
                train_target_bvh_aug,
                delta_indices,
                chunk_start,
                chunk_end,
            )
            Y_k.append(Y_k_i)
            Y_s_pos.append(Y_s_pos_i)
            Y_s_rots.append(Y_s_rots_i)
            x_p_i = np.concatenate(
                [np.zeros_like(Y_k_i[0:1, ...]), Y_k_i[:-1, ...]], axis=0
            )
            x_p.append(x_p_i)
            if abs(factor - 1.0) < 0.0001:
                x_p_no_aug.append(x_p_i)
                average_delta_rotations.append(average_delta_rotations_i)
                average_delta_pelvis_positions.append(average_delta_pelvis_positions_i)
                average_rotations.append(average_rotations_i)
                average_pelvis_positions.append(average_pelvis_positions_i)

    average_delta_rotations = np.concatenate(average_delta_rotations, axis=0)
    average_delta_pelvis_positions = np.concatenate(
        average_delta_pelvis_positions, axis=0
    )
    average_rotations = np.concatenate(average_rotations, axis=0)
    average_pelvis_positions = np.concatenate(average_pelvis_positions, axis=0)

    x_k = np.concatenate(x_k, axis=0)
    x_k_standardize = Standardize().fit(x_k)
    x_k = x_k_standardize.standardize(x_k)

    x_k_contacts = np.concatenate(x_k_contacts, axis=0)

    x_p = np.concatenate(x_p, axis=0)
    x_p_no_aug = np.concatenate(x_p_no_aug, axis=0)
    x_p_standardize = Standardize().fit(x_p)
    x_p = x_p_standardize.standardize(x_p)
    x_p_no_aug = x_p_standardize.standardize(x_p_no_aug)

    Y_k = np.concatenate(Y_k, axis=0)
    Y_s_pos = np.concatenate(Y_s_pos, axis=0)
    Y_s_rots = np.concatenate(Y_s_rots, axis=0)

    Y_k_standardize = Standardize().fit(Y_k)
    Y_k = Y_k_standardize.standardize(Y_k)

    x_k = torch.from_numpy(x_k).float().to(device)  # (N, window, features in)
    x_p = torch.from_numpy(x_p).float().to(device)  # (N, window, features out)
    x_p_no_aug = (
        torch.from_numpy(x_p_no_aug).float().to(device)
    )  # (N, window, features out)
    Y_k = torch.from_numpy(Y_k).float().to(device)  # (N, window, features out)
    Y_s_pos = torch.from_numpy(Y_s_pos).float().to(device)  # (N, window, joints, 3)
    Y_s_rots = torch.from_numpy(Y_s_rots).float().to(device)  # (N, window, joints, 4)
    x_k_contacts = torch.from_numpy(x_k_contacts).float().to(device)  # (N, window, 2)
    average_delta_rotations = (
        torch.from_numpy(average_delta_rotations).float().to(device)
    )  # (num_chunks, joints-1, 4)
    average_delta_pelvis_positions = (
        torch.from_numpy(average_delta_pelvis_positions).float().to(device)
    )  # (num_chunks, 3)

    paired_data = {
        "x_k": x_k,
        "x_p": x_p,
        "x_p_no_aug": x_p_no_aug,
        "Y_k": Y_k,
        "Y_s_pos": Y_s_pos,
        "Y_s_rots": Y_s_rots,
        "x_k_contacts": x_k_contacts,
    }

    # Create datasets
    # Paired dataset (stylized motion)
    dataset = torch.utils.data.TensorDataset(
        x_k, x_p, Y_k, Y_s_pos, Y_s_rots, x_k_contacts
    )

    # Split paired dataset into train and validation sets
    dataset_size = len(dataset)
    train_size = int(0.9 * dataset_size)
    val_size = dataset_size - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    # Create dataloaders
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )

    return (
        paired_data,
        train_dataloader,
        val_dataloader,
        x_k_standardize,
        x_p_standardize,
        Y_k_standardize,
        average_delta_rotations,  # [num_chunks, num_joints-1, 4]
        average_delta_pelvis_positions,  # [num_chunks, 3]
        average_rotations,  # [num_chunks, num_joints, 4]
        average_pelvis_positions,  # [num_chunks, 3]
    )


def create_average_samples(
    average_rotations,
    batch_size,
    device,
):
    """
    Create pairs of average inputs and corresponding chunk labels.

    Args:
        average_rotations: Per-chunk average rotations [num_chunks, num_joints, 4]
        batch_size: Batch size for dataloader
        device: Device to place tensors on

    Returns:
        dataloader: DataLoader with (rotations_6d, chunk_one_hot) tuples
    """
    # Convert quaternions to ortho6d all at once
    # average_rotations: [num_chunks, num_joints, 4] -> [num_chunks, num_joints-1, 6]
    average_rotations_6d = ortho6d.from_quat(
        torch.from_numpy(average_rotations[:, 1:]).to(device)
    )

    # Create one-hot vectors for chunk indices
    num_chunks = average_rotations_6d.shape[0]
    chunk_labels = torch.arange(num_chunks, device=device)
    chunk_one_hot = torch.nn.functional.one_hot(
        chunk_labels, num_classes=num_chunks
    ).float()

    # Create dataset and dataloader
    dataset = torch.utils.data.TensorDataset(
        average_rotations_6d,
        chunk_one_hot,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )

    return dataloader


def preprocess_source_input_data(
    source_bvhs,
    x_k_standardize,
    x_p_standardize,
    Y_k_standardize,
    delta_indices,
    feet_velocity_contact_th,
    parents,
    device,
    feet_indices,
    debug_visualize=False,
):
    input_sources = {}

    for source_bvh in source_bvhs:
        num_frames = source_bvh.data["rotations"].shape[0]

        print(f"Source input: {source_bvh.data['filename']} ({num_frames} frames)")

        x_k_source, x_k_contacts_source = get_source_kinematics_full_body(
            source_bvh, delta_indices, feet_velocity_contact_th
        )
        # TODO: this could potentially be replaced by identity operations (like 0 for vectors and 1,0,0,0 for quaternions)
        Y_k_source, _, _, Y_s_pos_source, Y_s_rots_source, _, _ = (
            get_full_body_target_kinematics(
                source_bvh,
                source_bvh,
                delta_indices,
                chunk_start=np.array([0]),
                chunk_end=np.array([num_frames]),
            )
        )

        if debug_visualize:
            Y_k_source_pos = get_global_positions(
                Y_k_source,
                None,
                None,
                source_local_positions=Y_s_pos_source,
                source_local_rotations=Y_s_rots_source,
                parents=parents,
            )

        x_p_source = np.concatenate(
            [np.zeros_like(Y_k_source[0:1, ...]), Y_k_source[:-1, ...]], axis=0
        )
        x_k_source = x_k_standardize.standardize(x_k_source)
        x_p_source = x_p_standardize.standardize(x_p_source)
        Y_k_source = Y_k_standardize.standardize(Y_k_source)

        if debug_visualize:
            with pymotion.render.blender.BlenderConnection() as conn:
                for Y_k_pos_i, x_k_contacts_i in zip(
                    Y_k_source_pos, x_k_contacts_source
                ):
                    conn.clear_scene()
                    conn.setup_rendering()
                    conn.render_checkerboard_floor()
                    Y_k_pos_i = Y_k_pos_i[..., [0, 2, 1]]
                    viz_contacts = np.where(
                        x_k_contacts_i[..., None],
                        Y_k_pos_i[:, feet_indices],
                        np.array([[[1000.0, 1000.0, 1000.0]]]),
                    )
                    conn.render_points_timeline(
                        Y_k_pos_i, np.array([1.0, 0.0, 0.0]), np.array([0.025])
                    )
                    conn.render_points_timeline(
                        viz_contacts, np.array([0.0, 1.0, 0.0]), np.array([0.035])
                    )
                    input("Press enter to continue...")

        input_sources[source_bvh.data["filename"]] = {
            "x_k": torch.from_numpy(x_k_source).float().to(device),
            "x_p": torch.from_numpy(x_p_source).float().to(device),
            "Y_k": torch.from_numpy(Y_k_source).float().to(device),
            "Y_s_pos": torch.from_numpy(Y_s_pos_source).float().to(device),
            "Y_s_rots": torch.from_numpy(Y_s_rots_source).float().to(device),
            "x_k_contacts": torch.from_numpy(x_k_contacts_source).float().to(device),
        }

    x_k = []
    x_p = []
    Y_k = []
    Y_s_pos = []
    Y_s_rots = []
    x_k_contacts = []
    for _, source_data in input_sources.items():
        x_k.append(source_data["x_k"])
        x_p.append(source_data["x_p"])
        Y_k.append(source_data["Y_k"])
        Y_s_pos.append(source_data["Y_s_pos"])
        Y_s_rots.append(source_data["Y_s_rots"])
        x_k_contacts.append(source_data["x_k_contacts"])
    input_sources_all = {
        "x_k": torch.cat(x_k, dim=0),
        "x_p": torch.cat(x_p, dim=0),
        "Y_k": torch.cat(Y_k, dim=0),
        "Y_s_pos": torch.cat(Y_s_pos, dim=0),
        "Y_s_rots": torch.cat(Y_s_rots, dim=0),
        "x_k_contacts": torch.cat(x_k_contacts, dim=0),
    }

    return input_sources, input_sources_all


def create_gating_samples(
    paired_data,
    input_sources_all,
    x_k,
    batch_size,
    nn_confidence_th,
    device,
):
    """
    Create training data for the gating model.

    Positive examples (label=1): Poses from source input that are CLOSE to paired data (below Q1)
    Negative examples (label=0): Poses from source input that are FAR from paired data

    Uses IQR-based threshold for negative examples:
    - Close (positive): distance <= Q1
    - Far (negative): distance > q3 + nn_confidence_th * IQR
        Distance distribution:
        |---------|---------|-----|----|
        0        q1        q3  far_th  ∞
            ↓         ↓        ↓       ↓
        POSITIVE       excluded    NEGATIVE
        (label=1)                  (label=0)

    This avoids training on ambiguous "middle zone" examples and prevents overfitting.

    Uses averaged rotations over the window (like the static model) to remove high-frequency
    details and facilitate training.

    Args:
        paired_data: Dictionary containing paired training data (from preprocess_paired_data)
        input_sources_all: Dictionary containing all source input data
        x_k: Source kinematics from paired data [N, window, features] (for NN search)
        batch_size: Batch size for dataloader
        nn_confidence_th: Threshold multiplier for IQR-based far classification.
                          Can be a float or "interactive" for interactive mode.
        device: Device to place tensors on

    Returns:
        gating_dataloader: DataLoader with (rotations_6d, labels) tuples
        or None if insufficient samples found
    """

    # Get source rotations from input sources (all poses)
    Y_s_rots_source = input_sources_all["Y_s_rots"]  # [M, window, num_joints, 4]
    # Compute average rotations over window and exclude root joint
    source_rots = Y_s_rots_source.mean(dim=1)[:, 1:, :]  # [M, num_joints-1, 4]
    source_rots_6d = ortho6d.from_quat(source_rots)  # [M, num_joints-1, 6]

    # Find close/far samples using nearest neighbor distance
    x_k_cpu = x_k.cpu().numpy() if torch.is_tensor(x_k) else x_k
    x_k_source = input_sources_all["x_k"].cpu().numpy()

    x_k_nn_db = x_k_cpu[:, x_k_cpu.shape[1] // 2, :]
    x_k_source_query = x_k_source[:, x_k_source.shape[1] // 2, :]
    x_k_nn = NearestNeighbors(n_neighbors=1).fit(x_k_nn_db)
    x_k_source_distances, _ = x_k_nn.kneighbors(x_k_source_query, return_distance=True)

    # Find thresholds based on IQR
    q1, q3 = np.percentile(x_k_source_distances, [25, 75])
    iqr = q3 - q1

    # Handle interactive mode
    if nn_confidence_th == "interactive":
        nn_confidence_th = _interactive_threshold_selection(
            x_k_source_distances, q1, q3, iqr
        )

    # Threshold for close (positive): all examples below Q1
    close_threshold = q1

    # Threshold for far (negative): above q3 + nn_confidence_th * IQR
    far_threshold = q3 + nn_confidence_th * iqr

    # Split source poses into close (positive) and far (negative)
    close_indices = np.where(x_k_source_distances.flatten() <= close_threshold)[0]
    far_indices = np.where(x_k_source_distances.flatten() > far_threshold)[0]

    print(f"Gating thresholds - Close: {close_threshold:.4f}, Far: {far_threshold:.4f}")
    print(f"Gating samples - Close (positive): {len(close_indices)}, Far (negative): {len(far_indices)}, "
          f"Middle (excluded): {len(x_k_source_distances) - len(close_indices) - len(far_indices)}")

    if len(far_indices) == 0:
        print("\033[93mNo negative samples found for gating model\033[0m")
        return None

    if len(close_indices) == 0:
        print("\033[93mNo positive samples found for gating model\033[0m")
        return None

    # Create positive samples from close source poses (label = 1)
    positive_rots_6d = source_rots_6d[close_indices]
    num_positive = positive_rots_6d.shape[0]
    positive_labels = torch.ones(num_positive, 1, device=device)

    # Create negative samples from far source poses (label = 0)
    negative_rots_6d = source_rots_6d[far_indices]
    num_negative = negative_rots_6d.shape[0]
    negative_labels = torch.zeros(num_negative, 1, device=device)

    # Combine positive and negative samples
    all_rots_6d = torch.cat([positive_rots_6d, negative_rots_6d], dim=0)
    all_labels = torch.cat([positive_labels, negative_labels], dim=0)

    print(f"Gating training data: {num_positive} positive + {num_negative} negative = {all_rots_6d.shape[0]} total")

    # Create dataset and dataloader
    dataset = torch.utils.data.TensorDataset(all_rots_6d, all_labels)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )

    return dataloader


def _interactive_threshold_selection(x_k_source_distances, q1, q3, iqr):
    """
    Interactive mode for selecting the nn_confidence_th value.
    Shows a matplotlib plot and asks the user to confirm or enter a new value.

    Note: Positive threshold is fixed at Q1 (all examples below Q1 are positive).
    Only the negative threshold (q3 + nn_confidence_th * IQR) is adjustable.

    Args:
        x_k_source_distances: Array of nearest neighbor distances [N, 1]
        q1: 25th percentile of distances
        q3: 75th percentile of distances
        iqr: Interquartile range (q3 - q1)

    Returns:
        nn_confidence_th: The selected threshold multiplier (for negative threshold only)
    """
    nn_confidence_th = 2.5  # Default starting value

    while True:
        # Compute thresholds with current nn_confidence_th
        # Positive threshold is fixed at Q1
        close_threshold = q1
        # Negative threshold uses the multiplier
        far_threshold = q3 + nn_confidence_th * iqr

        # Count samples in each category
        close_count = np.sum(x_k_source_distances.flatten() <= close_threshold)
        far_count = np.sum(x_k_source_distances.flatten() > far_threshold)
        middle_count = len(x_k_source_distances) - close_count - far_count

        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot distances
        frames = np.arange(len(x_k_source_distances))
        distances = x_k_source_distances.flatten()

        # Color points based on category
        colors = np.where(
            distances <= close_threshold, 'green',
            np.where(distances > far_threshold, 'red', 'gray')
        )
        ax.scatter(frames, distances, c=colors, s=1, alpha=0.7)

        # Plot threshold lines
        ax.axhline(y=close_threshold, color='green', linestyle='--', linewidth=2,
                   label=f'Close threshold (Q1): {close_threshold:.4f} ({close_count} samples)')
        ax.axhline(y=far_threshold, color='red', linestyle='--', linewidth=2,
                   label=f'Far threshold: {far_threshold:.4f} ({far_count} samples)')
        ax.axhline(y=q3, color='blue', linestyle=':', linewidth=1, alpha=0.5,
                   label=f'Q3: {q3:.4f}')

        ax.set_xlabel('Frame')
        ax.set_ylabel('Nearest Neighbor Distance')
        ax.set_title(f'Gating Threshold Selection (nn_confidence_th = {nn_confidence_th})\n'
                     f'Green: POSITIVE ({close_count}), Red: NEGATIVE ({far_count}), Gray: EXCLUDED ({middle_count})')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.1)

        # Ask user for input
        print(f"\n\033[93m--- Gating Threshold Selection ---\033[0m")
        print(f"Current nn_confidence_th: {nn_confidence_th}")
        print(f"Close threshold (Q1, fixed): {close_threshold:.4f} -> {close_count} positive samples")
        print(f"Far threshold (Q3 + {nn_confidence_th}*IQR): {far_threshold:.4f} -> {far_count} negative samples")
        print(f"Middle (excluded): {middle_count} samples")
        print(f"Q1: {q1:.4f}, Q3: {q3:.4f}, IQR: {iqr:.4f}")
        print()

        user_input = input("Enter new nn_confidence_th value (or press Enter to continue with current value): ").strip()

        plt.close(fig)

        if user_input == "":
            # User accepted current value
            print(f"\033[92mUsing nn_confidence_th = {nn_confidence_th}\033[0m")
            return nn_confidence_th
        else:
            try:
                nn_confidence_th = float(user_input)
                print(f"\033[94mUpdating nn_confidence_th to {nn_confidence_th}\033[0m")
            except ValueError:
                print(f"\033[91mInvalid input '{user_input}'. Please enter a number.\033[0m")


def load_user_static_poses(motion_path, reference_bvh, scale, target_frame_time):
    """
    Load user-provided static poses from sourceStatic/ and targetStatic/ folders.
    These are single-frame BVH files created by extract_average_pose.py.

    Args:
        motion_path: Path to the motion directory containing sourceStatic/ and targetStatic/
        reference_bvh: Reference BVH for skeleton structure
        scale: Scale factor for BVH files
        target_frame_time: Target frame time for resampling

    Returns:
        user_average_rotations: [num_user_chunks, num_joints, 4] - source rotations
        user_average_pelvis_positions: [num_user_chunks, 3] - source pelvis positions
        user_average_delta_rotations: [num_user_chunks, num_joints-1, 4] - delta rotations
        user_average_delta_pelvis_positions: [num_user_chunks, 3] - delta pelvis positions
        or None if folders don't exist or are empty
    """

    source_static_dir = os.path.join(motion_path, "sourceStatic")
    target_static_dir = os.path.join(motion_path, "targetStatic")

    # Check if both directories exist
    if not os.path.exists(source_static_dir) or not os.path.exists(target_static_dir):
        return None, None, None, None

    # Get list of source BVH files (sorted by numerical index)
    source_files = sorted(
        [f for f in os.listdir(source_static_dir) if f.endswith(".bvh")],
        key=lambda x: int(os.path.splitext(x)[0]),
    )

    if len(source_files) == 0:
        return None, None, None, None

    print(f"Loading {len(source_files)} user-provided static pose(s)")

    user_average_rotations_list = []
    user_average_pelvis_positions_list = []
    user_average_delta_rotations_list = []
    user_average_delta_pelvis_positions_list = []

    for file_name in source_files:
        source_path = os.path.join(source_static_dir, file_name)
        target_path = os.path.join(target_static_dir, file_name)

        if not os.path.exists(target_path):
            print(f"  Warning: Target file {file_name} not found, skipping...")
            continue

        # Load source BVH
        source_bvh = BVH()
        source_bvh.load(source_path)
        source_bvh.set_scale(scale)
        source_bvh = resample_bvh(source_bvh, target_frame_time)

        # Load target BVH
        target_bvh = BVH()
        target_bvh.load(target_path)
        target_bvh.set_scale(scale)
        target_bvh = resample_bvh(target_bvh, target_frame_time)

        # Extract data (should only have 1 frame)
        source_rotations, source_positions, _, _, _, _ = source_bvh.get_data()
        target_rotations, target_positions, _, _, _, _ = target_bvh.get_data()

        # Get single frame data
        source_rotations = source_rotations[0]  # [num_joints, 4]
        source_positions = source_positions[0]  # [num_joints, 3]
        target_rotations = target_rotations[0]  # [num_joints, 4]
        target_positions = target_positions[0]  # [num_joints, 3]

        # Compute deltas using continuous delta rotations (shortest path)
        delta_rotations = get_continuous_delta_rotations(
            source_rotations[1:],  # Exclude root (index 0)
            target_rotations[1:],
        )

        # For pelvis position (joint index 1)
        delta_pelvis_position = target_positions[1] - source_positions[1]

        # Store results
        user_average_rotations_list.append(source_rotations)
        user_average_pelvis_positions_list.append(source_positions[1])
        user_average_delta_rotations_list.append(delta_rotations)
        user_average_delta_pelvis_positions_list.append(delta_pelvis_position)


    if len(user_average_rotations_list) == 0:
        return None, None, None, None

    # Stack into arrays
    user_average_rotations = np.stack(
        user_average_rotations_list, axis=0
    )  # [num_user_chunks, num_joints, 4]
    user_average_pelvis_positions = np.stack(
        user_average_pelvis_positions_list, axis=0
    )  # [num_user_chunks, 3]
    user_average_delta_rotations = np.stack(
        user_average_delta_rotations_list, axis=0
    )  # [num_user_chunks, num_joints-1, 4]
    user_average_delta_pelvis_positions = np.stack(
        user_average_delta_pelvis_positions_list, axis=0
    )  # [num_user_chunks, 3]

    return (
        user_average_rotations,
        user_average_pelvis_positions,
        user_average_delta_rotations,
        user_average_delta_pelvis_positions,
    )
