# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch

import numpy as np
import pymotion.rotations.quat as quat
import pymotion.rotations.ortho6d as ortho6d

from pymotion.io.bvh import BVH
from pymotion.ops.skeleton import fk


def extract_Y_k(
    Y_k,
    global_translation_dim=3,
    pelvis_position_dim=3,
    joint_dim=6,
):
    # Y_k: [..., num_frames, 3 + 3 + 6 * num_joints]

    num_frames = Y_k.shape[-2]
    num_joints = (
        Y_k.shape[-1] - global_translation_dim - pelvis_position_dim
    ) // joint_dim
    batch_shape = Y_k.shape[:-2]

    delta_global_translations = Y_k[..., :, :global_translation_dim]
    delta_pelvis_positions = Y_k[
        ..., :, global_translation_dim : global_translation_dim + pelvis_position_dim
    ]

    delta_global_rot_increment = quat.normalize(
        ortho6d.to_quat(
            Y_k[
                ...,
                :,
                global_translation_dim
                + pelvis_position_dim : global_translation_dim
                + pelvis_position_dim
                + joint_dim,
            ].reshape(batch_shape + (num_frames, 3, 2))
        )
    )
    delta_global_rot_increment = quat.unroll(delta_global_rot_increment, dim=-2)

    delta_rotations = quat.normalize(
        ortho6d.to_quat(
            Y_k[
                ..., :, global_translation_dim + pelvis_position_dim + joint_dim :
            ].reshape(batch_shape + (num_frames, num_joints - 1, 3, 2))
        )
    )
    delta_rotations = quat.unroll(delta_rotations, dim=-2)

    return (
        delta_global_translations,  # [..., num_frames, 3]
        delta_pelvis_positions,  # [..., num_frames, 3]
        delta_global_rot_increment,  # [..., num_frames, 4]
        delta_rotations,  # [..., num_frames, num_joints - 1, 4]
    )


def smooth_rotations(
    delta_rotations,
    target_frame_time,
    smooth_factor=10.0,
):
    """
    delta_rotations: [..., num_frames, num_joints - 1, 4] (root excluded)
    """
    raise NotImplementedError(
        "Review this code before using again. Probably it requires some proper SLERP"
    )

    num_frames = delta_rotations.shape[-3]

    if torch.is_tensor(delta_rotations):
        smoothed_delta_rotations = delta_rotations.clone()
    else:
        smoothed_delta_rotations = delta_rotations.copy()
    for i in range(1, num_frames):
        # LERP
        t = smooth_factor * target_frame_time
        smoothed_delta_rotations[..., i, :, :] = (1.0 - t) * smoothed_delta_rotations[
            ..., i - 1, :, :
        ] + t * delta_rotations[..., i, :, :]

    return smoothed_delta_rotations  # [..., num_frames, num_joints - 1, 4]


def add_scaled_style_rots(
    dynamic_deltas,
    static_deltas,
    dynamic_scale=None,
    static_scale=None,
):
    """
    Apply scaled style rotations to dynamic and static deltas.

    Args:
        dynamic_deltas: [..., num_joints - 1, 4] (root excluded)
        static_deltas: [..., num_joints - 1, 4] (root excluded) or None
        dynamic_scale: Scale factor for dynamic deltas (scalar or per-joint array)
        static_scale: Scale factor for static deltas (scalar or per-joint array)
                      Note: static_scale does NOT affect the first joint (pelvis)

    Returns:
        scaled_deltas: [..., num_joints - 1, 4]
    """
    is_torch = torch.is_tensor(dynamic_deltas)

    # Convert numpy scale factors to torch if needed
    if is_torch:
        if dynamic_scale is not None and isinstance(dynamic_scale, np.ndarray):
            dynamic_scale = torch.from_numpy(dynamic_scale).to(dynamic_deltas.device, dynamic_deltas.dtype)
        if static_scale is not None and isinstance(static_scale, np.ndarray):
            static_scale = torch.from_numpy(static_scale).to(dynamic_deltas.device, dynamic_deltas.dtype)

    if dynamic_scale is not None:
        # Ensure dynamic_scale broadcasts correctly over [..., num_joints, 3]
        # Two cases:
        # 1. Per-frame scale: (frames, 1) -> expand to (frames, 1, 1) for broadcasting with (..., frames, joints, 3)
        # 2. Per-joint scale: (joints, 1) -> leave as-is for broadcasting with (..., joints, 3)
        if hasattr(dynamic_scale, 'ndim') and dynamic_scale.ndim == 2 and dynamic_scale.shape[-1] == 1:
            num_joints = dynamic_deltas.shape[-2]
            # Only expand if it's NOT a per-joint scale (first dim doesn't match joints)
            if dynamic_scale.shape[0] != num_joints:
                # Per-frame scale: expand to (frames, 1, 1)
                if is_torch:
                    dynamic_scale = dynamic_scale.unsqueeze(-1)
                else:
                    dynamic_scale = dynamic_scale[..., np.newaxis]

        # Scale rotations (handles canonicalization internally for shortest path)
        dynamic_deltas = quat.scale(dynamic_deltas, dynamic_scale)

    # If no static deltas, return the (potentially scaled) dynamic deltas
    if static_deltas is None:
        return dynamic_deltas

    # Create a copy to avoid modifying the original static_deltas
    if is_torch:
        scaled_static = static_deltas.clone()
    else:
        scaled_static = static_deltas.copy()

    if static_scale is not None:
        # Apply static_scale only to joints after the first one (pelvis)
        # Keep the first joint (pelvis) unscaled

        # If static_scale is a per-joint array, slice it to match joints [1:]
        # Per-joint arrays can be:
        # - 1D: (num_joints,)
        # - 2D: (num_joints, 1)
        # - 3D (per-frame, per-joint): (frames, num_joints, 1)
        # - 4D (batched, per-frame, per-joint): (batch, frames, num_joints, 1)
        # Per-frame scalar scales (3D with shape [frames, 1, 1]) are used as-is
        is_per_joint_scale = False
        num_joints_in_deltas = static_deltas.shape[-2]

        if hasattr(static_scale, "ndim"):
            if static_scale.ndim == 1 and len(static_scale) > 1:
                # 1D per-joint array: (num_joints,)
                is_per_joint_scale = True
            elif static_scale.ndim == 2 and static_scale.shape[-1] == 1:
                # Could be 2D per-joint (num_joints, 1) or per-frame (frames, 1)
                if static_scale.shape[0] == num_joints_in_deltas:
                    is_per_joint_scale = True
            elif static_scale.ndim == 3 and static_scale.shape[-1] == 1:
                # Could be 3D per-frame-per-joint (frames, num_joints, 1) or scalar (frames, 1, 1)
                if static_scale.shape[1] == num_joints_in_deltas:
                    is_per_joint_scale = True
            elif static_scale.ndim == 4 and static_scale.shape[-1] == 1:
                # 4D batched per-frame-per-joint (batch, frames, num_joints, 1)
                if static_scale.shape[2] == num_joints_in_deltas:
                    is_per_joint_scale = True

        if is_per_joint_scale:
            # Per-joint array: slice to exclude first joint
            if static_scale.ndim == 4:
                static_scale_sliced = static_scale[:, :, 1:, :]
            elif static_scale.ndim == 3:
                static_scale_sliced = static_scale[:, 1:, :]
            else:
                static_scale_sliced = static_scale[1:]
        else:
            # Scalar or per-frame: use as-is
            static_scale_sliced = static_scale

        # Scale only joints [1:] (all except first/pelvis) on the copy
        # Use quat.scale which handles canonicalization internally for shortest path
        scaled_static[..., 1:, :] = quat.scale(scaled_static[..., 1:, :], static_scale_sliced)

    scaled_deltas = quat.mul(scaled_static, dynamic_deltas)

    return scaled_deltas


def add_scaled_style_pos(
    dynamic_pelvis_pos,
    static_pelvis_pos,
    dynamic_pelvis_scale=1.0,
    static_pelvis_scale=1.0,
):
    """
    Apply scaled style positions to dynamic and static pelvis positions.

    Args:
        dynamic_pelvis_pos: [..., 3]
        static_pelvis_pos: [..., 3] or None
        dynamic_pelvis_scale: Scale factor for dynamic positions
        static_pelvis_scale: Scale factor for static positions

    Returns:
        scaled_pelvis_pos: [..., 3]
    """
    is_torch = torch.is_tensor(dynamic_pelvis_pos)

    # Convert numpy scale factors to torch if needed
    if is_torch:
        if isinstance(dynamic_pelvis_scale, np.ndarray):
            dynamic_pelvis_scale = torch.from_numpy(dynamic_pelvis_scale).to(dynamic_pelvis_pos.device, dynamic_pelvis_pos.dtype)
        if isinstance(static_pelvis_scale, np.ndarray):
            static_pelvis_scale = torch.from_numpy(static_pelvis_scale).to(dynamic_pelvis_pos.device, dynamic_pelvis_pos.dtype)

    if static_pelvis_pos is None:
        return dynamic_pelvis_pos * dynamic_pelvis_scale

    scaled_pelvis_pos = (
        static_pelvis_pos * static_pelvis_scale
        + dynamic_pelvis_pos * dynamic_pelvis_scale
    )
    return scaled_pelvis_pos


def compute_kinematics(
    delta_global_translations,
    delta_pelvis_positions,
    delta_global_rot_increment,
    delta_rotations,
    source_local_positions,
    source_local_rotations,
    fix_root,
):
    """
    delta_global_translations: [..., num_frames, 3]
    delta_pelvis_positions: [..., num_frames, 3]
    delta_global_rot_increment: [..., num_frames, 4]
    delta_rotations: [..., num_frames, num_joints - 1, 4] (root excluded)
    source_local_positions: [..., num_frames, num_joints, 3]
    source_local_rotations: [..., num_frames, num_joints, 4]
    """

    num_frames = source_local_rotations.shape[-3]

    if torch.is_tensor(source_local_positions):
        local_positions = source_local_positions.clone()
        local_rotations = torch.zeros(
            source_local_rotations.shape[:-3]
            + (num_frames, source_local_rotations.shape[-2], 4),
            device=source_local_positions.device,
            dtype=source_local_positions.dtype,
        )
    else:
        local_positions = source_local_positions.copy()
        local_rotations = np.zeros(
            source_local_rotations.shape[:-3]
            + (num_frames, source_local_rotations.shape[-2], 4)
        )

    # root joint ------ (vectorized)
    root_positions, root_rotations = integrate_root(
        source_local_positions[..., :, 0, :],
        source_local_rotations[..., :, 0, :],
        delta_global_translations,
        delta_global_rot_increment,
        fix_root,
    )
    local_positions[..., :, 0, :] = root_positions
    local_rotations[..., :, 0, :] = root_rotations

    # joints ------
    local_positions[..., :, 1, :] = (
        source_local_positions[..., :, 1, :] + delta_pelvis_positions
    )
    local_rotations[..., :, 1:, :] = quat.mul(
        source_local_rotations[..., :, 1:, :],
        delta_rotations,
    )

    return local_positions, local_rotations


def integrate_root(
    source_positions,
    source_rotations,
    delta_global_translations,
    delta_global_rot_increment,
    fix_root,
):
    """
    Vectorized version of integrate_root that processes all frames at once.

    source_positions: [..., num_frames, 3]
    source_rotations: [..., num_frames, 4]
    delta_global_translations: [..., num_frames, 3]
    delta_global_rot_increment: [..., num_frames, 4]
    """
    num_frames = source_positions.shape[-2]

    if torch.is_tensor(source_positions):
        # Prepare prev arrays by shifting
        prev_source_positions = torch.cat(
            [
                source_positions[..., 0:1, :],  # First frame uses itself
                source_positions[..., :-1, :],  # Rest use previous frame
            ],
            dim=-2,
        )

        # Zero out the Y component (avoid in-place operation for gradient computation)
        delta_global_translations = delta_global_translations.clone()
        delta_global_translations[..., 1] = 0

        # Compute all global rotations at once (independent of previous frames)
        global_rotations = quat.normalize(
            quat.mul(source_rotations, delta_global_rot_increment)
        )

        # Fix Root
        if fix_root:
            global_rotations = source_rotations.clone()

        # For positions, we need to accumulate using previous rotations
        # global_pos[i] = global_pos[i-1] + mul_vec(global_rot[i-1], s_global_translation[i] + delta[i])
        # Prepare rotations shifted for accumulation
        prev_global_rotations = torch.cat(
            [
                source_rotations[..., 0:1, :],  # First frame uses source rotation
                global_rotations[..., :-1, :],  # Rest use previous global rotation
            ],
            dim=-2,
        )

        # Compute source global translations
        s_global_translations = quat.mul_vec(
            quat.inverse(prev_global_rotations),
            source_positions - prev_source_positions,
        )

        # Compute all translation increments
        translation_increments = quat.mul_vec(
            prev_global_rotations, s_global_translations + delta_global_translations
        )

        # Accumulate positions using cumsum
        # Start with the initial position from source
        initial_pos = source_positions[..., 0:1, :]
        position_deltas = torch.cat(
            [
                torch.zeros_like(initial_pos),  # First frame: no delta, use initial
                translation_increments[..., 1:, :],  # Rest: accumulated deltas
            ],
            dim=-2,
        )

        global_positions = initial_pos + torch.cumsum(position_deltas, dim=-2)

        # Fix Root
        if fix_root:
            global_positions = source_positions.clone()

    else:
        # NumPy version
        prev_source_positions = np.concatenate(
            [source_positions[..., 0:1, :], source_positions[..., :-1, :]], axis=-2
        )

        delta_global_translations[..., 1] = 0

        global_rotations = quat.normalize(
            quat.mul(source_rotations, delta_global_rot_increment)
        )

        # Fix Root
        if fix_root:
            global_rotations = source_rotations.copy()

        prev_global_rotations = np.concatenate(
            [source_rotations[..., 0:1, :], global_rotations[..., :-1, :]], axis=-2
        )

        s_global_translations = quat.mul_vec(
            quat.inverse(prev_global_rotations),
            source_positions - prev_source_positions,
        )

        translation_increments = quat.mul_vec(
            prev_global_rotations, s_global_translations + delta_global_translations
        )

        initial_pos = source_positions[..., 0:1, :]
        position_deltas = np.concatenate(
            [np.zeros_like(initial_pos), translation_increments[..., 1:, :]], axis=-2
        )

        global_positions = initial_pos + np.cumsum(position_deltas, axis=-2)

        # Fix Root
        if fix_root:
            global_positions = source_positions.copy()

    return global_positions, global_rotations


def get_global_positions(
    Y_k,
    static_deltas,
    static_pelvis_pos,
    source_local_positions,
    source_local_rotations,
    parents,
    dynamic_scale=None,
    dynamic_pelvis_scale=None,
    static_scale=None,
    static_pelvis_scale=None,
):
    """
    Y_k: [..., num_frames, 3 + 3 + 6 * num_joints]
    static_deltas: [..., num_frames, num_joints - 1, 4]
    static_pelvis_pos: [..., num_frames, 3]
    source_local_positions: [..., num_frames, num_joints, 3]
    source_local_rotations: [..., num_frames, num_joints, 4]
    parents: [num_joints]
    dynamic_scale: Scale factor for dynamic deltas (scalar or per-joint array)
    dynamic_pelvis_scale: Scale factor for dynamic pelvis position
    static_scale: Scale factor for static deltas (scalar or per-joint array)
    static_pelvis_scale: Scale factor for static pelvis position
    """
    (
        delta_global_translations,
        delta_pelvis_positions,
        delta_global_rot_increment,
        delta_rotations,
    ) = extract_Y_k(Y_k)

    # Apply rotation scaling if static deltas exist and any scaling is provided
    if static_deltas is not None and (dynamic_scale is not None or static_scale is not None):
        delta_rotations = add_scaled_style_rots(
            dynamic_deltas=delta_rotations,
            static_deltas=static_deltas,
            dynamic_scale=dynamic_scale,
            static_scale=static_scale,
        )

    # Apply pelvis position scaling if static pelvis pos exists and any scaling is provided
    if static_pelvis_pos is not None and (dynamic_pelvis_scale is not None or static_pelvis_scale is not None):
        delta_pelvis_positions = add_scaled_style_pos(
            dynamic_pelvis_pos=delta_pelvis_positions,
            static_pelvis_pos=static_pelvis_pos,
            dynamic_pelvis_scale=dynamic_pelvis_scale if dynamic_pelvis_scale is not None else 1.0,
            static_pelvis_scale=static_pelvis_scale if static_pelvis_scale is not None else 1.0,
        )

    local_positions, local_rotations = compute_kinematics(
        delta_global_translations,
        delta_pelvis_positions,
        delta_global_rot_increment,
        delta_rotations,
        source_local_positions,
        source_local_rotations,
        fix_root=False,
    )

    positions, _ = fk(
        rot=local_rotations,
        global_pos=local_positions[..., 0, :],
        offsets=local_positions,
        parents=parents,
    )

    return positions  # [..., num_frames, num_joints, 3]


def get_local_positions(
    Y_k,
    static_deltas,
    source_local_rotations,
    parents,
    offsets,
):
    """
    Y_k: [..., num_frames, 3 + 3 + 6 * num_joints]
    static_deltas: [..., num_frames, num_joints - 1, 4]
    source_local_rotations: [..., num_frames, num_joints, 4]
    parents: [num_joints]
    offsets: [num_joints, 3]
    """
    (
        _,
        _,
        _,
        delta_rotations,
    ) = extract_Y_k(Y_k)

    if static_deltas is not None:
        delta_rotations = add_scaled_style_rots(
            dynamic_deltas=delta_rotations,
            static_deltas=static_deltas,
        )

    local_rotations = quat.mul(source_local_rotations[..., 1:, :], delta_rotations)

    if torch.is_tensor(Y_k):
        global_pos = torch.zeros(Y_k.shape[:-1] + (3,))
    else:
        global_pos = np.zeros(Y_k.shape[:-1] + (3,))

    positions, _ = fk(
        rot=local_rotations,
        global_pos=global_pos,
        offsets=offsets,
        parents=parents,
    )

    return positions[..., 1:, :]  # [..., num_frames, num_joints-2, 3]


def get_local_rotations(
    Y_k,
    source_local_rotations,
    static_deltas=None,
):
    """
    Y_k: [..., 3 + 3 + 6 * num_joints]
    static_deltas: [..., num_joints - 1, 4]
    source_local_rotations: [..., num_joints, 4]
    """
    (
        _,
        _,
        _,
        delta_rotations,
    ) = extract_Y_k(Y_k)

    if static_deltas is not None:
        delta_rotations = add_scaled_style_rots(
            dynamic_deltas=delta_rotations,
            static_deltas=static_deltas,
        )

    local_rotations = quat.mul(source_local_rotations[..., 1:, :], delta_rotations)

    return local_rotations  # [..., num_joints - 1, 4]


def load_average_deltas(experiment_dir):
    """
    Load average deltas from saved source/target average BVH files.
    Computes the inverse of what's done during saving in stylizer_train.py.

    Args:
        experiment_dir: Path to experiment directory containing sourceAvg and targetAvg folders

    Returns:
        average_delta_rotations: [num_chunks, num_joints-1, 4] - delta rotations (root excluded)
        average_delta_pelvis_positions: [num_chunks, 3] - delta pelvis positions
    """
    avg_source_experiment_dir = os.path.join(experiment_dir, "sourceAvg")
    avg_target_experiment_dir = os.path.join(experiment_dir, "targetAvg")

    # Get list of source BVH files (sorted by numerical index)
    source_files = sorted(
        [f for f in os.listdir(avg_source_experiment_dir) if f.endswith(".bvh")],
        key=lambda x: int(os.path.splitext(x)[0]),
    )

    average_delta_rotations_list = []
    average_delta_pelvis_positions_list = []
    target_bvhs_list = []

    for file_name in source_files:
        # Load source and target BVHs
        source_bvh = BVH()
        source_bvh.load(os.path.join(avg_source_experiment_dir, file_name))

        target_bvh = BVH()
        target_bvh.load(os.path.join(avg_target_experiment_dir, file_name))

        # Extract data (should only have 1 frame)
        source_rotations, source_positions, _, _, _, _ = source_bvh.get_data()
        target_rotations, target_positions, _, _, _, _ = target_bvh.get_data()

        # Get single frame data
        source_rotations = source_rotations[0]  # [num_joints, 4]
        source_positions = source_positions[0]  # [num_joints, 3]
        target_rotations = target_rotations[0]  # [num_joints, 4]
        target_positions = target_positions[0]  # [num_joints, 3]

        # Compute deltas (inverse of the saving operation)
        # Saving: target[1:] = quat.mul(source[1:], delta)
        # Loading: delta = quat.mul(quat.inverse(source[1:]), target[1:])
        delta_rotations = quat.mul(
            quat.inverse(source_rotations[1:]),  # Exclude root (index 0)
            target_rotations[1:],
        )

        # For pelvis position (joint index 1)
        # Saving: target[1] = source[1] + delta
        # Loading: delta = target[1] - source[1]
        delta_pelvis_position = target_positions[1] - source_positions[1]

        average_delta_rotations_list.append(delta_rotations)
        average_delta_pelvis_positions_list.append(delta_pelvis_position)
        target_bvhs_list.append(target_bvh)

    # Stack into arrays
    average_delta_rotations = np.stack(
        average_delta_rotations_list, axis=0
    )  # [num_chunks, num_joints-1, 4]
    average_delta_pelvis_positions = np.stack(
        average_delta_pelvis_positions_list, axis=0
    )  # [num_chunks, 3]

    return average_delta_rotations, average_delta_pelvis_positions, target_bvhs_list


def get_weighted_average_deltas(
    chunk_weights, average_delta_rotations, average_delta_pelvis_positions
):
    """
    Compute weighted average of deltas based on chunk weights using SLERP for quaternions.
    Uses the weighted_slerp function from pymotion for proper quaternion blending.

    Args:
        chunk_weights: One-hot or soft weights, shape [num_chunks] or [batch, num_chunks]
        average_delta_rotations: Delta rotations per chunk, shape [num_chunks, num_joints-1, 4]
        average_delta_pelvis_positions: Delta pelvis positions per chunk, shape [num_chunks, 3]

    Returns:
        weighted_delta_rotations: Weighted rotation deltas, shape [num_joints-1, 4] or [batch, num_joints-1, 4]
        weighted_delta_pelvis_positions: Weighted pelvis position deltas, shape [3] or [batch, 3]
    """
    is_torch = torch.is_tensor(chunk_weights)

    # Weighted sum for positions (linear space is fine)
    if chunk_weights.ndim == 1:
        # Single sample: [num_chunks]
        if is_torch:
            weights_pos = chunk_weights[:, None]  # [num_chunks, 1]
        else:
            weights_pos = chunk_weights[:, np.newaxis]  # [num_chunks, 1]

        weighted_delta_pelvis_positions = (
            weights_pos * average_delta_pelvis_positions
        ).sum(
            axis=0
        )  # [3]

        # Weighted SLERP for rotations using pymotion's weighted_slerp
        weighted_delta_rotations = quat.weighted_slerp(
            average_delta_rotations,  # [num_chunks, num_joints, 4]
            chunk_weights,  # [num_chunks]
            shortest=True,
        )  # [num_joints, 4]

    else:
        # Batch: [batch, num_chunks]
        batch_size = chunk_weights.shape[0]

        if is_torch:
            weights_pos = chunk_weights[:, :, None]  # [batch, num_chunks, 1]
            delta_pos_expanded = average_delta_pelvis_positions[
                None, :, :
            ]  # [1, num_chunks, 3]
            weighted_delta_pelvis_positions = (weights_pos * delta_pos_expanded).sum(
                dim=1
            )  # [batch, 3]
        else:
            weights_pos = chunk_weights[:, :, np.newaxis]  # [batch, num_chunks, 1]
            delta_pos_expanded = average_delta_pelvis_positions[
                np.newaxis, :, :
            ]  # [1, num_chunks, 3]
            weighted_delta_pelvis_positions = (weights_pos * delta_pos_expanded).sum(
                axis=1
            )  # [batch, 3]

        # Weighted SLERP for rotations - process each batch element separately
        # (pymotion's weighted_slerp only accepts [num_chunks] weights)
        if is_torch:
            weighted_delta_rotations = torch.stack(
                [
                    quat.weighted_slerp(
                        average_delta_rotations, chunk_weights[b], shortest=True
                    )
                    for b in range(batch_size)
                ]
            )  # [batch, num_joints, 4]
        else:
            weighted_delta_rotations = np.stack(
                [
                    quat.weighted_slerp(
                        average_delta_rotations, chunk_weights[b], shortest=True
                    )
                    for b in range(batch_size)
                ]
            )  # [batch, num_joints, 4]

    return weighted_delta_rotations, weighted_delta_pelvis_positions
