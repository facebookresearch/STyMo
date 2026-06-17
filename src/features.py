# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pymotion.rotations.quat as quat
import pymotion.rotations.ortho6d as ortho6d

from utils_inference import compute_kinematics
from pymotion.io.bvh import BVH
from pymotion.ops.skeleton import fk
from scipy.signal import savgol_filter
from scipy.ndimage import binary_opening, binary_closing, binary_dilation
from body_parts import body_parts, feet_joints_names
from utils import (
    second_order_backward_diff,
    second_order_forward_diff,
    fourth_order_central_diff,
    get_window_length,
)


def get_body_parts_indices(bvh):
    names = list(bvh.data["names"])
    target_body_part = None
    for body_parts_i in body_parts:
        all_names = [name for body_part in body_parts_i for name in body_part]
        if all(name in names for name in all_names):
            target_body_part = body_parts_i
            break
    if target_body_part is None:
        raise ValueError("No target body part found")
    body_parts_indices = []
    for body_part in target_body_part:
        indices = [names.index(name) for name in body_part]
        body_parts_indices.append(indices)
    return body_parts_indices


def get_feet_indices(bvh):
    names = list(bvh.data["names"])
    target_feet_joints = None
    for feet_joints_i in feet_joints_names:
        if all(name in names for name in feet_joints_i):
            target_feet_joints = feet_joints_i
            break
    if target_feet_joints is None:
        raise ValueError("No target feet joints found")
    feet_indices = [names.index(name) for name in target_feet_joints]
    return feet_indices


def get_source_kinematics(bvh, delta_indices):
    """
    Source kinematics
    """
    local_rotations, local_positions, parents, offsets, _, _ = bvh.get_data()
    num_frames = local_rotations.shape[0]
    global_positions = local_positions[:, 0, :]  # root joint
    pos, _ = fk(local_rotations, global_positions, offsets, parents)
    x_k = []
    for i in range(num_frames):
        indices = [max(-1, min(num_frames, d + i)) for d in delta_indices]
        mask = [i == -1 or i == num_frames for i in indices]
        indices = [i - 1 if i == num_frames else i for i in indices]
        x_k_window = pos[indices, ...].copy()
        x_k_current = pos[i, ...]
        # relative to current frame
        origin = x_k_current[0:1, :].copy()
        inv_root_rot = quat.inverse(local_rotations[i, 0:1, :])
        x_k_window = quat.mul_vec(inv_root_rot, x_k_window - origin)
        x_k_window[mask, ...] = 0
        # results
        x_k.append(x_k_window.flatten())
    x_k = np.stack(x_k, axis=0)
    return x_k  # [num_frames, window_size * num_joints * 3]


def get_source_kinematics_full_body(
    bvh, delta_indices, feet_velocity_contact_th, use_velocity=False
):
    """
    Source kinematics
    """
    feet_indices = get_feet_indices(bvh)
    local_rotations, local_positions, parents, offsets, _, _ = bvh.get_data()
    frame_time = bvh.data["frame_time"]
    num_frames = local_rotations.shape[0]
    global_positions = local_positions[:, 0, :]  # root joint
    if np.all(np.abs(global_positions[:, 1]) > 0.01):
        raise ValueError(
            "Global positions have non-zero y-axis. The provided animations should have a root joint projected to the ground and a pelvis joint with local positions to the root joint."
        )
    pelvis_positions = local_positions[:, 1, :]  # pelvis joint
    offsets = np.tile(offsets, (local_rotations.shape[0], 1, 1))
    offsets[:, 1, :] = pelvis_positions
    pos, _ = fk(local_rotations, global_positions, offsets, parents)
    feet_contacts = compute_feet_contacts(
        pos, feet_indices, frame_time, feet_velocity_contact_th
    )
    # # Uncomment to verify the fk result
    # with pymotion.render.blender.BlenderConnection() as conn:
    #     conn.clear_scene()
    #     conn.setup_rendering()
    #     conn.render_checkerboard_floor()
    #     pos = pos[..., [0, 2, 1]]
    #     pos[..., 1] = -pos[..., 1]
    #     viz_contacts = np.where(
    #         feet_contacts[..., None],
    #         pos[:, feet_indices],
    #         np.array([[[1000.0, 1000.0, 1000.0]]]),
    #     )
    #     conn.render_points_timeline(
    #         pos, np.array([1.0, 0.0, 0.0]), radius=np.array([0.025])
    #     )
    #     conn.render_points_timeline(
    #         viz_contacts, np.array([0.0, 1.0, 0.0]), np.array([0.035])
    #     )
    #     input("Press enter to continue...")
    if use_velocity:
        pos_p2 = np.concatenate(
            [pos[2:, ...], np.tile(pos[-2:-1, ...], (2, 1, 1))], axis=0
        )
        pos_p1 = np.concatenate([pos[1:, ...], pos[-2:-1, ...]], axis=0)
        pos_m1 = np.concatenate([pos[0:1, ...], pos[:-1, ...]], axis=0)
        pos_m2 = np.concatenate(
            [np.tile(pos[0:1, ...], (2, 1, 1)), pos[:-2, ...]], axis=0
        )
        vel = (-pos_p2 + 8 * pos_p1 - 8 * pos_m1 + pos_m2) / (12 * frame_time)
    x_k = []
    x_k_contacts = []
    for i in range(num_frames):
        indices = [max(-1, min(num_frames, d + i)) for d in delta_indices]
        mask = [i == -1 or i == num_frames for i in indices]
        indices = [i - 1 if i == num_frames else i for i in indices]
        x_k_window = pos[indices, ...].copy()
        x_k_contacts_window = feet_contacts[indices, ...].copy()
        if use_velocity:
            x_k_vel_window = vel[indices, ...].copy()
        x_k_current = pos[i, ...]
        # relative to current frame
        origin = x_k_current[0:1, :].copy()
        inv_root_rot = quat.inverse(local_rotations[i, 0:1, :])
        x_k_window = quat.mul_vec(inv_root_rot, x_k_window - origin)
        x_k_window[mask, ...] = 0
        if use_velocity:
            x_k_vel_window = quat.mul_vec(inv_root_rot, x_k_vel_window)
            x_k_vel_window[mask, ...] = 0
            x_k_window = np.concatenate([x_k_window, x_k_vel_window], axis=-1)
        x_k_contacts_window[mask, ...] = 0
        x_k_window = np.concatenate(
            [x_k_window.reshape(len(delta_indices), -1), x_k_contacts_window], axis=-1
        )
        # results
        x_k.append(x_k_window)
        x_k_contacts.append(x_k_contacts_window)
    x_k = np.stack(x_k, axis=0)  # [num_frames, window_size, num_joints * (3 + [3]) + 2]
    x_k_contacts = np.stack(x_k_contacts, axis=0)  # [num_frames, window_size, 2]
    return x_k, x_k_contacts


def compute_feet_contacts(positions, feet_joints, frame_time, feet_velocity_contact_th):
    """
    positions: [..., num_frames, num_joints, 3]
    feet_joints: [num_feet_joints]
    feet_velocity_contact_th: float
    """
    feet_positions = positions[..., feet_joints, :]
    prev_feet_positions = np.concatenate(
        [feet_positions[..., 0:1, :, :], feet_positions[..., :-1, :, :]], axis=-3
    )
    feet_velocity = (feet_positions - prev_feet_positions) / frame_time
    feet_velocity = np.linalg.norm(feet_velocity, axis=-1)
    feet_contacts = np.where(feet_velocity < feet_velocity_contact_th, 1, 0)
    # Grow (dilate) the contact regions by 3 frames on each side after closing
    feet_contacts = (
        binary_dilation(
            binary_closing(
                binary_opening(feet_contacts.astype(bool), structure=np.ones((3, 1))),
                structure=np.ones((3, 1)),
            ),
            structure=np.ones((7, 1)),  # 3 frames before and after (total 7)
        )
    ).astype(np.float32)
    return feet_contacts  # [..., num_frames, num_feet_joints]


def get_source_accelerations(bvh, delta_indices):
    """
    Source accelerations
    """
    local_rotations, local_positions, parents, offsets, _, _ = bvh.get_data()
    frame_time = bvh.data["frame_time"]
    num_frames = local_rotations.shape[0]
    global_positions = local_positions[:, 0, :]  # root joint
    pos, _ = fk(local_rotations, global_positions, offsets, parents)
    x_a = []
    for i in range(num_frames):
        if i < 2:
            # forward difference
            i_1 = i + 1
            i_2 = i + 2
            i_3 = i + 3
            pos_0 = pos[i, ...].copy()
            pos_1 = pos[i_1, ...].copy()
            pos_2 = pos[i_2, ...].copy()
            pos_3 = pos[i_3, ...].copy()
            # compute accelerations
            acc = second_order_forward_diff(pos_0, pos_1, pos_2, pos_3)
            # relative to current frame
            inv_root_rot = quat.inverse(local_rotations[i, 0:1, :])
            acc = quat.mul_vec(inv_root_rot, acc)
        elif i > num_frames - 3:
            # backward difference
            i_m3 = i - 3
            i_m2 = i - 2
            i_m1 = i - 1
            pos_0 = pos[i, ...].copy()
            pos_m1 = pos[i_m1, ...].copy()
            pos_m2 = pos[i_m2, ...].copy()
            pos_m3 = pos[i_m3, ...].copy()
            # compute accelerations
            acc = second_order_backward_diff(pos_m3, pos_m2, pos_m1, pos_0)
            # relative to current frame
            inv_root_rot = quat.inverse(local_rotations[i, 0:1, :])
            acc = quat.mul_vec(inv_root_rot, acc)
        else:
            # central difference
            i_m2 = i - 2
            i_m1 = i - 1
            i_1 = i + 1
            i_2 = i + 2
            pos_0 = pos[i, ...].copy()
            pos_m2 = pos[i_m2, ...].copy()
            pos_m1 = pos[i_m1, ...].copy()
            pos_1 = pos[i_1, ...].copy()
            pos_2 = pos[i_2, ...].copy()
            # compute accelerations
            acc = fourth_order_central_diff(pos_m2, pos_m1, pos_0, pos_1, pos_2)
            # relative to current frame
            inv_root_rot = quat.inverse(local_rotations[i, 0:1, :])
            acc = quat.mul_vec(inv_root_rot, acc)
        x_a.append(acc)
    x_a = np.stack(x_a, axis=0)
    # remove outliers
    mean = np.mean(x_a, axis=0)
    std = np.std(x_a, axis=0)
    x_a = np.where(
        np.abs(x_a - mean) > 3 * std,
        np.sign(x_a - mean) * 3 * std + mean,
        x_a,
    )
    if len(delta_indices) != 1:
        x_a_window = []
        for i in range(num_frames):
            indices = [max(-1, min(num_frames, d + i)) for d in delta_indices]
            mask = [i == -1 or i == num_frames for i in indices]
            indices = [i - 1 if i == num_frames else i for i in indices]
            x_a_window_i = x_a[indices, ...]
            x_a_window_i[mask, ...] = 0
            x_a_window.append(x_a_window_i.flatten())
        x_a = np.stack(x_a_window, axis=0)
    # smooth
    window_length = get_window_length(0.25, frame_time, polyorder=3)
    x_a = savgol_filter(x_a, window_length=window_length, polyorder=3, axis=0)
    x_a = x_a.reshape(x_a.shape[0], -1)
    return x_a


def get_target_kinematics(source_bvh, target_bvh, delta_indices):
    """
    Target kinematics
    """
    s_local_rotations, s_local_positions, _, _, _, _ = source_bvh.get_data()
    local_rotations, local_positions, _, _, _, _ = target_bvh.get_data()
    frame_time = target_bvh.data["frame_time"]
    num_frames = local_rotations.shape[0]
    num_joints = local_rotations.shape[1]
    s_global_positions = s_local_positions[:, 0, :]  # root joint
    s_global_rotations = s_local_rotations[:, 0, :]  # root joint
    global_positions = local_positions[:, 0, :]  # root joint
    global_rotations = local_rotations[:, 0, :]  # root joint
    Y_k = [[] for _ in range(num_joints + 1)]
    for i in range(num_frames):
        prev_i = max(0, i - 1)

        s_inv_prev_root_rot = quat.inverse(s_global_rotations[prev_i])
        inv_prev_root_rot = quat.inverse(global_rotations[prev_i])

        s_translation = quat.mul_vec(
            s_inv_prev_root_rot, s_global_positions[i] - s_global_positions[prev_i]
        )
        translation = quat.mul_vec(
            inv_prev_root_rot, global_positions[i] - global_positions[prev_i]
        )

        s_delta_rot = quat.mul(s_global_rotations[i], s_inv_prev_root_rot)
        delta_rot = quat.mul(global_rotations[i], inv_prev_root_rot)

        translation = translation - s_translation
        delta_rot = quat.mul(delta_rot, quat.inverse(s_delta_rot))

        Y_k[0].append(translation)
        Y_k[1].append(delta_rot)
        for j in range(2, num_joints + 1):
            Y_k[j].append(local_rotations[i, j - 1, :])
    for k in range(len(Y_k)):
        Y_k[k] = np.stack(Y_k[k], axis=0)
    # smooth translations
    window_length = get_window_length(0.25, frame_time, polyorder=3)
    Y_k[0] = savgol_filter(Y_k[0], window_length=window_length, polyorder=3, axis=0)
    if len(delta_indices) != 1:
        Y_k_window = []
        for j in range(len(Y_k)):
            Y_k_window_j = []
            for i in range(num_frames):
                indices = [max(-1, min(num_frames, d + i)) for d in delta_indices]
                mask = [i == -1 or i == num_frames for i in indices]
                indices = [i - 1 if i == num_frames else i for i in indices]
                Y_k_window_j.append(Y_k[j][indices, ...])
                Y_k_window_j[i][mask, ...] = 0
            Y_k_window_j = np.stack(Y_k_window_j, axis=0)
            Y_k_window.append(Y_k_window_j)
        Y_k = Y_k_window
    return Y_k


def get_target_accelerations_local(Y_k, frame_time):
    """
    Target accelerations
    """
    num_frames = Y_k[0].shape[0]
    num_joints = len(Y_k) - 1
    Y_a = [[] for _ in range(num_joints + 1)]
    for i in range(num_frames):
        if i < 2:
            # forward difference
            i_1 = i + 1
            i_2 = i + 2
            i_3 = i + 3
            for j in range(num_joints + 1):
                Y_a[j].append(
                    second_order_forward_diff(
                        Y_k[j][i], Y_k[j][i_1], Y_k[j][i_2], Y_k[j][i_3]
                    )
                )
        elif i > num_frames - 3:
            # backward difference
            i_m3 = i - 3
            i_m2 = i - 2
            i_m1 = i - 1
            for j in range(num_joints + 1):
                Y_a[j].append(
                    second_order_backward_diff(
                        Y_k[j][i_m3], Y_k[j][i_m2], Y_k[j][i_m1], Y_k[j][i]
                    )
                )
        else:
            # central difference
            i_m2 = i - 2
            i_m1 = i - 1
            i_1 = i + 1
            i_2 = i + 2
            for j in range(num_joints + 1):
                Y_a[j].append(
                    fourth_order_central_diff(
                        Y_k[j][i_m2], Y_k[j][i_m1], Y_k[j][i], Y_k[j][i_1], Y_k[j][i_2]
                    )
                )
    for k in range(len(Y_a)):
        Y_a[k] = np.stack(Y_a[k], axis=0)

    # smooth
    window_length = get_window_length(0.25, frame_time, polyorder=3)
    for k in range(len(Y_a)):
        Y_a[k] = savgol_filter(Y_a[k], window_length=window_length, polyorder=3, axis=0)

    return Y_a


def set_bvh_from_target_kinematics(bvh, Y_k, fix_root):
    """
    Set bvh from target kinematics
    """
    pred_bvh = BVH()
    pred_bvh.copy(bvh)
    channels = pred_bvh.data["channels"]
    channels[1:] = 3
    pred_bvh.data["channels"] = channels

    num_frames = Y_k[0].shape[0]

    local_rotations, local_positions, parents, offsets, _, _ = pred_bvh.get_data()

    global_pos = local_positions[0, 0, :].copy()  # root joint
    global_rot = local_rotations[0, 0, :].copy()
    s_pos = local_positions[:, 0, :].copy()
    s_rot = local_rotations[:, 0, :].copy()

    local_positions = np.zeros((num_frames, local_positions.shape[1], 3))
    local_rotations = np.zeros((num_frames, local_rotations.shape[1], 4))

    for i in range(num_frames):
        prev_i = max(0, i - 1)
        # root ------
        s_inv_prev_root_rot = quat.inverse(s_rot[prev_i])
        s_translation = quat.mul_vec(s_inv_prev_root_rot, s_pos[i] - s_pos[prev_i])
        s_delta_rot = quat.mul(s_rot[i], s_inv_prev_root_rot)

        translation = Y_k[0][i]
        s_translation = quat.mul_vec(global_rot, s_translation)
        translation = quat.mul_vec(global_rot, translation)
        # HARDCODED: remove vertical axis
        translation[1] = 0
        world_translation = translation + s_translation
        global_pos = global_pos + world_translation
        if fix_root:
            global_pos = s_pos[i]
        local_positions[i, 0, :] = global_pos

        delta_rot = Y_k[1][i]
        delta_rot = quat.mul(delta_rot, s_delta_rot)
        global_rot = quat.normalize(quat.mul(delta_rot, global_rot))
        if fix_root:
            global_rot = s_rot[i]
        local_rotations[i, 0, :] = global_rot

        # joints ------
        joint_rot = np.stack(Y_k[2:], axis=0).swapaxes(0, 1)[i]
        local_rotations[i, 1:, :] = quat.normalize(joint_rot)
    pred_bvh.set_data(local_rotations, local_positions)
    return pred_bvh


def set_bvh_from_full_body_target_kinematics(
    source_bvh,
    delta_global_translations,
    delta_pelvis_positions,
    delta_global_rot_increment,
    delta_rotations,
    fix_root,
    global_translation_dim=3,
    pelvis_position_dim=3,
):
    """
    Set bvh from target kinematics.
        delta_global_translations: [num_frames, 3]
        delta_pelvis_positions: [num_frames, 3]
        delta_global_rot_increment: [num_frames, 4]
        delta_rotations: [num_frames, num_joints - 1, 4] (root excluded)
    """

    pred_bvh = BVH()
    pred_bvh.copy(source_bvh)

    local_rotations, local_positions, _, _, _, _ = pred_bvh.get_data()

    local_positions, local_rotations = compute_kinematics(
        delta_global_translations,
        delta_pelvis_positions,
        delta_global_rot_increment,
        delta_rotations,
        local_positions,
        local_rotations,
        fix_root,
    )

    pred_bvh.set_data(local_rotations, local_positions)
    return pred_bvh


def get_continuous_delta_rotations(rotations, target_rotations):
    """
    Get continuous delta rotations
    """
    # 1. Calculate the dot product for all pairs at once
    dot_products = np.sum(rotations * target_rotations, axis=-1, keepdims=True)

    # 2. Create a "corrected" version of target_rotations
    # Where the dot product is negative, use the flipped quaternion (-target_rotations).
    # Otherwise, use the original target_rotations.
    target_rotations_closest = np.where(
        dot_products < 0, -target_rotations, target_rotations
    )

    # 3. Now, calculate the delta using the corrected quaternion.
    # This guarantees the shortest path rotation.
    delta_rotations = quat.mul(quat.inverse(rotations), target_rotations_closest)

    return delta_rotations


def get_chunks(chunk_size, num_frames):
    chunk_start = np.arange(0, num_frames, chunk_size)
    chunk_end = chunk_start + chunk_size
    if chunk_start[-1] != num_frames - chunk_size:
        chunk_start = chunk_start[:-1]
        chunk_end = chunk_end[:-1]
        chunk_end[-1] = num_frames
    return chunk_start, chunk_end


def get_full_body_target_kinematics(
    source_bvh, target_bvh, delta_indices, chunk_start, chunk_end
):
    """
    Target kinematics
    """
    s_local_rotations, s_local_positions, _, _, _, _ = source_bvh.get_data()
    local_rotations, local_positions, _, _, _, _ = target_bvh.get_data()
    frame_time = target_bvh.data["frame_time"]
    num_frames = local_rotations.shape[0]
    num_joints = local_rotations.shape[1]

    # Global (root joint)
    s_global_positions = s_local_positions[:, 0, :]
    global_positions = local_positions[:, 0, :]
    s_global_rotations = s_local_rotations[:, 0, :]
    global_rotations = local_rotations[:, 0, :]
    s_prev_global_positions = np.concatenate(
        (s_global_positions[0:1, :], s_global_positions[:-1, :]), axis=0
    )
    prev_global_positions = np.concatenate(
        (global_positions[0:1, :], global_positions[:-1, :]), axis=0
    )
    prev_global_rotations = np.concatenate(
        (global_rotations[0:1, :], global_rotations[:-1, :]), axis=0
    )
    s_global_translations = quat.mul_vec(
        quat.inverse(prev_global_rotations),
        s_global_positions - s_prev_global_positions,
    )
    global_translations = quat.mul_vec(
        quat.inverse(prev_global_rotations), global_positions - prev_global_positions
    )
    delta_global_translations = global_translations - s_global_translations
    delta_global_rot_increment = ortho6d.from_quat(
        get_continuous_delta_rotations(s_global_rotations, global_rotations)
    ).reshape(num_frames, 6)

    # Process poses in chunks to get per-chunk average deltas
    average_delta_pelvis_positions_list = []
    average_delta_rotations_list = []
    average_pelvis_positions_list = []
    average_rotations_list = []
    delta_pelvis_positions = np.zeros((num_frames, 3))
    delta_rotations = np.zeros((num_frames, (num_joints - 1) * 6))

    for c_s, c_e in zip(chunk_start, chunk_end):
        # Pelvis joint
        s_pelvis_positions_chunk = s_local_positions[c_s:c_e, 1, :]
        pelvis_positions_chunk = local_positions[c_s:c_e, 1, :]
        delta_pelvis_positions_chunk = pelvis_positions_chunk - s_pelvis_positions_chunk
        average_delta_pelvis_positions_chunk = np.mean(
            delta_pelvis_positions_chunk, axis=0
        )
        delta_pelvis_positions[c_s:c_e] = (
            delta_pelvis_positions_chunk - average_delta_pelvis_positions_chunk
        )
        average_delta_pelvis_positions_list.append(average_delta_pelvis_positions_chunk)
        average_pelvis_positions_list.append(np.mean(s_pelvis_positions_chunk, axis=0))

        # Joint Delta Rotations (excluding root joint)
        delta_rotations_quat_chunk = get_continuous_delta_rotations(
            s_local_rotations[c_s:c_e], local_rotations[c_s:c_e]
        )[:, 1:]
        average_delta_rotations_chunk = quat.normalize(
            np.mean(delta_rotations_quat_chunk, axis=0)
        )
        delta_rotations_quat_chunk = get_continuous_delta_rotations(
            average_delta_rotations_chunk, delta_rotations_quat_chunk
        )
        delta_rotations[c_s:c_e] = ortho6d.from_quat(
            delta_rotations_quat_chunk
        ).reshape((c_e - c_s), (num_joints - 1) * 6)
        average_delta_rotations_list.append(average_delta_rotations_chunk)
        average_rotations_list.append(
            quat.normalize(np.mean(s_local_rotations[c_s:c_e], axis=0))
        )

    # Stack the per-chunk averages
    average_delta_pelvis_positions = np.stack(
        average_delta_pelvis_positions_list, axis=0
    )
    average_delta_rotations = np.stack(average_delta_rotations_list, axis=0)
    average_pelvis_positions = np.stack(average_pelvis_positions_list, axis=0)
    average_rotations = np.stack(average_rotations_list, axis=0)

    # Results
    Y_k = np.empty(
        (num_frames, 3 + 3 + 6 * num_joints)
    )  # root translation, pelvis position, joint delta rotations (including root)
    Y_k[:, 0:3] = delta_global_translations
    Y_k[:, 3:6] = delta_pelvis_positions
    Y_k[:, 6:12] = delta_global_rot_increment
    Y_k[:, 12:] = delta_rotations

    if len(delta_indices) != 1:
        identity_rot = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        identity_vec = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        identity_6d = ortho6d.from_quat(identity_rot).reshape(6)
        identity_Y_k = np.concatenate(
            [identity_vec, identity_vec] + [identity_6d] * num_joints
        )
        Y_k_window = []
        Y_s_pos = []
        Y_s_rots = []
        for i in range(num_frames):
            indices = [max(-1, min(num_frames, d + i)) for d in delta_indices]
            mask = [i == -1 or i == num_frames for i in indices]
            indices = [i - 1 if i == num_frames else i for i in indices]
            Y_k_window.append(Y_k[indices, ...])
            Y_s_pos.append(s_local_positions[indices, ...])
            Y_s_rots.append(s_local_rotations[indices, ...])
            Y_k_window[i][mask, ...] = np.tile(identity_Y_k, (sum(mask), 1))
            Y_s_pos[i][mask, ...] = 0
            Y_s_rots[i][mask, ...] = np.tile(identity_rot, (sum(mask), num_joints, 1))
        Y_k = np.stack(Y_k_window, axis=0)
        Y_s_pos = np.stack(Y_s_pos, axis=0)
        Y_s_rots = np.stack(Y_s_rots, axis=0)
    # Results -----------------------------------------------------------------
    # Y_k has shape [num_frames, len(delta_indices), 3 + 3 + 6 * num_joints]
    # average_delta_rotations has shape [num_chunks, num_joints-1, 4]
    # average_delta_pelvis_positions has shape [num_chunks, 3]
    # Y_s_rots has shape [num_frames, len(delta_indices), num_joints, 4]
    # Y_s_pos has shape [num_frames, len(delta_indices), num_joints, 3]
    # average_rotations has shape [num_chunks, num_joints, 4]
    # average_pelvis_positions has shape [num_chunks, 3]
    return (
        Y_k,
        average_delta_rotations,
        average_delta_pelvis_positions,
        Y_s_pos,
        Y_s_rots,
        average_rotations,
        average_pelvis_positions,
    )
