# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Motion loss module for computing reconstruction and contact losses.
"""

import torch

import torch.nn as nn

from features import get_feet_indices
from utils_inference import get_global_positions


def get_joint_weights(offsets, parents, discount=0.9):
    """Compute joint weights based on kinematic chain depth."""
    n_joints = offsets.shape[0]
    lengths = offsets.norm(dim=-1)
    joint_weights = 0.01 * torch.ones(
        n_joints, dtype=torch.float32, device=offsets.device
    )
    for i in range(n_joints)[::-1]:
        assert parents[i] < i or (i == 0 and parents[i] == 0)
        if parents[i] != i:
            joint_weights[parents[i]] += discount * (lengths[i] + joint_weights[i])
    joint_weights = joint_weights / joint_weights.sum()
    return joint_weights


def get_weights(offsets, parents):
    """
    Compute higher weights for joints with higher kinematic chains.
    """
    amplifier = 2.0
    joint_weights = get_joint_weights(offsets, parents)
    joint_weights = torch.exp(amplifier * joint_weights) - 1.0
    joint_weights_extended = torch.repeat_interleave(joint_weights, 6)
    # global translation is in indices 0, 1, 2 (weights of the root joint in indices [0, 6])
    # pelvis position is in indices 3, 4, 5 (weights of the pelvis joint in indices [6, 12])
    joint_weights_extended = torch.cat(
        [
            joint_weights_extended[[0, 1, 2]],
            joint_weights_extended[[6, 7, 8]],
            joint_weights_extended,
        ]
    )
    return joint_weights_extended, joint_weights


class ReconstructionLoss(nn.Module):
    """
    Reconstruction loss for paired and negative examples.

    Computes weighted MSE between predicted and ground truth kinematics.
    """

    def __init__(self, source_bvh):
        super(ReconstructionLoss, self).__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.offsets = torch.from_numpy(source_bvh.data["offsets"]).float().to(device)
        self.parents = torch.from_numpy(source_bvh.data["parents"]).to(device)
        self.joint_weights, _ = get_weights(self.offsets, self.parents)

    def forward(self, Y_hat, Y):
        """
        Compute weighted reconstruction loss.

        Args:
            Y_hat: Predicted target kinematics [frames, window_frames, features]
            Y: Ground truth target kinematics [frames, window_frames, features]

        Returns:
            Reconstruction loss scalar
        """
        diff = (Y_hat - Y) * self.joint_weights
        loss = (diff**2).mean()
        return loss


class ContactLoss(nn.Module):
    """
    Contact loss for source motion examples.

    Minimizes foot velocity when feet are in contact with the ground.
    Uses pose regularization to keep non-root pose stable while optimizing root.
    """

    def __init__(
        self,
        source_bvh,
        Y_k_standardize,
        body_parts_indices,
        lambda_contacts=100.0,
        lambda_pose_reg_upperbody=1.0,
        lambda_pose_reg_legs=1.0,
        lambda_smooth=0.01, # 0.001 for Cranberry
        target_frame_time=1.0 / 60.0,
    ):
        super(ContactLoss, self).__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.offsets = torch.from_numpy(source_bvh.data["offsets"]).float().to(device)
        self.parents = torch.from_numpy(source_bvh.data["parents"]).to(device)
        self.Y_k_standardize = Y_k_standardize
        self.feet_indices = get_feet_indices(source_bvh)
        self.lambda_contacts = lambda_contacts
        self.lambda_pose_reg_upperbody = lambda_pose_reg_upperbody
        self.lambda_pose_reg_legs = lambda_pose_reg_legs
        self.lambda_smooth = lambda_smooth
        self.joint_weights_extended, self.joint_weights = get_weights(
            self.offsets, self.parents
        )
        self.target_frame_time = target_frame_time

        # Body parts indices
        body_parts_indices_upperbody = (
            body_parts_indices[0] + body_parts_indices[3] + body_parts_indices[4]
        )
        if 0 in body_parts_indices_upperbody:
            body_parts_indices_upperbody.remove(0)
        if 1 in body_parts_indices_upperbody:
            body_parts_indices_upperbody.remove(1)
        # Expand upperbody indices: for each joint index, include it and the next 5 indices (6 DOF)
        expanded_upperbody = []
        for joint_idx in body_parts_indices_upperbody:
            expanded_upperbody.extend(range(joint_idx * 6, (joint_idx + 1) * 6))
        body_parts_indices_upperbody = expanded_upperbody
        body_parts_indices_upperbody = torch.tensor(
            body_parts_indices_upperbody, device=device
        )
        self.body_parts_indices_upperbody = body_parts_indices_upperbody

        body_parts_indices_legs = body_parts_indices[1] + body_parts_indices[2]
        if 0 in body_parts_indices_legs:
            body_parts_indices_legs.remove(0)
        if 1 in body_parts_indices_legs:
            body_parts_indices_legs.remove(1)
        body_parts_indices_legs = torch.tensor(body_parts_indices_legs, device=device)
        # Expand legs indices: for each joint index, include it and the next 5 indices (6 DOF)
        expanded_legs = [
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
        ]  # include pelvis position, and pelvis rotation
        for joint_idx in body_parts_indices_legs:
            expanded_legs.extend(range(joint_idx * 6, (joint_idx + 1) * 6))
        body_parts_indices_legs = expanded_legs
        body_parts_indices_legs = torch.tensor(body_parts_indices_legs, device=device)
        self.body_parts_indices_legs = body_parts_indices_legs

    def compute_smooth_loss(self, positions, contacts, contact_transition_window=10):
        """
        Compute smooth loss by penalizing large changes in acceleration (jerk).
        Uses a data-driven approach: computes 95th percentile of jerk, then excludes
        only high-jerk frames (above threshold) that occur near contact starts.

        Args:
            positions: Global joint positions [..., frames, num_joints, 3]
            contacts: Contact information [..., frames, num_feet]
            contact_transition_window: Number of frames before/after contact start to check

        Returns:
            Smooth root loss scalar
        """
        # Compute acceleration using finite differences
        # acceleration[t] = position[t+2] - 2*position[t+1] + position[t]
        if positions.shape[-3] < 3:
            return torch.tensor(0.0, device=positions.device)

        pos_t = positions[..., :-2, :, :]  # [..., frames-2, num_joints, 3]
        pos_t1 = positions[..., 1:-1, :, :]  # [..., frames-2, num_joints, 3]
        pos_t2 = positions[..., 2:, :, :]  # [..., frames-2, num_joints, 3]

        acceleration = (pos_t2 - 2 * pos_t1 + pos_t) / (
            self.target_frame_time**2
        )  # [..., frames-2, num_joints, 3]

        # Compute jerk (change in acceleration)
        # jerk[t] = acceleration[t+1] - acceleration[t]
        if acceleration.shape[-3] < 2:
            return torch.tensor(0.0, device=positions.device)

        jerk = (
            (acceleration[..., 1:, :, :] - acceleration[..., :-1, :, :]) / self.target_frame_time
        ) * self.joint_weights[
            ..., None, :, None
        ]  # [..., frames-3, num_joints, 3]

        smooth_loss = (jerk**2).mean()

        return smooth_loss

    def forward(
        self,
        Y_hat,
        Y_hat_ref,
        Y_s_pos,
        Y_s_rots,
        x_k_contacts,
        average_delta_rotations,
        average_delta_pelvis_positions,
        dynamic_scale=None,
        dynamic_pelvis_scale=None,
        static_scale=None,
        static_pelvis_scale=None,
        debug=False,
    ):
        """
        Compute contact loss with pose regularization.

        Args:
            Y_hat: Predicted target kinematics [..., frames, features]
            Y_hat_ref: Reference predictions from phase 1 (frozen) [..., frames, features]
            Y_s_pos: Source positions [..., frames, num_joints, 3]
            Y_s_rots: Source rotations [..., frames, num_joints, 4]
            x_k_contacts: Contact information [..., frames, num_feet]
            average_delta_rotations: Static style rotation deltas [..., frames, num_joints-1, 4]
            average_delta_pelvis_positions: Static style pelvis position deltas [..., frames, 3]
            dynamic_scale: Scale factor for dynamic deltas (scalar or per-joint array)
            dynamic_pelvis_scale: Scale factor for dynamic pelvis position
            static_scale: Scale factor for static deltas (scalar or per-joint array)
            static_pelvis_scale: Scale factor for static pelvis position
            debug: Whether to enable debug visualizations

        Returns:
            Total loss scalar (contact loss + pose regularization loss)
        """
        # FK - unstandardize and compute global positions with scaling
        Y_hat_unstd = self.Y_k_standardize.unstandardize_torch(Y_hat.clone())
        Y_hat_pos = get_global_positions(
            Y_hat_unstd,
            average_delta_rotations,
            average_delta_pelvis_positions,
            source_local_positions=Y_s_pos,
            source_local_rotations=Y_s_rots,
            parents=self.parents,
            dynamic_scale=dynamic_scale,
            dynamic_pelvis_scale=dynamic_pelvis_scale,
            static_scale=static_scale,
            static_pelvis_scale=static_pelvis_scale,
        )  # [..., frames, num_joints, 3]

        # Extract feet positions: [..., frames, num_feet, 3]
        feet_positions = Y_hat_pos[..., self.feet_indices, :]

        # Compute feet velocities using finite differences
        # feet_velocities: [..., frames, num_feet, 3]
        feet_velocities = (
            feet_positions[..., 1:, :, :] - feet_positions[..., :-1, :, :]
        ) / self.target_frame_time

        # Compute velocity magnitudes: [..., frames, num_feet]
        feet_velocity_magnitudes = torch.linalg.norm(feet_velocities, dim=-1)

        # Match contacts shape with velocities
        # Use contacts from frame 1 onwards (corresponding to the velocity computed from frame i to i+1)
        # x_k_contacts: [..., frames, num_feet]
        contacts = x_k_contacts[..., :-1, :]  # [..., frames, num_feet]

        # # TODO: remove or update to use the new batche-friendly version of the loss
        # if debug:
        #     with pymotion.render.blender.BlenderConnection() as conn:
        #         conn.clear_scene()
        #         conn.setup_rendering()
        #         conn.render_checkerboard_floor()
        #         Y_hat_pos_np = Y_hat_pos[..., [0, 2, 1]].detach().cpu().numpy()
        #         Y_hat_pos_np[..., 1] = -Y_hat_pos_np[..., 1]
        #         viz_contacts = np.where(
        #             x_k_contacts[..., None].detach().cpu().numpy(),
        #             Y_hat_pos_np[..., self.feet_indices, :],
        #             np.array([[[1000.0, 1000.0, 1000.0]]]),
        #         )
        #         conn.render_points_timeline(
        #             Y_hat_pos_np,
        #             np.array([1.0, 0.0, 0.0]),
        #             np.array([0.025]),
        #         )
        #         conn.render_points_timeline(
        #             viz_contacts,
        #             np.array([0.0, 1.0, 0.0]),
        #             np.array([0.035]),
        #         )

        #     import matplotlib.pyplot as plt

        #     # Visualize feet velocity magnitudes
        #     fig, axes = plt.subplots(
        #         feet_velocity_magnitudes.shape[1],
        #         1,
        #         figsize=(12, 3 * feet_velocity_magnitudes.shape[1]),
        #     )
        #     if feet_velocity_magnitudes.shape[1] == 1:
        #         axes = [axes]

        #     for foot_idx in range(feet_velocity_magnitudes.shape[1]):
        #         ax = axes[foot_idx]
        #         velocities = (
        #             feet_velocity_magnitudes[:, foot_idx].detach().cpu().numpy()
        #         )
        #         contacts_vis = contacts[:, foot_idx].detach().cpu().numpy()

        #         frames = np.arange(1, len(velocities) + 1)
        #         ax.plot(
        #             frames,
        #             velocities,
        #             label=f"Foot {foot_idx} Velocity Magnitude",
        #             linewidth=1,
        #         )

        #         # Highlight contact regions
        #         contact_regions = np.where(contacts_vis > 0.5)[0]
        #         if len(contact_regions) > 0:
        #             ax.scatter(
        #                 contact_regions,
        #                 velocities[contact_regions],
        #                 c="red",
        #                 s=10,
        #                 alpha=0.5,
        #                 label="Contact Frames",
        #             )

        #         ax.set_xlabel("Frame")
        #         ax.set_ylabel("Velocity Magnitude (m/s)")
        #         ax.set_title(f"Foot {foot_idx} Velocity Magnitude")
        #         ax.legend()
        #         ax.grid(True, alpha=0.3)

        #     plt.tight_layout()
        #     plt.show()

        # Penalize velocities when foot is in contact
        # When contact == 1.0, we want velocity magnitude to be 0
        # Divide by sum of contacts to get the average error per contact frame
        # (not diluted by non-contact frames)
        contact_count = contacts.sum() + 1e-8  # Add epsilon to avoid division by zero
        contact_loss = (
            (feet_velocity_magnitudes * contacts) ** 2
        ).sum() / contact_count
        contact_loss = self.lambda_contacts * contact_loss

        # Pose regularization: penalize changes to pose (ignore root translation)
        pose_reg_loss_upperbody = (
            (
                Y_hat[..., self.body_parts_indices_upperbody]
                - Y_hat_ref[..., self.body_parts_indices_upperbody]
            )
            ** 2
        ).mean()
        pose_reg_loss_legs = (
            (
                Y_hat[..., self.body_parts_indices_legs]
                - Y_hat_ref[..., self.body_parts_indices_legs]
            )
            ** 2
        ).mean()

        # Smooth root loss: penalize large changes in acceleration for root and pelvis
        # Exclude contact frames where high jerk is expected
        smooth_loss = (
            self.compute_smooth_loss(Y_hat_pos, x_k_contacts) * self.lambda_smooth
        )

        pose_reg_loss = (
            self.lambda_pose_reg_upperbody * pose_reg_loss_upperbody
            + self.lambda_pose_reg_legs * pose_reg_loss_legs
        )

        total_loss = contact_loss + pose_reg_loss + smooth_loss

        return total_loss, contact_loss, pose_reg_loss, smooth_loss
