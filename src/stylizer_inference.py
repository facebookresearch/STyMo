# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

import numpy as np
import pymotion.rotations.ortho6d as ortho6d

from utils_inference import get_weighted_average_deltas


def optimize_contact(
    Y_hat_ref_list,
    average_delta_rotations_hat_ref_list,
    average_delta_pelvis_positions_hat_ref_list,
    input_source_list,
    contact_loss_fn,
    contact_epochs,
    window_frames,
    gated_scales_per_source=None,
    max_batch_size=10,
):
    """
    Optimize predictions using contact loss (batched version).

    Args:
        Y_hat_ref_list: List of reference predictions, each [frames, window_frames, features]
        average_delta_rotations_hat_ref_list: List of reference average rotations, each [frames, 1, num_joints-1, 4]
        average_delta_pelvis_positions_hat_ref_list: List of reference average pelvis positions, each [frames, 1, 3]
        input_source_list: List of input source dictionaries
        contact_loss_fn: ContactLoss instance
        contact_epochs: Number of optimization epochs
        window_frames: Number of padding frames to skip
        gated_scales_per_source: List of dicts with per-source gated scales:
            - 'dynamic': Scale factor for dynamic deltas [frames, 1, 1] or [frames, joints, 1]
            - 'dynamic_pelvis': Scale factor for dynamic pelvis position [frames, 1]
            - 'static': Scale factor for static deltas [frames, 1, 1] or [frames, joints, 1]
            - 'static_pelvis': Scale factor for static pelvis position [frames, 1]
        max_batch_size: Maximum number of sequences to process in a single batch (default: 10)

    Returns:
        List of optimized predictions, each [frames, features]
    """
    if len(Y_hat_ref_list) == 0:
        return []

    device = Y_hat_ref_list[0].device

    # Process sequences in batches if there are more than max_batch_size sequences
    num_sequences = len(Y_hat_ref_list)
    if num_sequences > max_batch_size:
        all_results = []
        for batch_start in range(0, num_sequences, max_batch_size):
            batch_end = min(batch_start + max_batch_size, num_sequences)
            print(
                f"\033[97mProcessing batch {batch_start // max_batch_size + 1}/{(num_sequences + max_batch_size - 1) // max_batch_size} (sequences {batch_start}-{batch_end-1})\033[0m"
            )

            # Slice gated_scales_per_source for this batch
            gated_scales_batch = None
            if gated_scales_per_source is not None:
                gated_scales_batch = gated_scales_per_source[batch_start:batch_end]

            batch_results = optimize_contact(
                Y_hat_ref_list[batch_start:batch_end],
                average_delta_rotations_hat_ref_list[batch_start:batch_end],
                average_delta_pelvis_positions_hat_ref_list[batch_start:batch_end],
                input_source_list[batch_start:batch_end],
                contact_loss_fn,
                contact_epochs,
                window_frames,
                gated_scales_per_source=gated_scales_batch,
                max_batch_size=max_batch_size,
            )
            all_results.extend(batch_results)
        return all_results

    # Get the maximum number of frames across all sequences
    max_frames = max(Y_hat_ref.shape[0] for Y_hat_ref in Y_hat_ref_list)
    num_sequences = len(Y_hat_ref_list)
    feature_dim = Y_hat_ref_list[0].shape[-1]

    # Store original lengths for unpacking later
    original_lengths = [Y_hat_ref.shape[0] for Y_hat_ref in Y_hat_ref_list]

    # Pad sequences by repeating the last element
    Y_hat_ref_padded = []
    Y_s_pos_padded = []
    Y_s_rots_padded = []
    x_k_contacts_padded = []
    avg_rots_padded = []
    avg_pos_padded = []

    # Also pad gated scales if provided
    gated_dynamic_padded = []
    gated_dynamic_pelvis_padded = []
    gated_static_padded = []
    gated_static_pelvis_padded = []

    for i in range(num_sequences):
        Y_hat_ref = Y_hat_ref_list[i]
        input_source = input_source_list[i]
        avg_rots = average_delta_rotations_hat_ref_list[i]
        avg_pos = average_delta_pelvis_positions_hat_ref_list[i]

        num_frames = Y_hat_ref.shape[0]
        padding_needed = max_frames - num_frames

        if padding_needed > 0:
            # Repeat last frame for padding
            Y_hat_ref_pad = torch.cat(
                [Y_hat_ref, Y_hat_ref[-1:].repeat(padding_needed, 1, 1)], dim=0
            )
            Y_s_pos_pad = torch.cat(
                [
                    input_source["Y_s_pos"],
                    input_source["Y_s_pos"][-1:].repeat(padding_needed, 1, 1, 1),
                ],
                dim=0,
            )
            Y_s_rots_pad = torch.cat(
                [
                    input_source["Y_s_rots"],
                    input_source["Y_s_rots"][-1:].repeat(padding_needed, 1, 1, 1),
                ],
                dim=0,
            )
            x_k_contacts_pad = torch.cat(
                [
                    input_source["x_k_contacts"],
                    input_source["x_k_contacts"][-1:].repeat(padding_needed, 1, 1),
                ],
                dim=0,
            )
            avg_rots_pad = torch.cat(
                [avg_rots, avg_rots[-1:].repeat(padding_needed, 1, 1, 1)], dim=0
            )
            avg_pos_pad = torch.cat(
                [avg_pos, avg_pos[-1:].repeat(padding_needed, 1, 1)], dim=0
            )
        else:
            Y_hat_ref_pad = Y_hat_ref
            Y_s_pos_pad = input_source["Y_s_pos"]
            Y_s_rots_pad = input_source["Y_s_rots"]
            x_k_contacts_pad = input_source["x_k_contacts"]
            avg_rots_pad = avg_rots
            avg_pos_pad = avg_pos

        Y_hat_ref_padded.append(Y_hat_ref_pad)
        Y_s_pos_padded.append(Y_s_pos_pad)
        Y_s_rots_padded.append(Y_s_rots_pad)
        x_k_contacts_padded.append(x_k_contacts_pad)
        avg_rots_padded.append(avg_rots_pad)
        avg_pos_padded.append(avg_pos_pad)

        # Handle gated scales if provided
        if gated_scales_per_source is not None:
            scales = gated_scales_per_source[i]
            if padding_needed > 0:
                # Pad scales by repeating last frame
                gated_dynamic_padded.append(np.concatenate([
                    scales['dynamic'],
                    np.tile(scales['dynamic'][-1:], (padding_needed, 1, 1))
                ], axis=0))
                gated_dynamic_pelvis_padded.append(np.concatenate([
                    scales['dynamic_pelvis'],
                    np.tile(scales['dynamic_pelvis'][-1:], (padding_needed, 1))
                ], axis=0))
                gated_static_padded.append(np.concatenate([
                    scales['static'],
                    np.tile(scales['static'][-1:], (padding_needed, 1, 1))
                ], axis=0))
                gated_static_pelvis_padded.append(np.concatenate([
                    scales['static_pelvis'],
                    np.tile(scales['static_pelvis'][-1:], (padding_needed, 1))
                ], axis=0))
            else:
                gated_dynamic_padded.append(scales['dynamic'])
                gated_dynamic_pelvis_padded.append(scales['dynamic_pelvis'])
                gated_static_padded.append(scales['static'])
                gated_static_pelvis_padded.append(scales['static_pelvis'])

    # Stack into batched tensors [batch, frames, ...]
    Y_hat_ref_batch = torch.stack(
        Y_hat_ref_padded, dim=0
    )  # [batch, frames, window_frames, features]
    Y_s_pos_batch = torch.stack(
        Y_s_pos_padded, dim=0
    )  # [batch, frames, window_frames, num_joints, 3]
    Y_s_rots_batch = torch.stack(
        Y_s_rots_padded, dim=0
    )  # [batch, frames, window_frames, num_joints, 4]
    x_k_contacts_batch = torch.stack(
        x_k_contacts_padded, dim=0
    )  # [batch, frames, window_frames, num_feet]
    avg_rots_batch = torch.stack(
        avg_rots_padded, dim=0
    )  # [batch, frames, 1, num_joints-1, 4]
    avg_pos_batch = torch.stack(avg_pos_padded, dim=0)  # [batch, frames, 1, 3]

    # Stack gated scales into batched tensors if provided
    gated_dynamic_batch = None
    gated_dynamic_pelvis_batch = None
    gated_static_batch = None
    gated_static_pelvis_batch = None
    if gated_scales_per_source is not None and len(gated_dynamic_padded) > 0:
        gated_dynamic_batch = torch.from_numpy(np.stack(gated_dynamic_padded, axis=0)).float().to(device)
        gated_dynamic_pelvis_batch = torch.from_numpy(np.stack(gated_dynamic_pelvis_padded, axis=0)).float().to(device)
        gated_static_batch = torch.from_numpy(np.stack(gated_static_padded, axis=0)).float().to(device)
        gated_static_pelvis_batch = torch.from_numpy(np.stack(gated_static_pelvis_padded, axis=0)).float().to(device)

    best_total_contact_loss = np.inf
    best_Y_hat_optimized = None

    # Create learnable parameter from the middle portion of Y_hat_ref sequence
    Y_pred_batch = (
        Y_hat_ref_batch[:, :, window_frames, :].clone().detach()
    )  # [batch, frames, features]
    Y_pred_batch.requires_grad = True

    # Create optimizer for the batched sequence
    optimizer_Y_k = torch.optim.AdamW([Y_pred_batch], lr=1e-3)

    for epoch in range(contact_epochs):
        optimizer_Y_k.zero_grad()

        # Physics losses with pose regularization on middle sequence only
        # Pass the gated scales so optimization matches the final rendered motion
        l_total_contacts, l_contacts, l_pos_reg, l_smooth = contact_loss_fn(
            Y_pred_batch,
            Y_hat_ref_batch[:, :, window_frames],
            Y_s_pos_batch[:, :, window_frames],
            Y_s_rots_batch[:, :, window_frames],
            x_k_contacts_batch[:, :, window_frames],
            avg_rots_batch[:, :, 0],
            avg_pos_batch[:, :, 0],
            dynamic_scale=gated_dynamic_batch,
            dynamic_pelvis_scale=gated_dynamic_pelvis_batch,
            static_scale=gated_static_batch,
            static_pelvis_scale=gated_static_pelvis_batch,
            debug=False,
        )

        # Backward pass
        l_total_contacts.backward()

        # Clip gradients only if norm is large enough (avoid NaN from division by near-zero norm)
        if Y_pred_batch.grad is not None:
            grad_norm = Y_pred_batch.grad.norm()
            if grad_norm > 1e-6:
                torch.nn.utils.clip_grad_norm_([Y_pred_batch], 100.0)

        if not torch.isnan(l_total_contacts):
            optimizer_Y_k.step()
        else:
            print(f"\033[93m[WARNING] NaN loss detected in contact refinement! "
                  f"contact={l_contacts.item():.4f}, pos_reg={l_pos_reg.item():.4f}, smooth={l_smooth.item():.4f}\033[0m")
            print(f"\033[93m  Y_pred_batch has NaN: {torch.isnan(Y_pred_batch).any().item()}\033[0m")

        # Track best results
        was_best_contact = False
        if epoch == 0 or l_total_contacts.item() < best_total_contact_loss:
            best_total_contact_loss = l_total_contacts.item()
            best_Y_hat_optimized = Y_pred_batch.clone().detach()
            was_best_contact = True

        # Print progress
        if (epoch + 1) % 100 == 0:
            print(
                f"\033[93mEpoch {epoch+1}/{contact_epochs} - "
                f"\033[96mTotal: {l_total_contacts.item():.4f} - "
                f"\033[97mContact: {l_contacts.item():.4f} - "
                f"\033[97mPose Reg: {l_pos_reg.item():.4f} - "
                f"\033[97mSmooth: {l_smooth.item():.4f}"
                + ("\033[92m*" if was_best_contact else "")
                + "\033[0m"
            )

    # Use best optimized Y_k predictions
    if best_Y_hat_optimized is not None:
        Y_pred_batch = best_Y_hat_optimized
        print(
            f"\033[92mBest optimized Y_k (loss: {best_total_contact_loss:.4f})\033[0m"
        )

    # Unpack batched results back to list
    Y_pred_list = []
    for i in range(num_sequences):
        Y_pred_list.append(Y_pred_batch[i, : original_lengths[i], :])

    return Y_pred_list


def predict(
    input_sources,
    model,
    avg_model,
    average_delta_rotations,
    average_delta_pelvis_positions,
    Y_k_standardize,
    induce_x_p=None,
    induce_x_p_frequency=0.0,
    induce_x_p_speed=1.0,
    induce_x_p_strength=1.0,
    encoder_output_scale=1.0,
    gating_model=None,
):
    """
    Run neural network prediction on a list of input sources.
    Does NOT apply contact optimization - use optimize() separately.

    Args:
        input_sources: List of dicts containing input data
        model: The transformer model
        avg_model: The average pose model (classifier)
        average_delta_rotations: [chunks, joints-1, 4]
        average_delta_pelvis_positions: [chunks, 3]
        Y_k_standardize: Standardization parameters for Y_k
        induce_x_p: Induced action features (optional)
        induce_x_p_frequency: Frequency of induced actions
        induce_x_p_speed: Speed of induced actions
        induce_x_p_strength: Strength of induced actions
        encoder_output_scale: Scale factor for encoder output
        gating_model: The gating model for stylizability (optional)

    Returns:
        Tuple of lists for each input source:
            - Y_k_pred_list: Dynamic predictions [frames, window_frames, features] (torch tensors, standardized)
            - average_delta_rotations_list: Weighted rotation deltas [frames, 1, joints-1, 4] (torch tensors)
            - average_delta_pelvis_positions_list: Weighted position deltas [frames, 1, 3] (torch tensors)
            - chunk_probs_list: Chunk probabilities after softmax [frames, num_chunks] (torch tensors)
            - gating_scores_list: Gating scores [frames, 1] (torch tensors, 0=no style, 1=full style)
    """
    Y_k_pred_list = []
    average_delta_rotations_list = []
    average_delta_pelvis_positions_list = []
    chunk_probs_list = []
    gating_scores_list = []

    # Get window size from first input source
    first_input = input_sources[0]
    window_length = first_input["x_k"].shape[-2]
    half_window_length = window_length // 2

    for input_src in input_sources:
        x_k = input_src["x_k"]
        device = x_k.device

        # Get Features
        n_out_features = input_src["Y_k"].shape[-1]
        current_and_future_frames = torch.arange(half_window_length, window_length)
        x_p = torch.zeros(
            (1, window_length, n_out_features), dtype=torch.float32, device=device
        )

        # Get Predictions
        model.eval()
        Y_k_pred = []
        action_active = False
        action_frame_offset = 0.0
        with torch.no_grad():
            for i in range(x_k.shape[0]):
                x_k_i = x_k[i : i + 1]
                if induce_x_p is not None:
                    x_p, action_active, action_frame_offset = x_p_interpolate(
                        x_p,
                        induce_x_p,
                        induce_x_p_frequency,
                        induce_x_p_speed,
                        action_active,
                        action_frame_offset,
                        induce_x_p_strength,
                    )
                Y_k_pred_i = model(
                    x_k=x_k_i,
                    x_p=x_p,
                    encoder_output_scale=encoder_output_scale,
                )
                Y_k_pred.append(Y_k_pred_i.clone())
                x_p = torch.roll(x_p, shifts=-1, dims=1)
                x_p[:, current_and_future_frames, :] = Y_k_pred_i[
                    :, current_and_future_frames, :
                ]
        model.train()

        Y_k_pred = torch.cat(Y_k_pred, dim=0)

        # Predict average pose using classifier
        avg_model.eval()
        with torch.no_grad():
            # Get source rotations and compute average over window
            Y_s_rots = input_src[
                "Y_s_rots"
            ]  # [num_frames, window_length, num_joints, 4]
            avg_source_rots = Y_s_rots.mean(dim=1)  # [num_frames, num_joints, 4]

            # Exclude root joint and convert to 6D
            avg_source_rots_6d = ortho6d.from_quat(
                avg_source_rots[:, 1:, :]
            )  # [num_frames, num_joints-1, 6]

            # Classify to get chunk probabilities
            chunk_logits = avg_model(avg_source_rots_6d)  # [num_frames, num_chunks]
            chunk_probs = torch.softmax(
                chunk_logits, dim=-1
            )  # [num_frames, num_chunks]

            # Get weighted average deltas based on chunk probabilities
            weighted_delta_rots, weighted_delta_pos = get_weighted_average_deltas(
                chunk_probs,
                torch.from_numpy(average_delta_rotations).to(device),
                torch.from_numpy(average_delta_pelvis_positions).to(device),
            )
        avg_model.train()

        # Predict gating scores (stylizability)
        # Gating model always exists (either trained or always_one=True mode)
        gating_model.eval()
        with torch.no_grad():
            # Use averaged rotations over window (like static model) for consistency
            # and to remove high-frequency details
            avg_source_rots_for_gating = Y_s_rots.mean(dim=1)[:, 1:, :]  # [num_frames, num_joints-1, 4]
            avg_source_rots_6d_for_gating = ortho6d.from_quat(avg_source_rots_for_gating)  # [num_frames, num_joints-1, 6]

            # Get gating scores
            gating_scores = gating_model(avg_source_rots_6d_for_gating)  # [num_frames, 1]
        gating_model.train()

        # Store predictions for this input (keep as torch tensors)
        Y_k_pred_list.append(Y_k_pred)
        average_delta_rotations_list.append(weighted_delta_rots[..., None, :, :])
        average_delta_pelvis_positions_list.append(weighted_delta_pos[..., None, :])
        chunk_probs_list.append(chunk_probs)
        gating_scores_list.append(gating_scores)

    return (
        Y_k_pred_list,
        average_delta_rotations_list,
        average_delta_pelvis_positions_list,
        chunk_probs_list,
        gating_scores_list,
    )


def optimize(
    Y_k_pred_list,
    average_delta_rotations_list,
    average_delta_pelvis_positions_list,
    chunk_probs_list,
    input_sources,
    contact_loss_fn,
    contact_epochs,
    window_size,
    Y_k_standardize,
    gated_scales_per_source=None,
):
    """
    Apply contact optimization to predictions and convert to numpy.

    Args:
        Y_k_pred_list: List of torch tensor predictions [frames, window_frames, features]
        average_delta_rotations_list: List of rotation deltas [frames, 1, joints-1, 4]
        average_delta_pelvis_positions_list: List of position deltas [frames, 1, 3]
        chunk_probs_list: List of chunk probabilities [frames, num_chunks]
        input_sources: List of input source dictionaries
        contact_loss_fn: ContactLoss instance (or None to skip optimization)
        contact_epochs: Number of optimization epochs (0 to skip)
        window_size: Window size for extracting middle frame
        Y_k_standardize: Standardization parameters for Y_k
        gated_scales_per_source: List of dicts with per-source gated scales:
            - 'dynamic': Scale factor for dynamic deltas [frames, 1, 1] or [frames, joints, 1]
            - 'dynamic_pelvis': Scale factor for dynamic pelvis position [frames, 1]
            - 'static': Scale factor for static deltas [frames, 1, 1] or [frames, joints, 1]
            - 'static_pelvis': Scale factor for static pelvis position [frames, 1]

    Returns:
        Tuple of lists (numpy arrays, unstandardized):
            - Y_k_pred_final_list: [frames, features] (middle frame extracted)
            - average_delta_rotations_final_list: [frames, 1, joints-1, 4]
            - average_delta_pelvis_positions_final_list: [frames, 1, 3]
            - chunk_probs_final_list: [frames, num_chunks]
    """
    half_window = window_size // 2

    # Clone predictions for optimization (avoid modifying cached originals)
    Y_k_pred_working = [pred.clone() for pred in Y_k_pred_list]

    # Apply batched contact optimization if requested
    if contact_epochs > 0 and contact_loss_fn is not None:
        # Check if any sequence has enough frames
        valid_indices = [
            i for i, Y_k_pred in enumerate(Y_k_pred_working) if Y_k_pred.shape[0] > 1
        ]

        if len(valid_indices) > 0:
            # Gather valid sequences for batched optimization
            Y_k_pred_valid = [Y_k_pred_working[i] for i in valid_indices]
            avg_rots_valid = [average_delta_rotations_list[i] for i in valid_indices]
            avg_pos_valid = [
                average_delta_pelvis_positions_list[i] for i in valid_indices
            ]
            input_source_valid = [input_sources[i] for i in valid_indices]

            # Gather gated scales for valid indices
            gated_scales_valid = None
            if gated_scales_per_source is not None:
                gated_scales_valid = [gated_scales_per_source[i] for i in valid_indices]

            # Run batched optimization with gated scaling factors
            Y_k_pred_optimized_list = optimize_contact(
                Y_hat_ref_list=Y_k_pred_valid,
                average_delta_rotations_hat_ref_list=avg_rots_valid,
                average_delta_pelvis_positions_hat_ref_list=avg_pos_valid,
                input_source_list=input_source_valid,
                contact_loss_fn=contact_loss_fn,
                contact_epochs=contact_epochs,
                window_frames=half_window,
                gated_scales_per_source=gated_scales_valid,
            )

            # Replace predictions with optimized versions
            for idx, orig_idx in enumerate(valid_indices):
                Y_k_pred_working[orig_idx][:, half_window, :] = Y_k_pred_optimized_list[
                    idx
                ]

    # Convert to numpy, extract middle frame, and unstandardize
    Y_k_pred_final_list = []
    average_delta_rotations_final_list = []
    average_delta_pelvis_positions_final_list = []
    chunk_probs_final_list = []

    for i in range(len(input_sources)):
        # Extract middle frame and convert to numpy
        Y_k_pred = Y_k_pred_working[i][:, half_window, :].detach().cpu().numpy()
        Y_k_pred = Y_k_standardize.unstandardize(Y_k_pred)

        avg_delta_rots = average_delta_rotations_list[i].detach().cpu().numpy()
        avg_delta_pos = average_delta_pelvis_positions_list[i].detach().cpu().numpy()
        chunk_probs = chunk_probs_list[i].detach().cpu().numpy()

        Y_k_pred_final_list.append(Y_k_pred)
        average_delta_rotations_final_list.append(avg_delta_rots)
        average_delta_pelvis_positions_final_list.append(avg_delta_pos)
        chunk_probs_final_list.append(chunk_probs)

    return (
        Y_k_pred_final_list,
        average_delta_rotations_final_list,
        average_delta_pelvis_positions_final_list,
        chunk_probs_final_list,
    )


def x_p_interpolate(
    x_p,
    induce_x_p,
    induce_x_p_frequency,
    induce_x_p_speed,
    action_active,
    action_frame_offset,
    induce_x_p_strength,
):
    # If not currently in an action window, check if we should start one
    if not action_active:
        if torch.rand(1) < induce_x_p_frequency:
            action_active = True
            action_frame_offset = 0.0

    # If in an action window, apply interpolation
    if action_active:
        # Calculate offset into induce_x_p array based on speed parameter
        # The offset accumulates based on the speed (e.g., speed=2.0 means we advance 2 frames per iteration)
        offset_idx = int(action_frame_offset)
        induce_x_p_i = induce_x_p[offset_idx : offset_idx + 1]

        # Apply interpolation with specified strength
        result_x_p = (
            x_p * (1.0 - induce_x_p_strength) + induce_x_p_i * induce_x_p_strength
        )

        # Advance the frame offset by the speed parameter
        action_frame_offset += induce_x_p_speed

        # Check if we've finished the current window (based on the accumulated offset)
        if action_frame_offset >= induce_x_p.shape[0]:
            action_active = False

        return result_x_p, action_active, action_frame_offset
    else:
        return x_p, action_active, action_frame_offset
