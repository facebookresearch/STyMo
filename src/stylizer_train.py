# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
import argparse
import torch
import shutil
import warnings

import numpy as np
import torch.nn.functional as F
import pymotion.rotations.quat as quat

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from utils import nan_in_grad
from motion_loss import ReconstructionLoss
from model_temporal import ModelTransformer
from model_static import AveragePoseModel
from model_gating import GatingModel
from pymotion.io.bvh import BVH
from utils_training_data import (
    read_bvh_files,
    preprocess_paired_data,
    preprocess_source_input_data,
    create_average_samples,
    create_gating_samples,
    load_user_static_poses,
)


def main(
    motion_path,
    source_input,
    scale,
    epochs,
    name,
    target_frame_time,
    fix_root,
    feet_velocity_contact_th,
    window_frames=32,
    nn_confidence_th=2.5,
    batch_size=32,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\033[92mDevice: {device}\033[0m")
    delta_indices = [f for f in range(-window_frames, window_frames)]
    window_size = len(delta_indices)

    # Preprocess data ----------------------------------------------------------
    start_time = time.time()
    print("\033[93mPreprocessing...\033[0m")
    (
        source_bvhs,
        train_source_bvhs,
        train_target_bvhs,
        source_bvh_paths,
        end_joints,
        body_parts_indices,
        feet_indices,
        parents,
    ) = read_bvh_files(motion_path, source_input, scale, target_frame_time)

    (
        paired_data,
        train_dataloader,
        val_dataloader,
        x_k_standardize,
        x_p_standardize,
        Y_k_standardize,
        average_delta_rotations,
        average_delta_pelvis_positions,
        average_rotations,
        average_pelvis_positions,
    ) = preprocess_paired_data(
        train_source_bvhs,
        train_target_bvhs,
        delta_indices,
        feet_velocity_contact_th,
        batch_size,
        device,
    )
    n_in_features = paired_data["x_k"].shape[-1]
    n_out_features = paired_data["Y_k"].shape[-1]

    # Load user-provided static poses if available
    (
        user_average_rotations,
        user_average_pelvis_positions,
        user_average_delta_rotations,
        user_average_delta_pelvis_positions,
    ) = load_user_static_poses(
        motion_path,
        source_bvhs[0],
        scale,
        target_frame_time,
    )

    # Combine with automatically extracted averages if user poses exist
    if user_average_rotations is not None:
        num_auto_chunks = average_rotations.shape[0]
        num_user_chunks = user_average_rotations.shape[0]

        # Concatenate user-provided data with auto-extracted data
        average_rotations = np.concatenate(
            [average_rotations, user_average_rotations], axis=0
        )
        average_pelvis_positions = np.concatenate(
            [average_pelvis_positions, user_average_pelvis_positions], axis=0
        )
        average_delta_rotations = torch.cat(
            [
                average_delta_rotations,
                torch.from_numpy(user_average_delta_rotations).float().to(device),
            ],
            dim=0,
        )
        average_delta_pelvis_positions = torch.cat(
            [
                average_delta_pelvis_positions,
                torch.from_numpy(user_average_delta_pelvis_positions)
                .float()
                .to(device),
            ],
            dim=0,
        )

        print(
            f"\033[92mCombining {num_auto_chunks} auto + "
            f"{num_user_chunks} user chunks = {average_rotations.shape[0]} total\033[0m"
        )

    # Create average dataloader (after combining with user poses if any)
    average_dataloader = create_average_samples(
        average_rotations,
        batch_size,
        device,
    )

    input_sources, input_sources_all = preprocess_source_input_data(
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
    )

    print(f"\033[38;5;228mPreprocessing: {time.time() - start_time:.2f}s\033[0m")

    # Static Model: Train AveragePoseModel ------------------------------
    print("\033[93mStatic Model: Training AveragePoseModel\033[0m")

    # Calculate input and output dimensions
    num_joints = average_rotations.shape[1]  # Number of joints (includes root)
    num_chunks = average_rotations.shape[0]  # Number of chunks
    avg_input_dim = (num_joints - 1) * 6  # Flattened 6D rotations (root excluded)
    avg_output_dim = num_chunks  # Number of classes for classification

    avg_pose_model = AveragePoseModel(
        input_dim=avg_input_dim,
        output_dim=avg_output_dim,
    ).to(device)
    model = ModelTransformer(
        input_dim=n_in_features,
        output_dim=n_out_features,
        window_size=window_size,
    ).to(device)

    # Create experiment directory and TensorBoard writer
    experiment_dir = os.path.join(motion_path, "results", f"Transformer-{name}")

    if os.path.exists(experiment_dir):
        shutil.rmtree(experiment_dir)
    os.makedirs(experiment_dir, exist_ok=True)

    # Tensorboard
    tensorboard_dir = os.path.join(experiment_dir, "tensorboard")
    os.makedirs(tensorboard_dir, exist_ok=True)
    writer = SummaryWriter(
        os.path.join(tensorboard_dir, datetime.now().strftime("%b%d_%H-%M-%S"))
    )

    start_time_static = time.time()

    avg_optimizer = torch.optim.AdamW(avg_pose_model.parameters(), lr=1e-3)
    avg_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        avg_optimizer, T_0=50, T_mult=1, eta_min=1e-5
    )

    avg_epochs = 100
    best_avg_loss = np.inf
    best_avg_weights = None
    best_avg_accuracy = 0.0

    for epoch in range(avg_epochs):
        avg_pose_model.train()
        train_avg_loss = 0.0
        train_avg_accuracy = 0.0
        num_samples = 0

        for avg_rots_i, chunk_one_hot_i in average_dataloader:
            avg_optimizer.zero_grad()

            # Forward pass - returns logits [batch, num_chunks]
            logits_i = avg_pose_model(avg_rots_i)

            # Convert one-hot to class indices for CrossEntropyLoss
            chunk_labels_i = torch.argmax(chunk_one_hot_i, dim=1)

            # CrossEntropyLoss for classification
            loss = F.cross_entropy(logits_i, chunk_labels_i)
            loss.backward()

            # Calculate accuracy
            predictions = torch.argmax(logits_i, dim=1)
            correct = (predictions == chunk_labels_i).sum().item()
            train_avg_accuracy += correct
            num_samples += chunk_labels_i.size(0)

            # Clip gradients and update
            torch.nn.utils.clip_grad_norm_(avg_pose_model.parameters(), 100.0)

            if not (torch.isnan(loss) or nan_in_grad(avg_optimizer)):
                avg_optimizer.step()
            else:
                print("\033[93mNaN loss in AveragePoseModel!\033[0m")

            train_avg_loss += loss.item()

        avg_scheduler.step()
        train_avg_loss /= len(average_dataloader)
        train_avg_accuracy = train_avg_accuracy / num_samples

        # Log to TensorBoard
        writer.add_scalar("Static_Model/l_avg_pose_train", train_avg_loss, epoch)
        writer.add_scalar("Static_Model/accuracy_train", train_avg_accuracy, epoch)
        writer.add_scalar(
            "Static_Model/Learning_rate", avg_scheduler.get_last_lr()[0], epoch
        )

        # Track best model (prefer by accuracy, then by loss)
        was_best = False
        if (
            epoch == 0
            or train_avg_accuracy > best_avg_accuracy
            or (
                train_avg_accuracy == best_avg_accuracy
                and train_avg_loss < best_avg_loss
            )
        ):
            best_avg_loss = train_avg_loss
            best_avg_accuracy = train_avg_accuracy
            best_avg_weights = avg_pose_model.state_dict()
            was_best = True

        # Print progress
        if was_best or (epoch + 1) % 100 == 0:
            print(
                f"\033[95mStatic Model Epoch {epoch+1}/{avg_epochs} - "
                f"\033[97mTrain Loss: {train_avg_loss:.6f} - "
                f"\033[97mTrain Acc: {train_avg_accuracy:.2f} - "
                f"\033[94mlr: {avg_scheduler.get_last_lr()[0]:.6f}"
                + ("\033[92m*" if was_best else "")
                + "\033[0m"
            )

    # Load best model weights
    if best_avg_weights is not None:
        avg_pose_model.load_state_dict(best_avg_weights)
        print(f"\033[93mBest AveragePoseModel: loss={best_avg_loss:.6f}\033[0m")

    end_time_static = time.time()
    print(
        f"\033[38;5;228mStatic Model: {end_time_static - start_time_static:.2f}s\033[0m"
    )

    # Gating Model: Train GatingModel (Stylizability Gate) ---------------------
    print("\033[93mGating Model: Training GatingModel (Stylizability Gate)\033[0m")

    # Create gating training data
    gating_dataloader = create_gating_samples(
        paired_data=paired_data,
        input_sources_all=input_sources_all,
        x_k=paired_data["x_k"],
        batch_size=batch_size,
        nn_confidence_th=nn_confidence_th,
        device=device,
    )

    start_time_gating = time.time()

    if gating_dataloader is not None:
        # Initialize gating model (same input dim as avg_pose_model)
        gating_model = GatingModel(
            input_dim=avg_input_dim,
        ).to(device)

        gating_optimizer = torch.optim.AdamW(gating_model.parameters(), lr=1e-3)
        gating_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            gating_optimizer, T_0=50, T_mult=1, eta_min=1e-5
        )

        gating_epochs = 30
        best_gating_loss = np.inf
        best_gating_weights = None
        best_gating_accuracy = 0.0

        for epoch in range(gating_epochs):
            gating_model.train()
            train_gating_loss = 0.0
            train_gating_accuracy = 0.0
            num_samples = 0

            for rots_6d_i, labels_i in gating_dataloader:
                gating_optimizer.zero_grad()

                # Forward pass - returns sigmoid output [batch, 1]
                pred_i = gating_model(rots_6d_i)

                # Binary cross-entropy loss
                loss = F.binary_cross_entropy(pred_i, labels_i)
                loss.backward()

                # Calculate accuracy
                predictions = (pred_i > 0.5).float()
                correct = (predictions == labels_i).sum().item()
                train_gating_accuracy += correct
                num_samples += labels_i.size(0)

                # Clip gradients and update
                torch.nn.utils.clip_grad_norm_(gating_model.parameters(), 100.0)

                if not (torch.isnan(loss) or nan_in_grad(gating_optimizer)):
                    gating_optimizer.step()
                else:
                    print("\033[93mNaN loss in GatingModel!\033[0m")

                train_gating_loss += loss.item()

            gating_scheduler.step()
            train_gating_loss /= len(gating_dataloader)
            train_gating_accuracy = train_gating_accuracy / num_samples

            # Log to TensorBoard
            writer.add_scalar("Gating_Model/l_gating_train", train_gating_loss, epoch)
            writer.add_scalar("Gating_Model/accuracy_train", train_gating_accuracy, epoch)
            writer.add_scalar(
                "Gating_Model/Learning_rate", gating_scheduler.get_last_lr()[0], epoch
            )

            # Track best model (prefer by accuracy, then by loss)
            was_best = False
            if (
                epoch == 0
                or train_gating_accuracy > best_gating_accuracy
                or (
                    train_gating_accuracy == best_gating_accuracy
                    and train_gating_loss < best_gating_loss
                )
            ):
                best_gating_loss = train_gating_loss
                best_gating_accuracy = train_gating_accuracy
                best_gating_weights = gating_model.state_dict()
                was_best = True

            # Print progress
            if was_best or (epoch + 1) % 100 == 0:
                print(
                    f"\033[94mGating Model Epoch {epoch+1}/{gating_epochs} - "
                    f"\033[97mTrain Loss: {train_gating_loss:.6f} - "
                    f"\033[97mTrain Acc: {train_gating_accuracy:.2%} - "
                    f"\033[94mlr: {gating_scheduler.get_last_lr()[0]:.6f}"
                    + ("\033[92m*" if was_best else "")
                    + "\033[0m"
                )

        # Load best model weights
        if best_gating_weights is not None:
            gating_model.load_state_dict(best_gating_weights)
            print(
                f"\033[93mBest GatingModel: loss={best_gating_loss:.6f}, "
                f"acc={best_gating_accuracy:.2%}\033[0m"
            )
    else:
        # No gating training data - create a gating model that always outputs 1
        gating_model = GatingModel(
            input_dim=avg_input_dim,
            always_one=True,
        ).to(device)
        print("\033[93mNo gating training data - using always_one=True gating model\033[0m")

    end_time_gating = time.time()
    print(
        f"\033[38;5;228mGating Model: {end_time_gating - start_time_gating:.2f}s\033[0m"
    )

    # Temporal Model: Main training on paired data (reconstruction loss)
    print(
        f"\033[93mTemporal Model: Training for {epochs} epochs\033[0m"
    )
    start_time_temporal = time.time()

    best_loss = np.inf
    best_weights = None

    # Create separate loss functions
    reconstruction_loss = ReconstructionLoss(source_bvh=source_bvhs[0])

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=1, eta_min=1e-5
    )

    for epoch in range(epochs):
        model.train()
        train_rec_loss = 0.0

        for (
            x_k_i,
            x_p_i,
            Y_i,
            Y_s_pos_i,
            Y_s_rots_i,
            x_k_contacts_i,
        ) in train_dataloader:
            optimizer.zero_grad()

            # Add noise to x_k_i
            x_k_i = x_k_i + torch.randn_like(x_k_i) * 0.001
            # Data augmentation: scale
            min_scale = 0.9
            max_scale = 1.1
            scales = (
                torch.rand(x_k_i.shape[0], 1, 1, device=device)
                * (max_scale - min_scale)
                + min_scale
            )
            x_k_i[..., :-2] = x_k_i[..., :-2] * scales

            batch_size_curr = x_k_i.shape[0]

            # Calculate model parameters
            seq_len = x_p_i.shape[1]
            n_heads = model.n_heads

            # Create attention masks
            prob_src_mask = 0.1
            mask = (
                torch.rand(
                    batch_size_curr * n_heads,
                    seq_len,
                    seq_len,
                    device=device,
                )
                < prob_src_mask
            )

            # Forward pass
            Y_hat_i = model(
                x_k=x_k_i,
                x_p=x_p_i,
                src_mask=mask,
                memory_mask=mask,
            )

            # Reconstruction loss
            l_rec = reconstruction_loss(Y_hat_i, Y_i)
            l_rec.backward()

            # Clip gradients and update
            torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)

            if not (torch.isnan(l_rec) or nan_in_grad(optimizer)):
                optimizer.step()
            else:
                print("\033[93mNaN loss in main training!\033[0m")

            train_rec_loss += l_rec.item()

        scheduler.step()

        train_rec_loss /= len(train_dataloader)

        # Validation phase
        model.eval()
        val_rec_loss = 0.0
        with torch.no_grad():
            for (
                x_k_i,
                x_p_i,
                Y_i,
                Y_s_pos_i,
                Y_s_rots_i,
                x_k_contacts_i,
            ) in val_dataloader:
                # Forward pass
                Y_hat_i = model(
                    x_k_i,
                    x_p_i,
                )

                # Reconstruction loss
                l_rec = reconstruction_loss(Y_hat_i, Y_i)
                val_rec_loss += l_rec.item()

            val_rec_loss /= len(val_dataloader)
            val_loss = val_rec_loss

        # Log metrics to TensorBoard
        writer.add_scalar("Dynamic_Model/l_train_rec", train_rec_loss, epoch)
        writer.add_scalar("Dynamic_Model/l_val_rec", val_rec_loss, epoch)
        writer.add_scalar(
            "Dynamic_Model/Learning_rate", scheduler.get_last_lr()[0], epoch
        )

        # Track best model
        was_best = False
        if epoch == 0 or val_loss < best_loss:
            best_loss = val_loss
            best_weights = model.state_dict()
            was_best = True

        # Print with colored output and best model indicator
        if was_best or (epoch + 1) % 100 == 0:
            print(
                f"\033[96mEpoch {epoch+1}/{epochs} - \033[97mTrain Rec: {train_rec_loss:.4f} - "
                f"\033[97mVal Rec: {val_loss:.4f} - "
                f"\033[94mlr: {scheduler.get_last_lr()[0]:.6f}"
                + ("\033[92m*" if was_best else "")
                + "\033[0m"
            )

    # Load best model weights
    if "best_weights" in locals():
        model.load_state_dict(best_weights)
        print("\033[93mBest Temporal Model loaded\033[0m")

    end_time_temporal = time.time()

    # Close TensorBoard writer
    writer.close()

    end_time_training = time.time()

    print(
        f"\033[38;5;228mTotal training: {end_time_training - start_time_static:.2f}s\033[0m"
    )
    print(
        f"\033[38;5;228mStatic Model: {end_time_static - start_time_static:.2f}s\033[0m"
    )
    print(
        f"\033[38;5;228mGating Model: {end_time_gating - start_time_gating:.2f}s\033[0m"
    )
    print(
        f"\033[38;5;228mTemporal Model: {end_time_temporal - start_time_temporal:.2f}s\033[0m"
    )

    # Save model ---------------------------------------------------------------
    print("\033[93mSaving model...\033[0m")
    start_time = time.time()

    saving_dict = {
        "model": model.state_dict(),
        "avg_pose_model": avg_pose_model.state_dict(),
        "gating_model": gating_model.state_dict(),
        "gating_model_always_one": gating_model.always_one,
        "delta_indices": delta_indices,
        "x_k_standardize": x_k_standardize,
        "x_p_standardize": x_p_standardize,
        "Y_k_standardize": Y_k_standardize,
        "target_frame_time": target_frame_time,
        "body_parts_indices": body_parts_indices,
        "n_in_features": n_in_features,
        "n_out_features": n_out_features,
        "n_in_features_avg_pose": avg_pose_model.input_dim,
        "n_out_features_avg_pose": avg_pose_model.output_dim,
        "n_in_features_gating": gating_model.input_dim,
        "x_p": paired_data["x_p_no_aug"],
        "feet_velocity_contact_th": feet_velocity_contact_th,
    }
    torch.save(saving_dict, os.path.join(experiment_dir, "model.pt"))

    # Save source and target (training pairs)
    source_experiment_dir = os.path.join(experiment_dir, "source")
    os.makedirs(source_experiment_dir, exist_ok=True)
    target_experiment_dir = os.path.join(experiment_dir, "target")
    os.makedirs(target_experiment_dir, exist_ok=True)
    for i, (source_bvh_i, target_bvh_i) in enumerate(
        zip(train_source_bvhs, train_target_bvhs)
    ):
        source_bvh_i.save(os.path.join(source_experiment_dir, f"{i}.bvh"))
        target_bvh_i.save(os.path.join(target_experiment_dir, f"{i}.bvh"))

    # Save average training data
    avg_source_experiment_dir = os.path.join(experiment_dir, "sourceAvg")
    os.makedirs(avg_source_experiment_dir, exist_ok=True)
    avg_target_experiment_dir = os.path.join(experiment_dir, "targetAvg")
    os.makedirs(avg_target_experiment_dir, exist_ok=True)
    for i, (
        avg_rotations_i,
        avg_pelvis_pos_i,
        avg_delta_rotations_i,
        avg_delta_pelvis_post_i,
    ) in enumerate(
        zip(
            average_rotations,
            average_pelvis_positions,
            average_delta_rotations,
            average_delta_pelvis_positions,
        )
    ):
        avg_bvh = BVH()
        avg_bvh.copy(source_bvhs[0])
        avg_local_rotations, avg_local_positions, _, _, _, _ = avg_bvh.get_data()
        # source
        avg_local_rotations = avg_rotations_i[None, ...]
        avg_local_rotations[0, 0] = np.array([1.0, 0.0, 0.0, 0.0])
        avg_local_positions = avg_local_positions[0:1]
        avg_local_positions[0, 0] = np.array([0.0, 0.0, 0.0])
        avg_local_positions[0, 1] = avg_pelvis_pos_i
        avg_bvh.set_data(avg_local_rotations, avg_local_positions)
        avg_bvh.save(os.path.join(avg_source_experiment_dir, f"{i}.bvh"))
        # target
        avg_local_rotations[0, 1:] = quat.mul(
            avg_rotations_i[1:], avg_delta_rotations_i.cpu().numpy()
        )
        avg_local_positions[0, 1] = (
            avg_pelvis_pos_i + avg_delta_pelvis_post_i.cpu().numpy()
        )
        avg_bvh.set_data(avg_local_rotations, avg_local_positions)
        avg_bvh.save(os.path.join(avg_target_experiment_dir, f"{i}.bvh"))

    # Save input source sequences
    input_experiment_dir = os.path.join(experiment_dir, "input")
    os.makedirs(input_experiment_dir, exist_ok=True)
    for source_bvh_i in source_bvhs:
        source_bvh_i.save(
            os.path.join(input_experiment_dir, f"{source_bvh_i.data['filename']}.bvh")
        )

    print(f"\033[38;5;228mSaving: {time.time() - start_time:.2f}s\033[0m")


if __name__ == "__main__":
    warnings.filterwarnings(
        "ignore", category=FutureWarning, message=".*torch.load.*weights_only.*"
    )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "motion_path",
        type=str,
    )
    parser.add_argument(
        "source_input",
        type=str,
    )
    parser.add_argument(
        "--scale", type=float, default=1.0
    )  # 0.01 for xsens DB, 1 for mocha DB
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--name",
        type=str,
        default="test",
        help="Name of the experiment. It will be added to the output file name.",
    )
    parser.add_argument(
        "--target_frame_time",
        type=float,
        default=1.0 / 60.0,
        help="All input and output motion will be retargeted to this frame time. Default is 1/60 (60 fps).",
    )
    parser.add_argument(
        "--window_frames",
        type=int,
        default=32,
        help="Number of frames to use for the prediction window. Default is 32.",
    )
    parser.add_argument(
        "--fix_root",
        action="store_true",
        default=False,
        help="If set, the root will be fixed to the source motion.",
    )
    parser.add_argument(
        "--nn_confidence_th",
        type=str,
        default="2.5",
        help="Nearest neighbor confidence threshold. Can be a number (e.g., 2.5) or 'iterative' for interactive mode. "
             "nn_confidence_th is multiplied by the IQR of the NN distances to find the threshold for positive/negative examples. "
             "Use 'iterative' to visualize and adjust the threshold interactively.",
    )
    parser.add_argument(
        "--feet_velocity_contact_th",
        type=float,
        default=0.25,
        help="Threshold for the feet velocity to determine contact. Default is 0.1.",
    )
    args = parser.parse_args()

    name = args.name.replace("-", "")

    # Parse nn_confidence_th - can be "iterative" or a float value
    nn_confidence_th = args.nn_confidence_th
    if nn_confidence_th != "iterative":
        try:
            nn_confidence_th = float(nn_confidence_th)
        except ValueError:
            print(f"\033[91mInvalid nn_confidence_th value: {nn_confidence_th}. Using default 2.5\033[0m")
            nn_confidence_th = 2.5

    main(
        args.motion_path,
        args.source_input,
        args.scale,
        args.epochs,
        name,
        args.target_frame_time,
        args.fix_root,
        args.feet_velocity_contact_th,
        nn_confidence_th=nn_confidence_th,
        window_frames=args.window_frames,
    )
