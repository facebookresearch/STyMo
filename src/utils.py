# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import math
import torch
import torch.nn as nn
import numpy as np
import pymotion.rotations.quat as quat
from pymotion.io.bvh import BVH


def get_bvh_data(
    motion_path,
    source_input,
    scale=1.0,
    target_frame_time=1.0 / 120.0,
    extra_end_joints_indices=[],
):
    if source_input.endswith(".bvh"):
        source_paths = [source_input]
    else:
        source_paths = []
        for file in os.listdir(source_input):
            if file.endswith(".bvh"):
                source_paths.append(os.path.join(source_input, file))
        if len(source_paths) == 0:
            raise FileNotFoundError("No .bvh file found in {}".format(source_input))
    source_bvhs = []
    for i, source_path in enumerate(source_paths):
        source_bvh = BVH()
        source_bvh.load(source_path)
        source_bvh.set_scale(scale)
        if i == 0:
            original_number_frames = source_bvh.data["rotations"].shape[0]
            original_frame_time = source_bvh.data["frame_time"]
            end_joints_idx = (
                list(map(int, source_bvh.data["end_sites_parents"]))
                + extra_end_joints_indices
            )
            end_joints = list(
                map(str, [source_bvh.data["names"][idx] for idx in end_joints_idx])
            )
        source_bvh = resample_bvh(source_bvh, target_frame_time)
        source_bvhs.append(source_bvh)

    source_train_path = os.path.join(motion_path, "source")
    target_train_path = os.path.join(motion_path, "target")
    if not os.path.exists(source_train_path):
        os.makedirs(source_train_path)
    if not os.path.exists(target_train_path):
        os.makedirs(target_train_path)

    source_names = []
    for file in os.listdir(source_train_path):
        if file.endswith(".bvh"):
            source_names.append(file)
    source_names.sort()

    target_names = []
    for file in os.listdir(target_train_path):
        if file.endswith(".bvh"):
            target_names.append(file)
    target_names.sort()

    # make sure same number of files and names
    if len(source_names) != len(target_names):
        raise ValueError("Number of source and target files do not match")
    if len(source_names) == 0:
        raise ValueError("No source or target files found")
    for s, t in zip(source_names, target_names):
        if s != t:
            raise ValueError("Source and target files do not match")

    source_train_bvhs = []
    source_train_bvhs_path = []
    target_train_bvhs = []
    target_train_bvhs_path = []
    source_original_number_frames = []
    for s, t in zip(source_names, target_names):
        # Source
        source_train_bvh = BVH()
        s_path = os.path.join(source_train_path, s)
        source_train_bvhs_path.append(s_path)
        source_train_bvh.load(s_path)
        source_train_bvh.set_scale(scale)
        source_original_number_frames.append(
            source_train_bvh.data["rotations"].shape[0]
        )
        source_train_bvh = resample_bvh(source_train_bvh, target_frame_time)
        source_train_bvhs.append(source_train_bvh)
        # Target
        target_train_bvh = BVH()
        t_path = os.path.join(target_train_path, t)
        target_train_bvhs_path.append(t_path)
        target_train_bvh.load(t_path)
        target_train_bvh.set_scale(scale)
        target_train_bvh = resample_bvh(target_train_bvh, target_frame_time)
        target_train_bvhs.append(target_train_bvh)

    print(f"Loaded {len(source_paths)} input file(s), {len(source_train_bvhs_path)} training pair(s)")

    return (
        source_bvhs,
        source_train_bvhs,
        target_train_bvhs,
        source_paths,
        source_train_bvhs_path,
        target_train_bvhs_path,
        original_number_frames,
        original_frame_time,
        source_original_number_frames,
        end_joints,
    )


def second_order_forward_diff(x, x1, x2, x3):
    """
    Second order forward difference
    f''(x) = (2f(x) - 5f(x1) + 4f(x2) - f(x3)) / delta^3
    """
    # ignore constant factor 1/delta^3
    return 2 * x - 5 * x1 + 4 * x2 - x3


def second_order_backward_diff(xm3, xm2, xm1, x):
    """
    Second order backward difference
    f''(x) = (2f(x) - 5f(xm1) + 4f(xm2) - f(xm3)) / delta^3
    """
    # ignore constant factor 1/delta^3
    return 2 * x - 5 * xm1 + 4 * xm2 - xm3


def fourth_order_central_diff(xm2, xm1, x, x1, x2):
    """
    Fourth order central difference
    f''(x) = (-f(xm2) + 16f(xm1) - 30f(x) + 16f(x1) - f(x2)) / 12(delta)^2
    """
    # ignore constant factor 1/12(delta)^2
    return -xm2 + 16 * xm1 - 30 * x + 16 * x1 - x2


def get_window_length(window_seconds, frame_time, polyorder):
    fps = int(round(1.0 / frame_time))
    window_length = int(round(window_seconds * fps))
    window_length += 1 - (window_length % 2)  # make it odd
    if window_length <= polyorder:
        window_length += 2
    return window_length


def resample_bvh(bvh, target_frame_time):
    frame_time = bvh.data["frame_time"]
    local_rotations, local_positions, _, _, _, _ = bvh.get_data()
    if int(round(1.0 / frame_time)) // 2 == int(round(1.0 / target_frame_time)):
        # 2x fps to x fps
        local_rotations = local_rotations[::2, ...]
        local_positions = local_positions[::2, ...]
        bvh.set_data(local_rotations, local_positions)
        bvh.data["frame_time"] = target_frame_time
        frame_time = target_frame_time
    elif int(round(1.0 / frame_time)) == int(round(1.0 / target_frame_time)):
        bvh.data["frame_time"] = target_frame_time
        frame_time = target_frame_time
    else:
        num_frames = local_rotations.shape[0]
        num_frames_target = int(round(num_frames * frame_time / target_frame_time))
        samples = np.linspace(0, num_frames - 1, num_frames_target)
        local_rotations_shape = local_rotations.shape
        reshaped_rotations = local_rotations.reshape(local_rotations_shape[0], -1)
        interpolated = np.zeros((len(samples), reshaped_rotations.shape[1]))
        for i in range(reshaped_rotations.shape[1]):
            interpolated[:, i] = np.interp(
                x=samples,
                xp=np.arange(num_frames),
                fp=reshaped_rotations[:, i],
                left=reshaped_rotations[-1, i],
                right=reshaped_rotations[0, i],
            )
        local_rotations = quat.normalize(interpolated).reshape(
            len(samples), *local_rotations_shape[1:]
        )
        local_positions_shape = local_positions.shape
        reshaped_positions = local_positions.reshape(local_positions_shape[0], -1)
        interpolated_positions = np.zeros((len(samples), reshaped_positions.shape[1]))
        for i in range(reshaped_positions.shape[1]):
            interpolated_positions[:, i] = np.interp(
                x=samples,
                xp=np.arange(num_frames),
                fp=reshaped_positions[:, i],
                left=reshaped_positions[-1, i],
                right=reshaped_positions[0, i],
            )
        local_positions = interpolated_positions.reshape(
            len(samples), *local_positions_shape[1:]
        )
        bvh.set_data(local_rotations, local_positions)
        bvh.data["frame_time"] = target_frame_time
        frame_time = target_frame_time
    return bvh


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def add_zero_conv(module, channels):
    """
    Wrap a module with a zero-initialized convolution.
    """
    return torch.nn.Sequential(
        module,
        zero_module(
            torch.nn.Conv1d(channels, channels, kernel_size=1, stride=1, padding="same")
        ),
    )


def add_zero_fc(module, channels):
    """
    Wrap a module with a zero-initialized convolution.
    """
    return torch.nn.Sequential(
        module,
        zero_module(torch.nn.Linear(channels, channels)),
    )


def split_train_val(x, train_split=0.8):
    indices = torch.randperm(x.shape[0])
    train_size = int(x.shape[0] * train_split)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    x_train, x_val = split_train_val_indices(x, train_indices, val_indices)

    return x_train, x_val, train_indices, val_indices


def split_train_val_indices(x, train_indices, val_indices):
    x_train = x[train_indices].clone()
    x_val = x[val_indices].clone()

    return x_train, x_val


def merge_bvhs(bvh1, bvh2):
    """
    Merge two BVHs into a single BVH.
    The first BVH is augmented with the data from the second BVH.
    """
    if np.any(bvh1.data["names"] != bvh2.data["names"]):
        raise ValueError("Names must match")
    if np.any(bvh1.data["parents"] != bvh2.data["parents"]):
        raise ValueError("Parents must match")
    if np.any(bvh1.data["rot_order"] != bvh2.data["rot_order"]):
        raise ValueError("Rotations orders must match")

    bvh1.data["rotations"] = np.concatenate(
        [bvh1.data["rotations"], bvh2.data["rotations"]], axis=0
    )
    bvh1.data["positions"] = np.concatenate(
        [bvh1.data["positions"], bvh2.data["positions"]], axis=0
    )


def nan_in_grad(model: torch.optim.Optimizer):
    for group in model.param_groups:
        for p in group["params"]:
            if p.requires_grad and p.grad is not None and torch.isnan(p.grad).any():
                return True
    return False

def sample_colors(n_samples, saturation=1.0, value=1.0):
    # Generate evenly spaced hues in HSV color space, then convert to RGB
    hues = np.linspace(0, 1, n_samples, endpoint=False)
    colors = []
    for h in hues:
        # Convert HSV to RGB with configurable saturation and value
        h_prime = h * 6
        c = value * saturation
        x = c * (1 - abs(h_prime % 2 - 1))
        m = value - c

        if h_prime < 1:
            r, g, b = c, x, 0
        elif h_prime < 2:
            r, g, b = x, c, 0
        elif h_prime < 3:
            r, g, b = 0, c, x
        elif h_prime < 4:
            r, g, b = 0, x, c
        elif h_prime < 5:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x

        # Add m to each component to account for value parameter
        colors.append(np.array([r + m, g + m, b + m]))
    return colors


class SinusoidalPositionalEncoding(nn.Module):
    """
    Embedding[position, 2k] = sin(position / (n^(2k / d_model)))
    Embedding[position, 2k+1] = cos(position / (n^(2k / d_model)))
    where
        'position' is the position in the sequence
        'k' is the dimension index within the embedding vector
        'd_model' is the dimension of the embedding vector
        'n' increases the frequency of the oscillation, typically set to 10000
    """

    def __init__(self, max_length: int, n: int = 10000, d_model: int = 512) -> None:
        assert d_model % 2 == 0, "d_model must be an even number"
        super(SinusoidalPositionalEncoding, self).__init__()

        position = torch.arange(end=max_length).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(n) / d_model))
        positional_encoding = torch.zeros(max_length, d_model)
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("positional_encoding", positional_encoding)

    def forward_timestep(self, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        :param t: Time step, shape (batch_size,)
        """
        return self.positional_encoding[t, :]

    def forward_temporality(self, m: int) -> torch.Tensor:
        """
        Forward pass
        :param m: Number of time steps
        """
        return self.positional_encoding[:m, :]


class FiLM(nn.Module):
    def __init__(self, in_channels: int, features_dim: int) -> None:
        super(FiLM, self).__init__()
        self.film = nn.Sequential(
            nn.Linear(in_channels, 2 * features_dim, bias=False),
            nn.ReLU(),
            nn.Linear(2 * features_dim, 2 * features_dim, bias=False),
        )

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass
        :param x: Input tensor, shape (..., features_dim)
        :param c: Conditioning tensor, shape (in_channels,)
        """
        gamma, beta = self.film(c).chunk(2, dim=-1)
        return gamma * x + beta
