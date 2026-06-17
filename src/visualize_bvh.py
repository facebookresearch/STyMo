# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import argparse
import tkinter as tk
from tkinter import filedialog

import numpy as np

from pymotion.io.bvh import BVH
from pymotion.render.blender import BlenderConnection

colors = [
    np.array([0.8, 0, 0]),
    np.array([0, 0.8, 0]),
    np.array([0, 0, 0.8]),
    np.array([0.8, 0.8, 0]),
    np.array([0, 0.8, 0.8]),
    np.array([0.8, 0, 0.8]),
    np.array([0.8, 0.8, 0.8]),
    np.array([0.8, 0.4, 0]),
    np.array([0, 0.8, 0.4]),
    np.array([0.4, 0, 0.8]),
    np.array([0.8, 0.4, 0.4]),
    np.array([0.4, 0.8, 0.8]),
    np.array([0.8, 0.8, 0.4]),
    np.array([0.4, 0.4, 0.8]),
    np.array([0.8, 0.4, 0.8]),
    np.array([0.4, 0.8, 0.4]),
    np.array([0.4, 0.4, 0.4]),
    np.array([0.8, 0.2, 0]),
    np.array([0, 0.8, 0.2]),
    np.array([0.2, 0, 0.8]),
    np.array([0.8, 0.2, 0.2]),
    np.array([0.2, 0.8, 0.8]),
    np.array([0.8, 0.8, 0.2]),
    np.array([0.2, 0.2, 0.8]),
    np.array([0.8, 0.2, 0.8]),
    np.array([0.2, 0.8, 0.2]),
    np.array([0.2, 0.2, 0.2]),
]

color_idx = 0


def blend_towards_white(color, factor):
    """Blends color towards white by the given factor (0 = original, 1 = white)."""
    white = np.array([1.0, 1.0, 1.0])
    return color + (white - color) * factor


def create_delayed_bvh(bvh, frames_to_delay):
    """Creates a copy of the BVH with the first frame repeated frames_to_delay times at the start."""
    delayed_bvh = BVH()
    delayed_bvh.copy(bvh)

    first_rot = delayed_bvh.data["rotations"][0:1]
    first_pos = delayed_bvh.data["positions"][0:1]

    repeated_rots = np.repeat(first_rot, frames_to_delay, axis=0)
    repeated_pos = np.repeat(first_pos, frames_to_delay, axis=0)

    delayed_bvh.data["rotations"] = np.concatenate(
        [repeated_rots, delayed_bvh.data["rotations"]], axis=0
    )
    delayed_bvh.data["positions"] = np.concatenate(
        [repeated_pos, delayed_bvh.data["positions"]], axis=0
    )

    return delayed_bvh


def get_end_joints(bvh):
    """Extract end joints from BVH."""
    end_joints_idx = list(map(int, bvh.data["end_sites_parents"])) + [0]
    end_joints = list(map(str, [bvh.data["names"][idx] for idx in end_joints_idx]))
    twist_joints = [
        str(joint)
        for joint in bvh.data["names"]
        if "twist" in joint or "twist" in joint.lower()
    ]
    end_joints += twist_joints
    return end_joints


def render_bvh_with_temporal(blender_connection, bvh, color, end_joints, num_temporal_frames=1, frames_between=30):
    """Render BVH with temporal ghost frames (delayed versions blending towards white).

    Args:
        blender_connection: BlenderConnection instance
        bvh: BVH object to render
        color: Base color for the main BVH
        end_joints: List of end joint names
        num_temporal_frames: Number of temporal ghost frames to render
        frames_between: Number of animation frames between each ghost
    """
    blender_connection.render_bvh(bvh, color, end_joints)

    if num_temporal_frames > 1:
        total_frames = bvh.data["rotations"].shape[0]

        for i in range(1, num_temporal_frames):
            frames_to_delay = i * frames_between

            if frames_to_delay > 0 and frames_to_delay < total_frames:
                delayed_bvh = create_delayed_bvh(bvh, frames_to_delay)
                t = i / (num_temporal_frames - 1) if num_temporal_frames > 1 else 0
                blend_factor = 0.3 * (t ** 2)
                delayed_color = blend_towards_white(color, blend_factor)
                blender_connection.render_bvh(delayed_bvh, delayed_color, end_joints)


def parse_color(color_input):
    """Parse color from string like '1, 0, 0' to numpy array."""
    if not color_input:
        return np.array([1.0, 0.0, 0.0])
    try:
        parts = [float(x.strip()) for x in color_input.split(",")]
        if len(parts) != 3:
            print("Invalid color format. Using default red.")
            return np.array([1.0, 0.0, 0.0])
        return np.array(parts)
    except ValueError:
        print("Invalid color format. Using default red.")
        return np.array([1.0, 0.0, 0.0])


def render_floor(blender_connection):
    """Render a checkerboard floor."""
    blender_connection.render_checkerboard_floor()


def manual_mode():
    """Interactive manual mode for adding BVH files to the scene."""
    blender_connection = BlenderConnection()

    blender_connection.clear_scene()
    blender_connection.setup_rendering()
    render_floor(blender_connection)
    print("Scene initialized.")

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    while True:
        print("\n" + "=" * 40)
        print("Commands:")
        print("  clear   - Clear and setup scene")
        print("  add bvh - Add a BVH file to scene")
        print("  quit    - Exit")
        print("=" * 40)

        command = input("Enter command: ").strip().lower()

        if command == "clear":
            blender_connection.clear_scene()
            blender_connection.setup_rendering()
            render_floor(blender_connection)
            print("Scene cleared and reinitialized.")

        elif command == "add bvh" or command == "add":
            root.update()
            root.deiconify()
            root.lift()
            root.focus_force()
            file_path = filedialog.askopenfilename(
                parent=root,
                title="Select BVH file",
                filetypes=[("BVH files", "*.bvh"), ("All files", "*.*")],
            )
            root.withdraw()

            if not file_path:
                print("No file selected.")
                continue

            color_input = input("Enter color (r, g, b) [default: 1, 0, 0]: ").strip()
            color = parse_color(color_input)

            frames_input = input("Number of temporal frames [default: 1]: ").strip()
            if not frames_input:
                num_frames = 1
            else:
                try:
                    num_frames = int(frames_input)
                    if num_frames < 1:
                        num_frames = 1
                except ValueError:
                    print("Invalid number. Using default 1.")
                    num_frames = 1

            frames_between_input = input("Frames between delayed frames [default: 30]: ").strip()
            if not frames_between_input:
                frames_between = 30
            else:
                try:
                    frames_between = int(frames_between_input)
                    if frames_between < 1:
                        frames_between = 30
                except ValueError:
                    print("Invalid number. Using default 30.")
                    frames_between = 30

            scale_input = input("Scale [default: 1]: ").strip()
            if not scale_input:
                scale = 1.0
            else:
                try:
                    scale = float(scale_input)
                except ValueError:
                    print("Invalid number. Using default 1.")
                    scale = 1.0

            bvh = BVH()
            bvh.load(file_path)
            bvh.set_scale(scale)
            end_joints = get_end_joints(bvh)

            render_bvh_with_temporal(blender_connection, bvh, color, end_joints, num_frames, frames_between)
            print(f"Added BVH: {os.path.basename(file_path)} with {num_frames} temporal frame(s), {frames_between} frames between, scale {scale}")

        elif command in ["quit", "q", "exit"]:
            break

        else:
            print(f"Unknown command: {command}")

    blender_connection.close()
    root.destroy()
    print("Connection closed.")


def main(bvh_path, blender_connection):
    global color_idx
    bvh = BVH()
    bvh.load(bvh_path)
    end_joints = get_end_joints(bvh)
    blender_connection.render_bvh_from_path(bvh_path, colors[color_idx], end_joints)
    color_idx = (color_idx + 1) % len(colors)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path",
        type=str,
        nargs="?",
        default=None,
        help="Path to the .bvh file or a directory of .bvh files",
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Whether to incrementally add bvh files to the scene based on user input",
    )
    parser.add_argument(
        "--manual",
        action="store_true",
        help="Interactive manual mode for adding BVH files with custom colors and temporal frames",
    )
    args = parser.parse_args()

    if args.manual:
        manual_mode()
    elif args.path is None:
        print("Error: path is required when not using --manual mode")
        parser.print_help()
        exit(1)
    else:
        if " " in args.path:
            print("Path cannot contain spaces")
            exit(1)

        blender_connection = BlenderConnection()
        blender_connection.clear_scene()
        blender_connection.setup_rendering()
        blender_connection.render_checkerboard_floor()

        if os.path.isfile(args.path):
            main(args.path, blender_connection)
        elif os.path.isdir(args.path):
            for bvh_file in os.listdir(args.path):
                if bvh_file.endswith(".bvh"):
                    bvh_path = os.path.join(args.path, bvh_file)
                    main(bvh_path, blender_connection)
                    if args.incremental:
                        input("Press enter to continue")
                        blender_connection.clear_scene()
                        blender_connection.setup_rendering()
                        blender_connection.render_checkerboard_floor()

        blender_connection.close()
