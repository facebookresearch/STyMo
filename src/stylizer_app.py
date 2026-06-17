# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import argparse
import torch
import warnings
import time

import numpy as np
import pymotion.rotations.quat as quat
import ttkbootstrap as ttk
from scipy.signal import savgol_filter

from tkinter import filedialog
from ttkbootstrap.constants import *
from model_temporal import ModelTransformer
from model_static import AveragePoseModel
from model_gating import GatingModel
from motion_loss import ContactLoss
from pymotion.io.bvh import BVH
from pymotion.render.blender import BlenderConnection
from stylizer_inference import predict, optimize
from utils import resample_bvh, merge_bvhs
from utils_training_data import preprocess_source_input_data
from utils_inference import (
    add_scaled_style_rots,
    add_scaled_style_pos,
    extract_Y_k,
    load_average_deltas,
)
from features import (
    set_bvh_from_full_body_target_kinematics,
    get_feet_indices,
)
from body_parts import body_parts_names
from pymotion.ops.skeleton import fk


class MotionStyleApp:
    def __init__(
        self,
        results_path,
        scale,
        merge_output,
        inplace,
        render_side,
        fix_root,
        post_processing,
        render_framerate,
        render_average,
        render_temp,
    ):
        """
        Initializes the application, performing all the one-time, expensive setup tasks.
        """
        print("\033[93m--- Initializing Application ---\033[0m")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\033[92mDevice: {self.device}\033[0m")

        self.prediction_dirty = True
        self.colors = [
            np.array([0.0, 0.4, 0.9]),
            np.array([0.7, 0.0, 0.9]),
            np.array([0.0, 0.0, 0.9]),
            np.array([0.8, 0.8, 0.0]),
            np.array([0.0, 0.0, 0.0]),
            np.array([0.4, 0.0, 0.8]),
            np.array([0.8, 0.0, 0.4]),
            np.array([0.4, 0.8, 0.0]),
            np.array([0.8, 0.4, 0.0]),
        ]
        self.color_idx = 0
        self.body_parts_names = body_parts_names + ["Pelvis"]

        # --- Load Model and Data (One-time setup) ---
        self.results_path = results_path
        model_path = os.path.join(results_path, "model.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")

        saving_dict = torch.load(
            model_path, map_location=self.device, weights_only=False
        )
        model_state_dict = saving_dict["model"]
        avg_pose_model_state_dict = saving_dict["avg_pose_model"]
        delta_indices = saving_dict["delta_indices"]
        self.x_k_standardize = saving_dict["x_k_standardize"]
        self.Y_k_standardize = saving_dict["Y_k_standardize"]
        self.x_p_standardize = saving_dict["x_p_standardize"]
        self.target_frame_time = saving_dict["target_frame_time"]
        n_in_features = saving_dict["n_in_features"]
        self.n_out_features = saving_dict["n_out_features"]
        n_in_features_avg_pose = saving_dict["n_in_features_avg_pose"]
        n_out_features_avg_pose = saving_dict["n_out_features_avg_pose"]
        n_in_features_gating = saving_dict.get("n_in_features_gating", None)
        gating_model_state_dict = saving_dict.get("gating_model", None)
        gating_model_always_one = saving_dict.get("gating_model_always_one", False)
        self.body_parts_indices = saving_dict["body_parts_indices"]
        self.x_p = saving_dict["x_p"]
        self.feet_velocity_contact_th = saving_dict["feet_velocity_contact_th"]
        self.scale = scale
        self.merge_output = merge_output
        self.inplace = inplace
        self.render_side = render_side
        self.fix_root = fix_root
        self.post_processing = post_processing
        self.window_size = len(delta_indices)
        self.render_framerate = render_framerate
        self.render_average = render_average
        self.render_temp = render_temp

        # --- Load Average Poses ---
        (
            self.average_delta_rotations,
            self.average_delta_pelvis_positions,
            self.target_avg_bvhs_list,
        ) = load_average_deltas(results_path)
        target_avg_local_rotations = []
        target_avg_local_positions = []
        for tgt_avg_bvh in self.target_avg_bvhs_list:
            tgt_avg_rotations, tgt_avg_positions, _, _, _, _ = tgt_avg_bvh.get_data()
            target_avg_local_rotations.append(tgt_avg_rotations[0])
            target_avg_local_positions.append(tgt_avg_positions[0])
        self.target_avg_local_rotations = np.stack(target_avg_local_rotations, axis=0)
        self.target_avg_local_positions = np.stack(target_avg_local_positions, axis=0)

        # --- Create Model ---
        self.model = ModelTransformer(
            input_dim=n_in_features,
            output_dim=self.n_out_features,
            window_size=self.window_size,
        ).to(self.device)
        self.model.load_state_dict(model_state_dict)
        self.model.eval()

        self.avg_model = AveragePoseModel(
            input_dim=n_in_features_avg_pose,
            output_dim=n_out_features_avg_pose,
        ).to(self.device)
        self.avg_model.load_state_dict(avg_pose_model_state_dict)
        self.avg_model.eval()

        # Load gating model (always create one for compatibility)
        if n_in_features_gating is not None:
            self.gating_model = GatingModel(
                input_dim=n_in_features_gating,
                always_one=gating_model_always_one,
            ).to(self.device)
            # Only load state dict if model has learnable parameters (not always_one mode)
            if gating_model_state_dict is not None and not gating_model_always_one:
                self.gating_model.load_state_dict(gating_model_state_dict)
            self.gating_model.eval()
            if gating_model_always_one:
                print(f"\033[93mGating model loaded (always_one=True, no gating effect)\033[0m")
            else:
                print(f"\033[92mGating model loaded\033[0m")
        else:
            # Backward compatibility for models without gating
            self.gating_model = None
            print(f"\033[93mNo gating model found (older model format)\033[0m")

        # --- Load available input BVH files from input folder ---
        self.input_folder = os.path.join(results_path, "input")
        if not os.path.exists(self.input_folder):
            raise FileNotFoundError(f"Input folder not found at {self.input_folder}")

        self.available_bvh_files = [
            f for f in os.listdir(self.input_folder) if f.endswith(".bvh")
        ]
        if len(self.available_bvh_files) == 0:
            raise ValueError(f"No .bvh files found in {self.input_folder}")

        self.available_bvh_files.sort()  # Sort for consistent ordering
        self.current_bvh_file = self.available_bvh_files[0]  # Default to first file

        # Store delta_indices for later use
        self.delta_indices = delta_indices

        # Create app_results directory for saving predictions
        self.app_results_dir = os.path.join(self.results_path, "app_results")
        os.makedirs(self.app_results_dir, exist_ok=True)

        # Load BVH file(s)
        if self.merge_output:
            # Load all BVH files when merge_output is enabled
            print(
                f"Loading all {len(self.available_bvh_files)} BVH files for merged processing..."
            )
            self.load_all_bvh_files()
        else:
            # Load only the initial BVH file
            self.load_bvh_file(self.current_bvh_file)

        print("\033[92m--- Initialization Complete ---\033[0m")

        # --- Setup Blender Connection & UI ---
        self.blender_conn = BlenderConnection()
        self.create_ui()

    def load_bvh_file(self, bvh_filename):
        """
        Loads a specific BVH file from the input folder and preprocesses it.
        """
        print(f"\033[93mLoading BVH file: {bvh_filename}\033[0m")
        bvh_path = os.path.join(self.input_folder, bvh_filename)

        if not bvh_path.endswith(".bvh"):
            raise ValueError(f"Invalid file: {bvh_filename}. Must be a .bvh file.")

        # Load the BVH file
        source_bvh = BVH()
        source_bvh.load(bvh_path)
        source_bvh.set_scale(self.scale)
        source_bvh = resample_bvh(source_bvh, self.target_frame_time)

        # Store in list for compatibility with existing code
        self.source_bvhs = [source_bvh]

        # Extract end joints
        end_joints_idx = list(map(int, source_bvh.data["end_sites_parents"])) + [0]
        self.end_joints = list(
            map(
                str,
                [source_bvh.data["names"][idx] for idx in end_joints_idx],
            )
        )

        # Preprocess kinematics
        input_sources, _ = preprocess_source_input_data(
            self.source_bvhs,
            self.x_k_standardize,
            self.x_p_standardize,
            self.Y_k_standardize,
            self.delta_indices,
            self.feet_velocity_contact_th,
            None,
            self.device,
            None,
        )

        self.input_sources = input_sources

        # Mark prediction as dirty so it will be recomputed
        self.prediction_dirty = True
        print(f"\033[92mLoaded BVH file successfully\033[0m")

    def load_all_bvh_files(self):
        """
        Loads all BVH files from the input folder and preprocesses them.
        Used when merge_output is enabled.
        """
        print(f"\033[93mLoading all BVH files...\033[0m")
        self.source_bvhs = []

        for bvh_filename in self.available_bvh_files:
            bvh_path = os.path.join(self.input_folder, bvh_filename)

            # Load the BVH file
            source_bvh = BVH()
            source_bvh.load(bvh_path)
            source_bvh.set_scale(self.scale)
            source_bvh = resample_bvh(source_bvh, self.target_frame_time)

            self.source_bvhs.append(source_bvh)
            print(f"  \033[97m- {bvh_filename}\033[0m")

        # Extract end joints from the first file (should be same for all)
        first_bvh = self.source_bvhs[0]
        end_joints_idx = list(map(int, first_bvh.data["end_sites_parents"])) + [0]
        self.end_joints = list(
            map(
                str,
                [first_bvh.data["names"][idx] for idx in end_joints_idx],
            )
        )

        # Preprocess kinematics for all files
        input_sources, _ = preprocess_source_input_data(
            self.source_bvhs,
            self.x_k_standardize,
            self.x_p_standardize,
            self.Y_k_standardize,
            self.delta_indices,
            self.feet_velocity_contact_th,
            None,
            self.device,
            None,
        )

        self.input_sources = input_sources

        # Mark prediction as dirty so it will be recomputed
        self.prediction_dirty = True
        print(f"\033[92mLoaded {len(self.source_bvhs)} BVH files successfully\033[0m")

    def on_bvh_selection_changed(self, event=None):
        """
        Called when a new BVH file is selected from the dropdown.
        """
        selected_file = self.bvh_selector.get()
        if selected_file != self.current_bvh_file:
            self.current_bvh_file = selected_file
            self.load_bvh_file(selected_file)
            self.render_scene(clear_scene=True)

    def on_postprocessing_changed(self, event=None):
        """
        Called when postprocessing quality is changed.
        """
        pass

    def on_contact_strength_changed(self, event=None):
        """
        Called when contact strength is changed.
        """
        pass

    def on_smoothing_changed(self, event=None):
        """
        Called when smoothing is changed.
        """
        pass

    def get_contact_epochs(self):
        """
        Maps the postprocessing selection to the number of contact epochs.
        """
        mapping = {"None": 0, "Fast": 100, "Normal": 500, "High": 2000}
        return mapping.get(self.postprocessing_selector.get(), 100)

    def get_lambda_contacts(self):
        """
        Maps the contact strength selection to lambda_contacts value.
        """
        mapping = {"Normal": 100.0, "Strong": 1000.0, "Very Strong": 10000.0}
        return mapping.get(self.contact_strength_selector.get(), 100.0)

    def get_lambda_smooth(self):
        """
        Maps the smoothing selection to lambda_smooth value.
        """
        mapping = {"Low": 0.001, "Normal": 0.01, "High": 0.1}
        return mapping.get(self.smoothing_selector.get(), 0.01)

    def create_ui(self):
        """
        Creates the dynamic UI with simple and advanced modes.
        """
        self.root = ttk.Window(themename="darkly")
        self.root.title("Style Controller")

        main_frame = ttk.Frame(self.root, padding=20)
        main_frame.pack(fill=BOTH, expand=YES)

        selector_frame = ttk.Frame(main_frame)
        selector_frame.pack(fill=X, expand=YES, pady=(0, 15), anchor=N)
        selector_frame.configure(relief="solid", borderwidth=1, padding=10)

        # --- BVH FILE SELECTOR (only shown when not merging) ---
        if not self.merge_output:
            ttk.Label(
                selector_frame, text="Input BVH File", font=("-weight bold", 14)
            ).pack(fill=X, anchor=W, pady=(0, 5))

            ttk.Separator(selector_frame, style="secondary").pack(fill=X, pady=5)

            selector_row = ttk.Frame(selector_frame)
            selector_row.pack(fill=X, pady=5)

            ttk.Label(selector_row, text="Select File:").pack(side=LEFT, padx=(0, 10))

            self.bvh_selector = ttk.Combobox(
                selector_row,
                values=self.available_bvh_files,
                state="readonly",
                width=40,
            )
            self.bvh_selector.set(self.current_bvh_file)
            self.bvh_selector.bind(
                "<<ComboboxSelected>>", self.on_bvh_selection_changed
            )
            self.bvh_selector.pack(side=LEFT, fill=X, expand=YES)
        else:
            # Show info about merged processing
            info_frame = ttk.Frame(main_frame)
            info_frame.pack(fill=X, expand=YES, pady=(0, 15), anchor=N)
            info_frame.configure(relief="solid", borderwidth=1, padding=10)

            ttk.Label(info_frame, text="Merge Mode", font=("-weight bold", 14)).pack(
                fill=X, anchor=W, pady=(0, 5)
            )

            ttk.Separator(info_frame, style="secondary").pack(fill=X, pady=5)

            ttk.Label(
                info_frame,
                text=f"Processing {len(self.available_bvh_files)} BVH file(s) in merged mode",
                wraplength=400,
            ).pack(fill=X, pady=5)

        # Show Foot Contacts toggle
        self.show_foot_contacts_var = ttk.BooleanVar(value=True)
        foot_contacts_check = ttk.Checkbutton(
            selector_frame,
            text="Show Foot Contacts",
            variable=self.show_foot_contacts_var,
            bootstyle="primary-round-toggle",
        )
        foot_contacts_check.pack(fill=X, pady=(10, 5))

        # --- PHYSICS LOSS SECTION ---
        physics_frame = ttk.Frame(main_frame)
        physics_frame.pack(fill=X, expand=YES, pady=(0, 10), anchor=N)
        physics_frame.configure(relief="solid", borderwidth=1, padding=10)

        ttk.Label(physics_frame, text="Physics Loss", font=("-weight bold", 14)).pack(
            fill=X, anchor=W, pady=(0, 5)
        )

        ttk.Separator(physics_frame, style="secondary").pack(fill=X, pady=5)

        # Postprocessing quality selector
        postprocessing_row = ttk.Frame(physics_frame)
        postprocessing_row.pack(fill=X, pady=(5, 5))

        ttk.Label(postprocessing_row, text="Postprocessing:").pack(
            side=LEFT, padx=(0, 10)
        )

        self.postprocessing_options = ["None", "Fast", "Normal", "High"]
        self.postprocessing_selector = ttk.Combobox(
            postprocessing_row,
            values=self.postprocessing_options,
            state="readonly",
            width=15,
        )
        self.postprocessing_selector.set(self.post_processing.capitalize())
        self.postprocessing_selector.bind(
            "<<ComboboxSelected>>", self.on_postprocessing_changed
        )
        self.postprocessing_selector.pack(side=LEFT)

        # Foot Contact Strength selector
        contact_strength_row = ttk.Frame(physics_frame)
        contact_strength_row.pack(fill=X, pady=(5, 5))

        ttk.Label(contact_strength_row, text="Foot Contact Strength:").pack(
            side=LEFT, padx=(0, 10)
        )

        self.contact_strength_options = ["Normal", "Strong", "Very Strong"]
        self.contact_strength_selector = ttk.Combobox(
            contact_strength_row,
            values=self.contact_strength_options,
            state="readonly",
            width=15,
        )
        self.contact_strength_selector.set("Normal")
        self.contact_strength_selector.bind(
            "<<ComboboxSelected>>", self.on_contact_strength_changed
        )
        self.contact_strength_selector.pack(side=LEFT)

        # Smoothing selector
        smoothing_row = ttk.Frame(physics_frame)
        smoothing_row.pack(fill=X, pady=(5, 5))

        ttk.Label(smoothing_row, text="Smoothing:").pack(side=LEFT, padx=(0, 10))

        self.smoothing_options = ["Low", "Normal", "High"]
        self.smoothing_selector = ttk.Combobox(
            smoothing_row,
            values=self.smoothing_options,
            state="readonly",
            width=15,
        )
        self.smoothing_selector.set("Normal")
        self.smoothing_selector.bind("<<ComboboxSelected>>", self.on_smoothing_changed)
        self.smoothing_selector.pack(side=LEFT)

        ttk.Separator(main_frame, style="primary").pack(fill=X, pady=(0, 15))

        # --- DYNAMIC STYLE SECTION ---
        dynamic_frame = ttk.Frame(main_frame)
        dynamic_frame.pack(fill=X, expand=YES, pady=(0, 10), anchor=N)
        dynamic_frame.configure(relief="solid", borderwidth=1, padding=10)

        ttk.Label(dynamic_frame, text="Dynamic Style", font=("-weight bold", 14)).pack(
            fill=X, anchor=W, pady=(0, 5)
        )

        ttk.Separator(dynamic_frame, style="secondary").pack(fill=X, pady=5)

        # Block Encoder
        block_header = ttk.Frame(dynamic_frame)
        block_header.pack(fill=X)

        ttk.Label(block_header, text="Encoder", font="-weight bold").pack(
            side=LEFT, fill=X, anchor=W, pady=5
        )

        # Encoder Scale
        encoder_scale_frame = ttk.Frame(dynamic_frame)
        encoder_scale_frame.pack(fill=X, pady=5)
        ttk.Label(encoder_scale_frame, text="Output").pack(side=LEFT, padx=(0, 10))
        self.encoder_scale_slider = ttk.Scale(
            encoder_scale_frame,
            from_=0.0,
            to=3.0,
            value=1.0,
            command=self.on_encoder_scale_change,
        )
        self.encoder_scale_slider.pack(side=LEFT, fill=X, expand=YES, padx=(0, 10))
        self.encoder_scale_value_var = ttk.StringVar(value="1.00")
        ttk.Label(encoder_scale_frame, textvariable=self.encoder_scale_value_var).pack(
            side=RIGHT
        )

        ttk.Separator(dynamic_frame, style="secondary").pack(fill=X, pady=5)

        # Dynamic Output
        output_row = ttk.Frame(dynamic_frame)
        output_row.pack(fill=X, pady=5)
        ttk.Label(output_row, text="Decoder", font="-weight bold").pack(side=LEFT)

        # Advanced Mode Checkbox
        self.advanced_mode_var_dyn = ttk.BooleanVar(value=False)
        advanced_check = ttk.Checkbutton(
            output_row,
            text="Body Parts",
            variable=self.advanced_mode_var_dyn,
            command=lambda: self.toggle_advanced_mode(
                self.advanced_mode_var_dyn,
                self.advanced_frame_dyn,
                self.simple_frame_dyn,
            ),
            bootstyle="primary-round-toggle",
        )
        advanced_check.pack(side=RIGHT, anchor=E, pady=0, padx=5)

        # Smoothing checkbox
        # self.smoothing_var = ttk.BooleanVar(value=True)
        # smoothing_check = ttk.Checkbutton(
        #     output_row,
        #     text="Smoothing",
        #     variable=self.smoothing_var,
        #     bootstyle="primary-round-toggle",
        # )
        # smoothing_check.pack(side=RIGHT, anchor=E, pady=0, padx=5)

        # Simple Mode Frame
        self.simple_frame_dyn = ttk.Frame(dynamic_frame)
        self.simple_frame_dyn.pack(fill=X, expand=YES)

        header_frame = ttk.Frame(self.simple_frame_dyn)
        header_frame.pack(fill=X)
        ttk.Label(header_frame, text="Output").pack(side=LEFT, padx=(0, 10))
        self.style_slider_dyn = ttk.Scale(
            header_frame,
            from_=0.0,
            to=3.0,
            value=1.0,
            command=lambda v: self.slider_value_var_dyn.set(f"{float(v):.2f}"),
        )
        self.style_slider_dyn.pack(side=LEFT, fill=X, expand=YES, padx=(0, 10))
        self.slider_value_var_dyn = ttk.StringVar(value="1.00")
        ttk.Label(header_frame, textvariable=self.slider_value_var_dyn).pack(side=RIGHT)

        # Advanced Mode Frame (initially hidden)
        self.advanced_frame_dyn = ttk.Frame(dynamic_frame)
        self.advanced_sliders_dyn = {}
        self.advanced_sliders_value_Var_dyn = {}
        ttk.Label(self.advanced_frame_dyn, text="Output").pack(fill=X)
        for part_name in self.body_parts_names:
            part_frame = ttk.Frame(self.advanced_frame_dyn)
            part_frame.pack(fill=X, expand=YES, pady=2)
            label = ttk.Label(part_frame, text=f"{part_name}:", width=12, anchor=W)
            label.pack(side=LEFT)
            slider = ttk.Scale(
                part_frame,
                from_=0.0,
                to=3.0,
                value=1.0,
                command=lambda v, part=part_name: self.advanced_sliders_value_Var_dyn[
                    part
                ].set(f"{float(v):.2f}"),
            )
            slider.pack(side=LEFT, fill=X, expand=YES, padx=10)
            self.advanced_sliders_dyn[part_name] = slider
            self.advanced_sliders_value_Var_dyn[part_name] = ttk.StringVar(value="1.00")
            value_label = ttk.Label(
                part_frame,
                textvariable=self.advanced_sliders_value_Var_dyn[part_name],
                width=5,
                anchor=E,
            )
            value_label.pack(side=RIGHT)

        ttk.Separator(dynamic_frame, style="secondary").pack(fill=X, pady=5)

        # Induce Action Header with Toggle
        induce_action_header = ttk.Frame(dynamic_frame)
        induce_action_header.pack(fill=X, pady=5)
        ttk.Label(
            induce_action_header,
            text="Induce Action",
            font="-weight bold",
        ).pack(side=LEFT)

        # Toggle for Induce Action section
        self.induce_action_expanded_var = ttk.BooleanVar(value=False)
        induce_action_toggle = ttk.Checkbutton(
            induce_action_header,
            text="Expand",
            variable=self.induce_action_expanded_var,
            command=self.toggle_induce_action,
            bootstyle="primary-round-toggle",
        )
        induce_action_toggle.pack(side=RIGHT, anchor=E, pady=0, padx=5)

        # Collapsible frame for Induce Action controls
        self.induce_action_frame = ttk.Frame(dynamic_frame)

        # Induce Action Index Start Slider (with label and value)
        induce_action_start_frame = ttk.Frame(self.induce_action_frame)
        induce_action_start_frame.pack(fill=X, pady=5)
        ttk.Label(induce_action_start_frame, text="Index Start").pack(side=LEFT)
        self.induce_action_start_slider = ttk.Scale(
            induce_action_start_frame,
            from_=0,
            to=self.x_p.shape[0] - 1,
            value=0,
            command=self.on_induce_action_start_change,
        )
        self.induce_action_start_slider.pack(side=LEFT, fill=X, expand=YES, padx=10)
        self.induce_action_start_value_var = ttk.StringVar(value="0")
        ttk.Label(
            induce_action_start_frame, textvariable=self.induce_action_start_value_var
        ).pack(side=RIGHT)

        # Induce Action Index End Slider (with label and value)
        induce_action_end_frame = ttk.Frame(self.induce_action_frame)
        induce_action_end_frame.pack(fill=X, pady=5)
        ttk.Label(induce_action_end_frame, text="Index End").pack(side=LEFT)
        self.induce_action_end_slider = ttk.Scale(
            induce_action_end_frame,
            from_=0,
            to=self.x_p.shape[0] - 1,
            value=0,
            command=self.on_induce_action_end_change,
        )
        self.induce_action_end_slider.pack(side=LEFT, fill=X, expand=YES, padx=10)
        self.induce_action_end_value_var = ttk.StringVar(value="0")
        ttk.Label(
            induce_action_end_frame, textvariable=self.induce_action_end_value_var
        ).pack(side=RIGHT)

        # Induce Action Frequency Slider (with label and value)
        induce_action_frequency_frame = ttk.Frame(self.induce_action_frame)
        induce_action_frequency_frame.pack(fill=X, pady=5)
        ttk.Label(induce_action_frequency_frame, text="Frequency").pack(side=LEFT)
        self.induce_action_frequency_slider = ttk.Scale(
            induce_action_frequency_frame,
            from_=0.0,
            to=1.0,
            value=0.5,
            command=self.on_induce_action_frequency_change,
        )
        self.induce_action_frequency_slider.pack(side=LEFT, fill=X, expand=YES, padx=10)
        self.induce_action_frequency_value_var = ttk.StringVar(value="0.50")
        ttk.Label(
            induce_action_frequency_frame,
            textvariable=self.induce_action_frequency_value_var,
        ).pack(side=RIGHT)

        # Induce Action Speed Slider (with label and value)
        induce_action_speed_frame = ttk.Frame(self.induce_action_frame)
        induce_action_speed_frame.pack(fill=X, pady=5)
        ttk.Label(induce_action_speed_frame, text="Speed").pack(side=LEFT)
        self.induce_action_speed_slider = ttk.Scale(
            induce_action_speed_frame,
            from_=0.1,
            to=4.0,
            value=1.0,
            command=self.on_induce_action_speed_change,
        )
        self.induce_action_speed_slider.pack(side=LEFT, fill=X, expand=YES, padx=10)
        self.induce_action_speed_value_var = ttk.StringVar(value="1.00")
        ttk.Label(
            induce_action_speed_frame,
            textvariable=self.induce_action_speed_value_var,
        ).pack(side=RIGHT)

        # Induce Action Strength Slider (with label and value)
        induce_action_strength_frame = ttk.Frame(self.induce_action_frame)
        induce_action_strength_frame.pack(fill=X, pady=5)
        ttk.Label(induce_action_strength_frame, text="Strength").pack(side=LEFT)
        self.induce_action_strength_slider = ttk.Scale(
            induce_action_strength_frame,
            from_=0.0,
            to=1.0,
            value=0.0,
            command=self.on_induce_action_strength_change,
        )
        self.induce_action_strength_slider.pack(side=LEFT, fill=X, expand=YES, padx=10)
        self.induce_action_strength_value_var = ttk.StringVar(value="0.00")
        ttk.Label(
            induce_action_strength_frame,
            textvariable=self.induce_action_strength_value_var,
        ).pack(side=RIGHT)

        ttk.Separator(main_frame, style="primary").pack(fill=X, pady=(15, 20))

        # --- STATIC STYLE SECTION ---
        static_frame = ttk.Frame(main_frame)
        static_frame.pack(fill=X, expand=YES, pady=(0, 10), anchor=N)
        static_frame.configure(relief="solid", borderwidth=1, padding=10)

        ttk.Label(static_frame, text="Static Style", font=("-weight bold", 14)).pack(
            fill=X, anchor=W, pady=(0, 5)
        )

        ttk.Separator(static_frame, style="secondary").pack(fill=X, pady=5)

        # Static Output
        output_row = ttk.Frame(static_frame)
        output_row.pack(fill=X, pady=5)
        ttk.Label(output_row, text="Average", font="-weight bold").pack(side=LEFT)

        # Advanced Mode Checkbox
        self.advanced_mode_var = ttk.BooleanVar(value=False)
        advanced_check = ttk.Checkbutton(
            output_row,
            text="Body Parts",
            variable=self.advanced_mode_var,
            command=lambda: self.toggle_advanced_mode(
                self.advanced_mode_var, self.advanced_frame, self.simple_frame
            ),
            bootstyle="primary-round-toggle",
        )
        advanced_check.pack(side=RIGHT, anchor=E, pady=0)

        # Simple Mode Frame
        self.simple_frame = ttk.Frame(static_frame)
        self.simple_frame.pack(fill=X, expand=YES)

        header_frame = ttk.Frame(self.simple_frame)
        header_frame.pack(fill=X)
        ttk.Label(header_frame, text="Output").pack(side=LEFT, padx=(0, 10))
        self.style_slider = ttk.Scale(
            header_frame,
            from_=0.0,
            to=3.0,
            value=1.0,
            command=lambda v: self.slider_value_var.set(f"{float(v):.2f}"),
        )
        self.style_slider.pack(side=LEFT, fill=X, expand=YES, padx=(0, 10))
        self.slider_value_var = ttk.StringVar(value="1.00")
        ttk.Label(header_frame, textvariable=self.slider_value_var).pack(side=RIGHT)

        # Advanced Mode Frame (initially hidden)
        self.advanced_frame = ttk.Frame(static_frame)
        self.advanced_sliders = {}
        self.advanced_sliders_value_Var = {}
        ttk.Label(self.advanced_frame, text="Output").pack(fill=X)
        for part_name in self.body_parts_names:
            part_frame = ttk.Frame(self.advanced_frame)
            part_frame.pack(fill=X, expand=YES, pady=2)
            # Arrange: text left, slider center, number right, all in one line
            label = ttk.Label(part_frame, text=f"{part_name}:", width=12, anchor=W)
            label.pack(side=LEFT, padx=(0, 5))
            slider = ttk.Scale(
                part_frame,
                from_=0.0,
                to=3.0,
                value=1.0,
                command=lambda v, part=part_name: self.advanced_sliders_value_Var[
                    part
                ].set(f"{float(v):.2f}"),
            )
            slider.pack(side=LEFT, fill=X, expand=YES, padx=(0, 5))
            self.advanced_sliders[part_name] = slider
            self.advanced_sliders_value_Var[part_name] = ttk.StringVar(value="1.00")
            value_label = ttk.Label(
                part_frame,
                textvariable=self.advanced_sliders_value_Var[part_name],
                width=5,
                anchor=E,
            )
            value_label.pack(side=RIGHT)

        # --- GATING MODEL SECTION ---
        if self.gating_model is not None:
            gating_frame = ttk.Frame(main_frame)
            gating_frame.pack(fill=X, expand=YES, pady=(0, 10), anchor=N)
            gating_frame.configure(relief="solid", borderwidth=1, padding=10)

            ttk.Label(gating_frame, text="Stylizability Gate", font=("-weight bold", 14)).pack(
                fill=X, anchor=W, pady=(0, 5)
            )

            ttk.Separator(gating_frame, style="secondary").pack(fill=X, pady=5)

            # Description
            ttk.Label(
                gating_frame,
                text="0.0 = ignore gating, 1.0 = full gating effect",
                wraplength=400,
            ).pack(fill=X, pady=5)

            # Gating strength for dynamic style
            gating_dyn_frame = ttk.Frame(gating_frame)
            gating_dyn_frame.pack(fill=X, pady=5)
            ttk.Label(gating_dyn_frame, text="Dynamic Gate").pack(side=LEFT, padx=(0, 10))
            self.gating_dynamic_slider = ttk.Scale(
                gating_dyn_frame,
                from_=0.0,
                to=1.0,
                value=0.0,
                command=lambda v: self.gating_dynamic_value_var.set(f"{float(v):.2f}"),
            )
            self.gating_dynamic_slider.pack(side=LEFT, fill=X, expand=YES, padx=(0, 10))
            self.gating_dynamic_value_var = ttk.StringVar(value="0.00")
            ttk.Label(gating_dyn_frame, textvariable=self.gating_dynamic_value_var).pack(side=RIGHT)

            # Gating strength for static style
            gating_static_frame = ttk.Frame(gating_frame)
            gating_static_frame.pack(fill=X, pady=5)
            ttk.Label(gating_static_frame, text="Static Gate").pack(side=LEFT, padx=(0, 10))
            self.gating_static_slider = ttk.Scale(
                gating_static_frame,
                from_=0.0,
                to=1.0,
                value=0.0,
                command=lambda v: self.gating_static_value_var.set(f"{float(v):.2f}"),
            )
            self.gating_static_slider.pack(side=LEFT, fill=X, expand=YES, padx=(0, 10))
            self.gating_static_value_var = ttk.StringVar(value="0.00")
            ttk.Label(gating_static_frame, textvariable=self.gating_static_value_var).pack(side=RIGHT)
        else:
            # No gating model - set default values
            self.gating_dynamic_slider = None
            self.gating_static_slider = None

        # --- SHARED CONTROLS ---

        # Render Button
        ttk.Separator(main_frame, style="primary").pack(fill=X, pady=15)
        render_button = ttk.Button(
            main_frame,
            text="Render Scene",
            command=lambda: self.render_scene(clear_scene=True),
            bootstyle="primary",
        )
        render_button.pack(fill=X)

        # Add to Scene Button
        add_button = ttk.Button(
            main_frame,
            text="Add to Scene",
            command=lambda: self.render_scene(clear_scene=False),
            bootstyle="secondary",
        )
        add_button.pack(fill=X, pady=(10, 0))

        # Add BVH File Button
        add_bvh_button = ttk.Button(
            main_frame,
            text="Add BVH File",
            command=self.add_bvh_file_to_scene,
            bootstyle="secondary",
        )
        add_bvh_button.pack(fill=X, pady=(10, 0))

        # Add Training Data Button
        add_training_data_button = ttk.Button(
            main_frame,
            text="Add Training Data",
            command=self.add_training_data_to_scene,
            bootstyle="secondary",
        )
        add_training_data_button.pack(fill=X, pady=(10, 0))

        # --- EXPERIMENTS ---
        ttk.Separator(main_frame, style="primary").pack(fill=X, pady=15)
        ttk.Label(main_frame, text="Experiments", font="-weight bold").pack(fill=X)

        ttk.Separator(main_frame, style="secondary").pack(fill=X, pady=5)

        self.experiments_frame = ttk.Frame(main_frame)
        self.experiments_frame.pack(fill=X, expand=YES, pady=(0, 10), anchor=N)

        # --- INIT ---
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.toggle_advanced_mode(
            self.advanced_mode_var, self.advanced_frame, self.simple_frame, first=True
        )  # Set initial UI state
        self.toggle_advanced_mode(
            self.advanced_mode_var_dyn,
            self.advanced_frame_dyn,
            self.simple_frame_dyn,
            first=True,
        )  # Set initial UI state

    def on_encoder_scale_change(self, value):
        """Called when the encoder scale slider is moved."""
        self.encoder_scale_value_var.set(f"{float(value):.2f}")
        self.prediction_dirty = True

    def on_induce_action_start_change(self, value):
        """Called when the induce action start slider is moved."""
        value = int(round(float(value)))
        if value != self.induce_action_start_slider.get():
            self.induce_action_start_slider.set(value)
        self.induce_action_start_value_var.set(f"{value}")
        # Ensure start doesn't exceed end
        if value > self.induce_action_end_slider.get():
            self.induce_action_end_slider.set(value)
            self.induce_action_end_value_var.set(f"{value}")
        self.prediction_dirty = True

    def on_induce_action_end_change(self, value):
        """Called when the induce action end slider is moved."""
        value = int(round(float(value)))
        if value != self.induce_action_end_slider.get():
            self.induce_action_end_slider.set(value)
        self.induce_action_end_value_var.set(f"{value}")
        # Ensure end doesn't go below start
        if value < self.induce_action_start_slider.get():
            self.induce_action_start_slider.set(value)
            self.induce_action_start_value_var.set(f"{value}")
        self.prediction_dirty = True

    def on_induce_action_frequency_change(self, value):
        """Called when the induce action frequency slider is moved."""
        self.induce_action_frequency_value_var.set(f"{float(value):.2f}")
        self.prediction_dirty = True

    def on_induce_action_speed_change(self, value):
        """Called when the induce action speed slider is moved."""
        self.induce_action_speed_value_var.set(f"{float(value):.2f}")
        self.prediction_dirty = True

    def on_induce_action_strength_change(self, value):
        """Called when the induce action strength slider is moved."""
        self.induce_action_strength_value_var.set(f"{float(value):.2f}")
        self.prediction_dirty = True

    def toggle_advanced_mode(
        self, advanced_mode_var, advanced_frame, simple_frame, first=False
    ):
        """Shows or hides the advanced control frame."""
        if advanced_mode_var.get():
            if first:
                advanced_frame.pack(fill=X, expand=YES)
            else:
                advanced_frame.pack(fill=X, expand=YES, before=simple_frame)
                simple_frame.pack_forget()
        else:
            if first:
                simple_frame.pack(fill=X, expand=YES)
            else:
                simple_frame.pack(fill=X, expand=YES, before=advanced_frame)
                advanced_frame.pack_forget()

    def toggle_induce_action(self):
        """Shows or hides the induce action controls frame."""
        if self.induce_action_expanded_var.get():
            self.induce_action_frame.pack(fill=X, expand=YES, pady=(5, 0))
        else:
            self.induce_action_frame.pack_forget()

    def add_experiment(self, experiment):
        """
        Adds an experiment to the UI.
        """
        experiment_frame = ttk.Frame(self.experiments_frame)
        experiment_frame.pack(fill=X, expand=YES, pady=2)

        # Color square only (no label)
        color_rgb = experiment["color"]
        hex_color = f"#{int(color_rgb[0]*255):02x}{int(color_rgb[1]*255):02x}{int(color_rgb[2]*255):02x}"
        color_square = ttk.Label(experiment_frame, width=2)
        color_square.pack(side=LEFT, anchor=W, padx=5, pady=2)
        color_square.config(background=hex_color)

        # Add labels for other properties
        for name, value in experiment.items():
            if name in ["color"]:
                continue

            if isinstance(value, dict):
                value_str = (
                    "{" + ", ".join(f"{k}: {v:.1f}" for k, v in value.items()) + "}"
                )
            elif isinstance(value, np.ndarray):
                value_str = f"[{', '.join(f'{v:.1f}' for v in value.flatten())}]"
            elif isinstance(value, (int, float)):
                value_str = f"{value:.1f}"
            else:
                value_str = str(value)

            label = ttk.Label(experiment_frame, text=f"{name}: {value_str}")
            label.configure(font=("TkDefaultFont", 8))
            label.pack(side=LEFT, padx=5)

    def clear_experiments(self):
        """
        Clears all experiments from the UI.
        """
        for child in self.experiments_frame.winfo_children():
            child.destroy()

    def predict_data(self):
        """
        Run neural network prediction and cache raw results (torch tensors).
        Does NOT apply contact optimization - that happens in optimize_data().
        """
        print(f"\033[93mRunning Prediction...\033[0m")
        start_time = time.time()

        induce_x_p_idx_start = int(self.induce_action_start_slider.get())
        induce_x_p_idx_end = int(self.induce_action_end_slider.get())
        induce_x_p_frequency = (
            (self.induce_action_frequency_slider.get() * 0.4) ** 2
        ) * 5.0
        induce_x_p_speed = self.induce_action_speed_slider.get()
        induce_x_p_strength = self.induce_action_strength_slider.get() * 0.2

        # Gather all input sources into a list
        self.input_source_list = [
            self.input_sources[source_bvh.data["filename"]]
            for source_bvh in self.source_bvhs
        ]

        # Call predict (returns torch tensors, standardized)
        (
            self.cached_Y_k_pred_list,
            self.cached_avg_delta_rots_list,
            self.cached_avg_delta_pos_list,
            self.cached_chunk_probs_list,
            self.cached_gating_scores_list,
        ) = predict(
            input_sources=self.input_source_list,
            model=self.model,
            avg_model=self.avg_model,
            average_delta_rotations=self.average_delta_rotations,
            average_delta_pelvis_positions=self.average_delta_pelvis_positions,
            Y_k_standardize=self.Y_k_standardize,
            induce_x_p=self.x_p[induce_x_p_idx_start : induce_x_p_idx_end + 1],
            induce_x_p_frequency=induce_x_p_frequency,
            induce_x_p_speed=induce_x_p_speed,
            induce_x_p_strength=induce_x_p_strength,
            encoder_output_scale=self.encoder_scale_slider.get(),
            gating_model=self.gating_model,
        )

        # Apply Savitzky-Golay filter to smooth gating scores
        if self.cached_gating_scores_list is not None:
            self.cached_gating_scores_list = self._smooth_gating_scores(
                self.cached_gating_scores_list
            )

        print(f"\033[38;5;228mPrediction: {time.time() - start_time:.2f}s\033[0m")
        self.prediction_dirty = False

    def _smooth_gating_scores(self, gating_scores_list, window_length=61, polyorder=2):
        """
        Apply Savitzky-Golay filter to smooth gating scores.

        Args:
            gating_scores_list: List of torch tensors with gating scores [frames, 1]
            window_length: Length of the filter window (must be odd). Default 61.
            polyorder: Order of the polynomial used to fit samples. Default 2.

        Returns:
            List of smoothed torch tensors
        """
        import torch

        smoothed_list = []
        for gating_scores in gating_scores_list:
            scores_np = gating_scores.cpu().numpy()  # [frames, 1]
            num_frames = scores_np.shape[0]

            # Adjust window_length if data is too short
            effective_window = min(window_length, num_frames)
            if effective_window % 2 == 0:
                effective_window -= 1  # Must be odd
            effective_window = max(effective_window, polyorder + 2)  # Must be > polyorder

            if num_frames > polyorder + 1:
                # Apply Savitzky-Golay filter
                smoothed_scores = savgol_filter(
                    scores_np[:, 0],
                    window_length=effective_window,
                    polyorder=polyorder,
                    mode='nearest'
                )
                # Clip to [0, 1] range since gating scores should be in this range
                smoothed_scores = np.clip(smoothed_scores, 0.0, 1.0)
                smoothed_scores = smoothed_scores[:, np.newaxis]  # Restore shape [frames, 1]
            else:
                smoothed_scores = scores_np

            # Convert back to torch tensor on the same device
            smoothed_tensor = torch.from_numpy(smoothed_scores).float().to(gating_scores.device)
            smoothed_list.append(smoothed_tensor)

        return smoothed_list

    def optimize_data(
        self,
        gated_scales_per_source=None,
    ):
        """
        Apply contact optimization to cached predictions and convert to numpy.
        Creates ContactLoss with current UI parameters.

        Args:
            gated_scales_per_source: List of dicts with per-source gated scales:
                - 'dynamic': Scale factor for dynamic deltas
                - 'dynamic_pelvis': Scale factor for dynamic pelvis position
                - 'static': Scale factor for static deltas
                - 'static_pelvis': Scale factor for static pelvis position
        """
        contact_epochs = self.get_contact_epochs()

        # Create ContactLoss instance with current UI parameters
        contact_loss_fn = None
        if contact_epochs > 0:
            contact_loss_fn = ContactLoss(
                source_bvh=self.source_bvhs[0],
                Y_k_standardize=self.Y_k_standardize,
                body_parts_indices=self.body_parts_indices,
                target_frame_time=self.target_frame_time,
                lambda_contacts=self.get_lambda_contacts(),
                lambda_smooth=self.get_lambda_smooth(),
            )

        # Call optimize with scaling factors (returns numpy arrays, unstandardized, middle frame extracted)
        (
            Y_k_pred_list,
            average_delta_rotations_pred_list,
            average_delta_pelvis_positions_pred_list,
            chunk_probs_final_list,
        ) = optimize(
            Y_k_pred_list=self.cached_Y_k_pred_list,
            average_delta_rotations_list=self.cached_avg_delta_rots_list,
            average_delta_pelvis_positions_list=self.cached_avg_delta_pos_list,
            chunk_probs_list=self.cached_chunk_probs_list,
            input_sources=self.input_source_list,
            contact_loss_fn=contact_loss_fn,
            contact_epochs=contact_epochs,
            window_size=self.window_size,
            Y_k_standardize=self.Y_k_standardize,
            gated_scales_per_source=gated_scales_per_source,
        )

        # Store results for rendering
        self.Y_k_preds = Y_k_pred_list
        self.average_delta_rotations_preds = average_delta_rotations_pred_list
        self.average_delta_pelvis_positions_preds = (
            average_delta_pelvis_positions_pred_list
        )
        self.chunk_probs_preds = chunk_probs_final_list

    def render_scene(self, clear_scene):
        """
        Gathers style factors from the active UI mode and renders the scene.
        """
        # Clear scene --------------------------------
        if clear_scene:
            self.blender_conn.clear_scene()
            self.blender_conn.setup_rendering()
            self.blender_conn.render_checkerboard_floor()
            merged_source_bvh = None
            viz_contacts = []
            for i, source_bvh_i in enumerate(self.source_bvhs):
                feet_indices = get_feet_indices(source_bvh_i)
                input_source = self.input_sources[source_bvh_i.data["filename"]]
                local_rotations_i, local_positions_i, parents_i, offsets_i, _, _ = (
                    source_bvh_i.get_data()
                )
                offsets_i = np.tile(offsets_i, (local_positions_i.shape[0], 1, 1))
                offsets_i[:, 1, :] = local_positions_i[:, 1, :]
                pos_i, _ = fk(
                    local_rotations_i, local_positions_i[:, 0, :], offsets_i, parents_i
                )
                pos_i = pos_i[..., [0, 2, 1]]
                pos_i[..., 1] = -pos_i[..., 1]
                viz_contacts_i = np.where(
                    input_source["x_k_contacts"]
                    .cpu()
                    .numpy()[:, self.window_size // 2, :, None],
                    pos_i[:, feet_indices],
                    np.array([[[1000.0, 1000.0, 1000.0]]]),
                )
                if self.merge_output:
                    source_bvh_i_i = self.detect_loop_and_trim(source_bvh_i)
                    if merged_source_bvh is None:
                        merged_source_bvh = BVH()
                        merged_source_bvh.copy(source_bvh_i_i)
                    else:
                        merge_bvhs(merged_source_bvh, source_bvh_i_i)
                    viz_contacts.append(viz_contacts_i)
                else:
                    self.render_bvh(source_bvh_i, np.array([0.0, 0.8, 0.0]))
                    if self.show_foot_contacts_var.get():
                        self.render_points(
                            viz_contacts_i, np.array([0.4, 0.8, 0.0]), np.array([0.045])
                        )
            if self.merge_output:
                self.render_bvh(merged_source_bvh, np.array([0.0, 0.8, 0.0]))
                if self.show_foot_contacts_var.get():
                    viz_contacts = np.concatenate(viz_contacts, axis=0)
                    self.render_points(
                        viz_contacts, np.array([0.4, 0.8, 0.0]), np.array([0.045])
                    )
            self.color_idx = 0
            self.clear_experiments()

        # Prediction -----------------------
        if self.prediction_dirty:
            self.predict_data()

        # Style factors from UI (BEFORE optimization) -------------
        (
            dynamic_style_factor,
            dynamic_pelvis_style_factor,
            constant_style_factor,
            constant_pelvis_style_factor,
            experiment_style_factor_dyn,
            experiment_style_factor,
        ) = self.get_style_factors_from_ui()

        # Get gating strength from UI sliders (if available)
        gating_dynamic_strength = 0.0
        gating_static_strength = 0.0
        if self.gating_dynamic_slider is not None:
            gating_dynamic_strength = self.gating_dynamic_slider.get()
        if self.gating_static_slider is not None:
            gating_static_strength = self.gating_static_slider.get()

        # Compute gated scales for each source BEFORE optimization
        # This ensures optimization uses the same scales as rendering
        self.gated_scales_per_source = []
        for idx, source_bvh in enumerate(self.source_bvhs):
            # Get gating scores for this source
            target_frames = self.cached_Y_k_pred_list[idx].shape[0]
            if hasattr(self, 'cached_gating_scores_list') and idx < len(self.cached_gating_scores_list):
                gating_scores = self.cached_gating_scores_list[idx].cpu().numpy()  # [frames, 1]
                # Pad gating scores if frame count doesn't match (can happen due to windowing)
                if gating_scores.shape[0] < target_frames:
                    pad_size = target_frames - gating_scores.shape[0]
                    padding = np.tile(gating_scores[-1:, :], (pad_size, 1))
                    gating_scores = np.concatenate([gating_scores, padding], axis=0)
                elif gating_scores.shape[0] > target_frames:
                    gating_scores = gating_scores[:target_frames, :]
            else:
                gating_scores = np.ones((target_frames, 1))

            # Use encapsulated function to compute gated scales
            gated_scales = self.compute_gated_scales(
                gating_scores=gating_scores,
                gating_dynamic_strength=gating_dynamic_strength,
                gating_static_strength=gating_static_strength,
                dynamic_style_factor=dynamic_style_factor,
                dynamic_pelvis_style_factor=dynamic_pelvis_style_factor,
                constant_style_factor=constant_style_factor,
                constant_pelvis_style_factor=constant_pelvis_style_factor,
            )
            self.gated_scales_per_source.append(gated_scales)

        # Optimization (with gated scaling applied) ----
        # The optimization now applies the same gated scaling that will be used for rendering
        self.optimize_data(
            gated_scales_per_source=self.gated_scales_per_source,
        )

        # Add experiment -------------------
        bvh_color = self.colors[self.color_idx]
        self.color_idx += 1
        experiment = {
            "color": bvh_color,
            "ind": f"{int(self.induce_action_start_slider.get())}-{int(self.induce_action_end_slider.get())}",
            "ind_s": self.induce_action_strength_slider.get(),
            "ind_f": self.induce_action_frequency_slider.get(),
            "ind_v": self.induce_action_speed_slider.get(),
            "enc": self.encoder_scale_slider.get(),
            "dec": experiment_style_factor_dyn,
            "sta": experiment_style_factor,
            "post": self.postprocessing_selector.get(),
            "g_d": self.gating_dynamic_slider.get() if self.gating_dynamic_slider else 0.0,
            "g_s": self.gating_static_slider.get() if self.gating_static_slider else 0.0,
        }
        self.add_experiment(experiment)

        # Build BVH and render (scaling already applied in optimization) -----------------
        merged_bvh = None

        # Get gating strength from UI sliders (if available)
        gating_dynamic_strength = 0.0
        gating_static_strength = 0.0
        if self.gating_dynamic_slider is not None:
            gating_dynamic_strength = self.gating_dynamic_slider.get()
        if self.gating_static_slider is not None:
            gating_static_strength = self.gating_static_slider.get()

        for idx, (
            Y_k_pred,
            average_delta_rotations_pred,
            average_delta_pelvis_positions_pred,
            chunk_probs_pred,
            source_bvh,
        ) in enumerate(zip(
            self.Y_k_preds,
            self.average_delta_rotations_preds,
            self.average_delta_pelvis_positions_preds,
            self.chunk_probs_preds,
            self.source_bvhs,
        )):
            (
                delta_global_translations,
                delta_pelvis_positions,
                delta_global_rot_increment,
                delta_rotations,
            ) = extract_Y_k(Y_k=Y_k_pred)

            # Use pre-computed gated scales (same as used in optimization)
            gated_scales = self.gated_scales_per_source[idx]
            gated_dynamic_scale = gated_scales['dynamic']
            gated_dynamic_pelvis_scale = gated_scales['dynamic_pelvis']
            gated_static_scale = gated_scales['static']
            gated_static_pelvis_scale = gated_scales['static_pelvis']

            # Apply scaling with gating to build the final BVH
            # Note: The contact optimization was done with these same scale factors,
            # so the Y_k_pred is already optimized for the scaled motion
            stylized_deltas = add_scaled_style_rots(
                dynamic_deltas=delta_rotations,
                static_deltas=average_delta_rotations_pred[..., 0, :, :],
                dynamic_scale=gated_dynamic_scale,
                static_scale=gated_static_scale,
            )

            stylized_pelvis_pos = add_scaled_style_pos(
                dynamic_pelvis_pos=delta_pelvis_positions,
                static_pelvis_pos=average_delta_pelvis_positions_pred[..., 0, :],
                dynamic_pelvis_scale=gated_dynamic_pelvis_scale,
                static_pelvis_scale=gated_static_pelvis_scale,
            )

            pred_bvh = set_bvh_from_full_body_target_kinematics(
                source_bvh,
                delta_global_translations,
                stylized_pelvis_pos,
                delta_global_rot_increment,
                stylized_deltas,
                fix_root=self.fix_root,
            )

            if self.render_average:
                bvh_avg_debug = BVH()
                bvh_avg_debug.copy(source_bvh)

                # chunk_probs_pred: [num_frames, num_chunks]
                # target_avg_local_rotations: [num_chunks, num_joints, 4]
                # target_avg_local_positions: [num_chunks, num_joints, 3]

                # Compute weighted average for rotations using SLERP
                # chunk_probs_pred: [num_frames, num_chunks]
                # self.target_avg_local_rotations: [num_chunks, num_joints, 4]
                weighted_avg_rotations = np.stack(
                    [
                        quat.weighted_slerp(
                            self.target_avg_local_rotations,  # [num_chunks, num_joints, 4]
                            chunk_probs_pred[frame],  # [num_chunks]
                            shortest=True,
                        )
                        for frame in range(chunk_probs_pred.shape[0])
                    ]
                )  # [num_frames, num_joints, 4]

                # Compute weighted average for positions
                chunk_probs_expanded_pos = chunk_probs_pred[
                    :, :, None, None
                ]  # [num_frames, num_chunks, 1, 1]
                target_pos_expanded = self.target_avg_local_positions[
                    None, :, :, :
                ]  # [1, num_chunks, num_joints, 3]
                weighted_avg_positions = (
                    chunk_probs_expanded_pos * target_pos_expanded
                ).sum(
                    axis=1
                )  # [num_frames, num_joints, 3]

                bvh_avg_debug.set_data(weighted_avg_rotations, weighted_avg_positions)
                self.render_bvh(bvh_avg_debug, bvh_color * 2.0)

            # Save the pred_bvh to app_results folder
            filename = source_bvh.data["filename"]
            save_bvh = BVH()
            save_bvh.copy(pred_bvh)
            save_bvh.set_scale(1 / self.scale)
            save_bvh.save(os.path.join(self.app_results_dir, f"pred_{filename}.bvh"))
            print(
                f"\033[97mSaved: {os.path.join(self.app_results_dir, f'pred_{filename}.bvh')}\033[0m"
            )

            if not self.inplace:
                pred_bvh.data["positions"][:, 0, 0] += 1.0

            if self.merge_output:
                pred_bvh_i = self.detect_loop_and_trim(pred_bvh)
                if merged_bvh is None:
                    merged_bvh = BVH()
                    merged_bvh.copy(pred_bvh_i)
                else:
                    merge_bvhs(merged_bvh, pred_bvh_i)
            else:
                self.render_bvh(pred_bvh, bvh_color)

        if self.merge_output:
            self.render_bvh(merged_bvh, bvh_color)
        print("\033[92mScene rendered in Blender\033[0m")

    def add_bvh_file_to_scene(self):
        # open a file dialog to select a bvh file
        file_path = filedialog.askopenfilename(
            initialdir=self.input_folder,
            filetypes=[("bvh files", "*.bvh"), ("All files", "*")],
        )

        if file_path:
            # load the bvh file
            bvh = BVH()
            bvh.load(file_path)
            bvh.set_scale(self.scale)
            bvh = resample_bvh(bvh, self.target_frame_time)
            # render the bvh file in blender
            self.render_bvh(bvh, np.array([0.0, 0.0, 0.9]))

    def add_training_data_to_scene(self):
        train_source_bvhs = os.path.join(self.results_path, "source")
        train_target_bvhs = os.path.join(self.results_path, "target")
        source_color = np.array([0.8, 0.0, 0.0])
        target_color = np.array([0.0, 0.0, 0.8])
        self.add_experiment({"color": source_color, "name": "source"})
        self.add_experiment({"color": target_color, "name": "target"})

        for source_bvh_i in os.listdir(train_source_bvhs):
            source_bvh_i = os.path.join(train_source_bvhs, source_bvh_i)
            source_bvh = BVH()
            source_bvh.load(source_bvh_i)
            source_bvh.set_scale(self.scale)
            source_bvh = resample_bvh(source_bvh, self.target_frame_time)
            self.render_bvh(source_bvh, source_color)

        for target_bvh_i in os.listdir(train_target_bvhs):
            target_bvh_i = os.path.join(train_target_bvhs, target_bvh_i)
            target_bvh = BVH()
            target_bvh.load(target_bvh_i)
            target_bvh.set_scale(self.scale)
            target_bvh = resample_bvh(target_bvh, self.target_frame_time)
            self.render_bvh(target_bvh, target_color)

    def render_bvh(self, bvh, color, render_temp=True):
        render_bvh = bvh
        if self.render_framerate is not None:
            render_bvh = BVH()
            render_bvh.copy(bvh)
            render_bvh = resample_bvh(render_bvh, 1.0 / self.render_framerate)
        self.blender_conn.render_bvh(
            render_bvh,
            color,
            end_joints=self.end_joints,
        )
        if self.render_side:
            bvh_rotated = BVH()
            bvh_rotated.copy(render_bvh)
            local_rotations_i, local_positions_i, _, _, _, _ = bvh_rotated.get_data()
            local_rotations_i[:, 0] = quat.mul(
                quat.from_angle_axis(
                    np.array([np.pi / 2.0]), np.array([0.0, 1.0, 0.0])
                ),
                local_rotations_i[:, 0],
            )
            local_positions_i[:, 0] = local_positions_i[:, 0] + np.array(
                [2.0, 0.0, 0.0]
            )
            bvh_rotated.set_data(local_rotations_i, local_positions_i)
            self.blender_conn.render_bvh(
                bvh_rotated,
                color,
                end_joints=self.end_joints,
            )
        if self.render_temp and render_temp:
            # Calculate frames to trim for 0.5s and 1.0s delays
            frames_05s = int(0.5 / self.target_frame_time)
            frames_1s = int(1.0 / self.target_frame_time)

            # Create and render 0.5s delayed version
            delayed_bvh_05 = self.create_delayed_bvh(render_bvh, frames_05s)
            if delayed_bvh_05 is not None:
                color_05 = self.blend_towards_white(color, 0.3)
                self.render_bvh(delayed_bvh_05, color_05, render_temp=False)

            # Create and render 1.0s delayed version
            delayed_bvh_1 = self.create_delayed_bvh(render_bvh, frames_1s)
            if delayed_bvh_1 is not None:
                color_1 = self.blend_towards_white(color, 0.6)
                self.render_bvh(delayed_bvh_1, color_1, render_temp=False)

    def render_points(self, points, color, radius):
        if self.render_framerate is not None:
            print(
                f"\033[93m[WARNING] Points cannot be resampled yet. Do not specify render_framerate\033[0m"
            )
        else:
            self.blender_conn.render_points_timeline(points, color, radius)

    def create_delayed_bvh(self, bvh, frames_to_delay):
        """Creates a copy of the BVH with the first frame repeated frames_to_delay times at the start."""
        delayed_bvh = BVH()
        delayed_bvh.copy(bvh)

        # Get the first frame
        first_rot = delayed_bvh.data["rotations"][0:1]  # Shape: (1, num_joints, 4)
        first_pos = delayed_bvh.data["positions"][0:1]  # Shape: (1, num_joints, 3)

        # Repeat the first frame frames_to_delay times
        repeated_rots = np.repeat(first_rot, frames_to_delay, axis=0)
        repeated_pos = np.repeat(first_pos, frames_to_delay, axis=0)

        # Prepend the repeated frames to the original animation
        delayed_bvh.data["rotations"] = np.concatenate([repeated_rots, delayed_bvh.data["rotations"]], axis=0)
        delayed_bvh.data["positions"] = np.concatenate([repeated_pos, delayed_bvh.data["positions"]], axis=0)

        return delayed_bvh

    def blend_towards_white(self, color, factor):
        """Blends color towards white by the given factor (0 = original, 1 = white)."""
        white = np.array([1.0, 1.0, 1.0])
        return color + (white - color) * factor

    def detect_loop_and_trim(self, bvh):
        """
        Detects loppable animations and trims the first and last loops (assume loopable animations have been looped 5 times)
        """
        num_frames = bvh.data["rotations"].shape[0]
        if num_frames % 5 != 0:
            return bvh
        loop_length = num_frames // 5

        if not bvh.data["filename"].endswith("_looped"):
            return bvh

        print(
            f"\033[97mDetected loop in {bvh.data['filename']}, trimming {loop_length} frames\033[0m"
        )

        # Trim the first and last loops
        new_bvh = BVH()
        new_bvh.copy(bvh)
        rotations = new_bvh.data["rotations"]
        rotations = rotations[loop_length:-loop_length]
        positions = new_bvh.data["positions"]
        positions = positions[loop_length:-loop_length]
        new_bvh.data["rotations"] = rotations
        new_bvh.data["positions"] = positions
        return new_bvh

    def get_per_joint_factors(self, style_factors_dict):
        """
        Converts a dictionary of body part factors into a numpy array that
        can be broadcast to the joint rotations.
        """
        # Get number of joints from the first BVH skeleton
        num_joints = len(self.source_bvhs[0].data["names"])
        factors = np.ones((num_joints, 1))  # Default to 1.0
        pelvis_factor = 1.0

        for i, name in enumerate(self.body_parts_names):
            slider_value = style_factors_dict.get(name, 1.0)
            slider_value = max(slider_value, 0.001)
            if name == "Pelvis":
                pelvis_factor = slider_value
            else:
                joint_indices = self.body_parts_indices[i]
                factors[joint_indices] = slider_value

        factors = factors[1:]  # remove root

        return factors, pelvis_factor

    def _smooth_gated_scales(self, scales, window_length=31, polyorder=2):
        """
        Apply Savitzky-Golay filter to smooth gated scale factors.
        This prevents jittering when gating values transition (around 0.5).

        Args:
            scales: Scale factors [frames, ...] - can be any shape with frames as first dim
            window_length: Length of the filter window (must be odd). Default 31.
            polyorder: Order of the polynomial used to fit samples. Default 2.

        Returns:
            Smoothed scales with same shape as input
        """
        num_frames = scales.shape[0]

        # Adjust window_length if data is too short
        effective_window = min(window_length, num_frames)
        if effective_window % 2 == 0:
            effective_window -= 1  # Must be odd
        effective_window = max(effective_window, polyorder + 2)  # Must be > polyorder

        if num_frames <= polyorder + 1:
            return scales

        # Reshape to 2D for filtering: [frames, -1]
        original_shape = scales.shape
        scales_2d = scales.reshape(num_frames, -1)

        # Apply filter to each column
        smoothed_2d = np.zeros_like(scales_2d)
        for i in range(scales_2d.shape[1]):
            smoothed_2d[:, i] = savgol_filter(
                scales_2d[:, i],
                window_length=effective_window,
                polyorder=polyorder,
                mode='nearest'
            )

        # Reshape back to original shape
        return smoothed_2d.reshape(original_shape)

    def compute_gated_scales(
        self,
        gating_scores,
        gating_dynamic_strength,
        gating_static_strength,
        dynamic_style_factor,
        dynamic_pelvis_style_factor,
        constant_style_factor,
        constant_pelvis_style_factor,
    ):
        """
        Compute gated scale factors by combining style factors with gating scores.

        Args:
            gating_scores: Gating model output [frames, 1], values in [0, 1]
            gating_dynamic_strength: UI slider value for dynamic gating strength [0, 1]
            gating_static_strength: UI slider value for static gating strength [0, 1]
            dynamic_style_factor: Dynamic style factor (scalar or per-joint array [joints-1, 1])
            dynamic_pelvis_style_factor: Dynamic pelvis style factor (scalar)
            constant_style_factor: Static style factor (scalar or per-joint array [joints-1, 1])
            constant_pelvis_style_factor: Static pelvis style factor (scalar)

        Returns:
            Dict with gated scales:
                - 'dynamic': [frames, 1, 1] or [frames, joints-1, 1]
                - 'dynamic_pelvis': [frames, 1]
                - 'static': [frames, 1, 1] or [frames, joints-1, 1]
                - 'static_pelvis': [frames, 1]
        """
        # Compute effective gating for dynamic and static styles
        # effective_gate = 1 - gating_strength * (1 - gating_score)
        # When gating_score=1 (stylizable), effective_gate=1 (full style)
        # When gating_score=0 (not stylizable), effective_gate = 1 - gating_strength
        effective_dynamic_gate = 1.0 - gating_dynamic_strength * (1.0 - gating_scores)  # [frames, 1]
        effective_static_gate = 1.0 - gating_static_strength * (1.0 - gating_scores)  # [frames, 1]

        # Apply gating to the style factors
        # For dynamic style
        if isinstance(dynamic_style_factor, np.ndarray):
            # Per-joint: [joints-1, 1] * [frames, 1] -> broadcast to [frames, joints-1, 1]
            gated_dynamic_scale = dynamic_style_factor[None, :, :] * effective_dynamic_gate[:, None, :]
        else:
            # Scalar: expand to [frames, 1, 1] for proper broadcasting over [frames, joints, 3]
            gated_dynamic_scale = dynamic_style_factor * effective_dynamic_gate[:, :, np.newaxis]

        gated_dynamic_pelvis_scale = dynamic_pelvis_style_factor * effective_dynamic_gate

        # For static style
        if isinstance(constant_style_factor, np.ndarray):
            gated_static_scale = constant_style_factor[None, :, :] * effective_static_gate[:, None, :]
        else:
            gated_static_scale = constant_style_factor * effective_static_gate[:, :, np.newaxis]

        gated_static_pelvis_scale = constant_pelvis_style_factor * effective_static_gate

        # Apply additional smoothing to gated scales to prevent jittering
        # during gating transitions (when gating score is around 0.5)
        # Only smooth if gating is actually active (strength > 0)
        if gating_dynamic_strength > 0.0 or gating_static_strength > 0.0:
            gated_dynamic_scale = self._smooth_gated_scales(gated_dynamic_scale)
            gated_dynamic_pelvis_scale = self._smooth_gated_scales(gated_dynamic_pelvis_scale)
            gated_static_scale = self._smooth_gated_scales(gated_static_scale)
            gated_static_pelvis_scale = self._smooth_gated_scales(gated_static_pelvis_scale)

        return {
            'dynamic': gated_dynamic_scale,
            'dynamic_pelvis': gated_dynamic_pelvis_scale,
            'static': gated_static_scale,
            'static_pelvis': gated_static_pelvis_scale,
        }

    def get_style_factors_from_ui(self):
        """
        Gathers style factors from the UI sliders.

        Returns:
            Tuple of:
                - dynamic_style_factor: Scale for dynamic deltas (scalar or per-joint array)
                - dynamic_pelvis_style_factor: Scale for dynamic pelvis position
                - constant_style_factor: Scale for static deltas (scalar or per-joint array)
                - constant_pelvis_style_factor: Scale for static pelvis position
                - experiment_style_factor_dyn: For experiment logging (dict or scalar)
                - experiment_style_factor: For experiment logging (dict or scalar)
        """
        # Dynamic style factors
        experiment_style_factor_dyn = None
        if self.advanced_mode_var_dyn.get():
            style_factors_dict = {
                name: slider.get() for name, slider in self.advanced_sliders_dyn.items()
            }
            print(
                f"\033[96mDynamic Style (Advanced): "
                + ", ".join(
                    f"{name}={val:.2f}" for name, val in style_factors_dict.items()
                )
                + "\033[0m"
            )
            dynamic_style_factor, dynamic_pelvis_style_factor = (
                self.get_per_joint_factors(style_factors_dict)
            )
            experiment_style_factor_dyn = style_factors_dict
        else:
            dynamic_style_factor = self.style_slider_dyn.get()
            dynamic_style_factor = max(dynamic_style_factor, 0.001)
            dynamic_pelvis_style_factor = dynamic_style_factor
            print(f"\033[96mDynamic Style: {dynamic_style_factor:.2f}\033[0m")
            experiment_style_factor_dyn = dynamic_style_factor

        # Static style factors
        experiment_style_factor = None
        if self.advanced_mode_var.get():
            style_factors_dict = {
                name: slider.get() for name, slider in self.advanced_sliders.items()
            }
            print(
                f"\033[95mStatic Style (Advanced): "
                + ", ".join(
                    f"{name}={val:.2f}" for name, val in style_factors_dict.items()
                )
                + "\033[0m"
            )
            constant_style_factor, constant_pelvis_style_factor = (
                self.get_per_joint_factors(style_factors_dict)
            )
            experiment_style_factor = style_factors_dict
        else:
            constant_style_factor = self.style_slider.get()
            constant_style_factor = max(constant_style_factor, 0.001)
            constant_pelvis_style_factor = constant_style_factor
            print(f"\033[95mStatic Style: {constant_style_factor:.2f}\033[0m")
            experiment_style_factor = constant_style_factor

        return (
            dynamic_style_factor,
            dynamic_pelvis_style_factor,
            constant_style_factor,
            constant_pelvis_style_factor,
            experiment_style_factor_dyn,
            experiment_style_factor,
        )

    def run(self):
        """
        Starts the tkinter main loop.
        """
        self.render_scene(clear_scene=True)
        print("\033[92mUI is ready. Adjust sliders and press 'Render Scene'\033[0m")
        self.root.mainloop()

    def on_closing(self):
        """
        Handles cleanup when the UI window is closed.
        """
        print("\033[93mClosing Blender connection...\033[0m")
        self.blender_conn.close()
        self.root.destroy()
        print("\033[92mApplication closed\033[0m")


if __name__ == "__main__":
    warnings.filterwarnings(
        "ignore", category=FutureWarning, message=".*torch.load.*weights_only.*"
    )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "results_path",
        type=str,
        default="motion/eval_mocha/angry/results/Transformer-base",
        help="Path to the results directory (must contain 'input' folder with .bvh files)",
    )
    parser.add_argument(
        "--scale", type=float, default=1.0
    )  # 0.01 for xsens DB, 1 for mocha DB
    parser.add_argument(
        "--merge_output",
        action="store_true",
        help="Merge output from multiple .bvh files for rendering",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Render skeletons in-place",
    )
    parser.add_argument(
        "--render_side",
        action="store_true",
        help="Render skeletons twice rotated 90 degrees to the side",
    )
    parser.add_argument(
        "--fix_root",
        action="store_true",
        help="Fix root joint to the ground truth",
    )
    parser.add_argument(
        "--post_processing",
        type=str,
        default="fast",
        choices=["none", "fast", "normal", "high"],
        help="Default postprocessing quality (default: fast)",
    )
    parser.add_argument(
        "--render_framerate",
        type=float,
        default=None,  # e.g., 30.0
        help="Resample render to this framerate, e.g., 30.0 (default: None)",
    )
    parser.add_argument(
        "--render_average",
        action="store_true",
        help="Debug render the predicted average poses",
    )
    parser.add_argument(
        "--render_temp",
        action="store_true",
        help="Render 2 additional delayed versions (0.5s and 1s) with colors towards white",
    )
    args = parser.parse_args()

    # Create and run the application
    app = MotionStyleApp(
        args.results_path,
        args.scale,
        args.merge_output,
        args.inplace,
        args.render_side,
        args.fix_root,
        args.post_processing,
        args.render_framerate,
        args.render_average,
        args.render_temp,
    )
    app.run()
