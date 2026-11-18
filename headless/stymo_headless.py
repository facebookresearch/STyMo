"""Headless STyMo app: runs stylizer_app's real predict -> optimize -> save pipeline
without Tk or a live Blender, and records the Blender commands the app would send so
headless/blender_replay.py can rebuild the exact same scene in background Blender.

Usage (from the repo root, inside the stymo env):
    python headless/stymo_headless.py <results_path> [--clips Dance Walk] \
        [--variant 1.0,1.0 --variant 1.5,1.0] [--post_processing fast] [--out_dir DIR]

Each --variant is "dynamic,static" style strength (the two main GUI sliders). The first
variant is rendered like the app's first "Render Scene" click; later ones are added to
the same scene (like clicking "Render Scene" again) and spread out sideways on replay.
"""
import argparse
import json
import os
import shutil
import sys
import warnings
from unittest.mock import MagicMock

import numpy as np

sys.path.insert(0, os.path.join(os.getcwd(), "src"))
import stylizer_app  # noqa: E402
from pymotion.render.blender import BlenderConnection  # noqa: E402


class RecordingConnection(BlenderConnection):
    """Drop-in BlenderConnection that records commands instead of sending them.

    Commands are stored as the (id, data, color, scale) tuples that pymotion's Blender
    script rebuilds from the socket stream (floats rounded to float32, like the wire)."""

    def __init__(self, *args, **kwargs):
        self.s = None
        self.commands = []
        self.bvh_dir = None
        self.variant = 0

    def _send_message_code(self, message_id):
        self.commands.append(
            {"id": message_id, "data": None, "color": None, "scale": None, "variant": self.variant}
        )

    def _send_data(self, data, color=None, scale=None):
        cmd = self.commands[-1]
        if isinstance(data, str):
            # "<path>.bvh<end;joints>;<axis_forward>;<axis_up>": keep a copy of the BVH
            split = data.rfind(".bvh") + len(".bvh")
            src_path, rest = data[:split], data[split:]
            n = sum(1 for c in self.commands if c["id"] == 3)
            dst_path = os.path.join(self.bvh_dir, f"{n:03d}_{os.path.basename(src_path)}")
            shutil.copy(src_path, dst_path)
            if scale is not None and scale[0] == 1:  # temp file pymotion meant Blender to delete
                os.remove(src_path)
            cmd["data"] = dst_path + rest
            scale = np.array([0.0])  # replay must not delete our copy
        else:
            cmd["data"] = np.asarray(data, dtype=np.float32).tolist()
        cmd["color"] = None if color is None else np.asarray(color, dtype=np.float32).tolist()
        cmd["scale"] = None if scale is None else np.asarray(scale, dtype=np.float32).tolist()

    def close(self):
        pass


class Val:
    """Stand-in for a ttk widget/variable: .get() returns a fixed value."""

    def __init__(self, v):
        self.v = v

    def get(self):
        return self.v

    def set(self, v):
        self.v = v


class HeadlessApp(stylizer_app.MotionStyleApp):
    def create_ui(self):
        # Defaults copied from MotionStyleApp.create_ui
        self.show_foot_contacts_var = Val(True)
        self.postprocessing_selector = Val(self.post_processing.capitalize())
        self.contact_strength_selector = Val("Normal")
        self.smoothing_selector = Val("Normal")
        self.encoder_scale_slider = Val(1.0)
        self.advanced_mode_var_dyn = Val(False)
        self.style_slider_dyn = Val(1.0)
        self.induce_action_start_slider = Val(0)
        self.induce_action_end_slider = Val(0)
        self.induce_action_frequency_slider = Val(0.5)
        self.induce_action_speed_slider = Val(1.0)
        self.induce_action_strength_slider = Val(0.0)
        self.advanced_mode_var = Val(False)
        self.style_slider = Val(1.0)
        has_gate = self.gating_model is not None
        self.gating_dynamic_slider = Val(0.0) if has_gate else None
        self.gating_static_slider = Val(0.0) if has_gate else None
        self.root = MagicMock()

    def add_experiment(self, experiment):  # experiment-history widgets only
        pass

    def clear_experiments(self):
        pass


def parse_variant(s):
    dyn, sta = (float(x) for x in s.split(","))
    return dyn, sta


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.load.*weights_only.*")
    p = argparse.ArgumentParser()
    p.add_argument("results_path")
    p.add_argument("--clips", nargs="*", default=None,
                   help="Substrings of input BVH names to process (default: all)")
    p.add_argument("--variant", type=parse_variant, action="append", default=None,
                   help='"dynamic,static" style strength; repeatable (default: 1.0,1.0)')
    p.add_argument("--gating", type=float, nargs=2, default=(0.0, 0.0), metavar=("DYN", "STA"),
                   help="Dynamic/static gate sliders (default: 0 0)")
    p.add_argument("--post_processing", default="fast", choices=["none", "fast", "normal", "high"])
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--fix_root", action="store_true")
    p.add_argument("--inplace", action="store_true")
    p.add_argument("--out_dir", default=None, help="Default: <results_path>/headless")
    a = p.parse_args()
    variants = a.variant or [(1.0, 1.0)]

    stylizer_app.BlenderConnection = RecordingConnection
    app = HeadlessApp(a.results_path, a.scale, False, a.inplace, False, a.fix_root,
                      a.post_processing, None, False, False)
    if app.gating_model is not None:
        app.gating_dynamic_slider.set(a.gating[0])
        app.gating_static_slider.set(a.gating[1])
    out_dir = a.out_dir or os.path.join(a.results_path, "headless")

    clips = [f for f in app.available_bvh_files
             if a.clips is None or any(c.lower() in f.lower() for c in a.clips)]
    if not clips:
        sys.exit(f"No input clips match {a.clips}; available: {app.available_bvh_files}")

    for clip in clips:
        name = os.path.splitext(clip)[0]
        clip_dir = os.path.join(out_dir, name)
        shutil.rmtree(clip_dir, ignore_errors=True)
        os.makedirs(os.path.join(clip_dir, "bvh"))
        conn = app.blender_conn
        conn.commands, conn.bvh_dir = [], os.path.join(clip_dir, "bvh")

        if clip != app.current_bvh_file or not hasattr(app, "input_sources"):
            app.load_bvh_file(clip)
            app.current_bvh_file = clip
        for k, (dyn, sta) in enumerate(variants):
            conn.variant = k
            app.style_slider_dyn.set(dyn)
            app.style_slider.set(sta)
            app.render_scene(clear_scene=(k == 0))
            # app_results/pred_<clip>.bvh is overwritten per variant; keep each one
            pred = os.path.join(app.app_results_dir, f"pred_{app.source_bvhs[0].data['filename']}.bvh")
            shutil.copy(pred, os.path.join(clip_dir, f"pred_dyn{dyn:g}_sta{sta:g}.bvh"))

        meta = {
            "clip": clip,
            "fps": 1.0 / app.target_frame_time,
            "variants": [{"dynamic": d, "static": s, "color": c.tolist()}
                         for (d, s), c in zip(variants, app.colors)],
            "gating": list(a.gating),
            "post_processing": a.post_processing,
            "inplace": a.inplace,
            "commands": conn.commands,
        }
        with open(os.path.join(clip_dir, "commands.json"), "w") as f:
            json.dump(meta, f)
        print(f"\033[92mRecorded {len(conn.commands)} Blender commands -> {clip_dir}\033[0m")
