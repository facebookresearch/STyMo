"""Rebuild a recorded STyMo app scene in background Blender, save it as .blend and
optionally render an .mp4. Uses pymotion's own Blender script to execute the commands,
so the scene matches what stylizer_app shows live.

Usage:
    blender -b --factory-startup -P headless/blender_replay.py -- <clip_dir> \
        --pymotion_script <site-packages>/pymotion/render/internal/pymotion_blender.py \
        [--blend out.blend] [--mp4 out.mp4] [--resolution 1920 1080] [--samples 32] \
        [--engine eevee|cycles|workbench] [--spacing 1.0]
"""
import argparse
import importlib.util
import json
import os
import sys
import traceback
import types

import bpy


def load_pymotion_blender(path):
    spec = importlib.util.spec_from_file_location("pymotion_blender", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # __name__ != "__main__", so no socket server starts
    return mod


def root_bone(arm):
    return next(b for b in arm.pose.bones if b.parent is None)


def add_follow_camera(armatures, spacing_span, lens=30.0):
    """Camera that tracks the source character's root (smoothed over ~1 s), framing all skeletons."""
    scene = bpy.context.scene
    src = armatures[0]
    rb = root_bone(src)
    start, end = scene.frame_start, scene.frame_end
    xs, ys = [], []
    for f in range(start, end + 1):
        scene.frame_set(f)
        p = src.matrix_world @ rb.head
        xs.append(p.x)
        ys.append(p.y)
    win = max(1, int(scene.render.fps))  # ~1 second moving average
    half = win // 2

    def smooth(v):
        out = []
        for i in range(len(v)):
            seg = v[max(0, i - half): i + half + 1]
            out.append(sum(seg) / len(seg))
        return out

    xs, ys = smooth(xs), smooth(ys)

    target = bpy.data.objects.new("CameraTarget", None)
    scene.collection.objects.link(target)
    step = max(1, win // 6)
    frames = list(range(start, end + 1, step)) + ([end] if (end - start) % step else [])
    for f in frames:
        i = f - start
        target.location = (xs[i] + spacing_span / 2.0, ys[i], 0.9)
        target.keyframe_insert("location", frame=f)

    cam_data = bpy.data.cameras.new("Camera")
    cam_data.lens = lens
    cam = bpy.data.objects.new("Camera", cam_data)
    scene.collection.objects.link(cam)
    cam.parent = target
    dist = 3.2 + spacing_span
    cam.location = (0.0, -dist, 0.9)  # in front of the characters, slightly above
    track = cam.constraints.new("TRACK_TO")
    track.target = target
    track.track_axis = "TRACK_NEGATIVE_Z"
    track.up_axis = "UP_Y"
    scene.camera = cam
    scene.frame_set(start)


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("clip_dir")
    p.add_argument("--pymotion_script", required=True)
    p.add_argument("--blend", default=None)
    p.add_argument("--mp4", default=None)
    p.add_argument("--resolution", type=int, nargs=2, default=None)
    p.add_argument("--samples", type=int, default=None)
    p.add_argument("--engine", default="eevee", choices=["eevee", "cycles", "workbench"])
    p.add_argument("--spacing", type=float, default=1.0,
                   help="Extra sideways offset (m) between variants")
    a = p.parse_args(argv)

    with open(os.path.join(a.clip_dir, "commands.json")) as f:
        meta = json.load(f)
    pm = load_pymotion_blender(a.pymotion_script)

    # pymotion swallows command errors (prints a traceback); count them so we fail loudly
    errors = []

    def print_exc():
        errors.append(1)
        traceback.print_exc()

    pm.traceback = types.SimpleNamespace(print_exc=print_exc)

    armatures = []
    for cmd in meta["commands"]:
        before = set(bpy.data.objects)
        pm.process_single_command((cmd["id"], cmd["data"], cmd["color"], cmd["scale"]))
        new_arms = [o for o in bpy.data.objects if o not in before and o.type == "ARMATURE"]
        for arm in new_arms:  # spread extra variants sideways (app overlays them)
            arm.location.x += cmd["variant"] * a.spacing
        armatures += new_arms
    if errors:
        sys.exit(f"{len(errors)} pymotion command(s) failed")
    if not armatures:
        sys.exit("No skeletons were created")

    scene = bpy.context.scene
    scene.render.fps = int(round(meta["fps"]))  # BVH frames are imported 1:1
    scene.render.fps_base = 1.0
    scene.frame_start = 1

    span = 0.0 if meta.get("inplace") else 1.0 + (len(meta["variants"]) - 1) * a.spacing
    add_follow_camera(armatures, span)

    legend = "green = source (neutral)   " + "   ".join(
        f"{'blue purple navy yellow black violet magenta lime orange'.split()[i]} = "
        f"dyn {v['dynamic']:g} / static {v['static']:g}"
        for i, v in enumerate(meta["variants"]))
    scene.render.use_stamp = True
    for k in ("date", "time", "render_time", "frame", "frame_range", "memory", "hostname",
              "camera", "lens", "scene", "marker", "filename", "sequencer_strip", "note"):
        setattr(scene.render, f"use_stamp_{k}", k in ("frame", "note"))
    scene.render.stamp_note_text = f"{meta['clip']}  |  {legend}  |  post: {meta['post_processing']}"
    scene.render.stamp_font_size = 22

    if a.blend:
        bpy.context.preferences.filepaths.save_version = 0  # no .blend1 backups
        bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(a.blend), compress=True)
        print(f"[replay] Saved {a.blend}")

    if a.mp4:
        if a.resolution:
            scene.render.resolution_x, scene.render.resolution_y = a.resolution
        if a.engine == "cycles":
            scene.render.engine = "CYCLES"
            prefs = bpy.context.preferences.addons["cycles"].preferences
            prefs.compute_device_type = "CUDA"
            prefs.get_devices()
            for d in prefs.devices:
                d.use = d.type == "CUDA"
            scene.cycles.device = "GPU"
            scene.cycles.samples = a.samples or 64
        elif a.engine == "workbench":
            scene.render.engine = "BLENDER_WORKBENCH"
        elif a.samples:
            scene.eevee.taa_render_samples = a.samples
        scene.render.image_settings.file_format = "FFMPEG"
        scene.render.ffmpeg.format = "MPEG4"
        scene.render.ffmpeg.codec = "H264"
        scene.render.ffmpeg.constant_rate_factor = "HIGH"
        out = os.path.abspath(a.mp4)
        scene.render.filepath = out
        scene.render.use_file_extension = False
        bpy.ops.render.render(animation=True)
        print(f"[replay] Rendered {out}")


if __name__ == "__main__":
    main()
