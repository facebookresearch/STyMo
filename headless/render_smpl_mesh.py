"""Render STyMo SMPL BVHs (Root + SMPL-22, metres) as human body meshes, side by side.

Each BVH frame is turned back into SMPL-X parameters (global_orient = Root * pelvis,
body_pose = the 21 local joint rotations) and skinned with GVHMR's body model, then
drawn with GVHMR's pytorch3d renderer on a checkerboard floor. Run in the `gvhmr` env:

  conda run -n gvhmr python headless/render_smpl_mesh.py out.mp4 \
      input.bvh:hiphop pred_dyn1_sta1.bvh:ballet pred_dyn2_sta1.5.bvh:"ballet x2" \
      [--fps 30] [--spacing 1.4] [--seconds 10]

Clips of different frame rates are sampled at the same times, so a 30 fps input and a
60 fps output stay in sync.
"""
import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
GVHMR = Path.home() / "projects/GVHMR"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(GVHMR))

from smpl_to_stymo import parse_bvh, unpack  # noqa: E402
from hmr4d.utils.geo.hmr_cam import create_camera_sensor  # noqa: E402
from hmr4d.utils.smplx_utils import make_smplx  # noqa: E402
from hmr4d.utils.video_io_utils import get_writer  # noqa: E402
from hmr4d.utils.vis.renderer import Renderer, get_global_cameras_static, get_ground_params_from_points  # noqa: E402

SMPL_ORDER = [
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee", "spine2", "left_ankle",
    "right_ankle", "spine3", "left_foot", "right_foot", "neck", "left_collar", "right_collar", "head",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
]  # fmt: skip
COLORS = [(0.45, 0.8, 0.35), (0.35, 0.6, 0.9), (0.75, 0.4, 0.85), (0.9, 0.6, 0.3)]  # green, blue, purple, orange


def load_smpl_params(path, times):
    """BVH -> (global_orient, body_pose, pelvis_world) sampled at `times` (seconds)."""
    joints, ft, data = parse_bvh(path)
    names = [j["name"] for j in joints]
    assert names[:2] == ["Root", "pelvis"], f"{path}: expected the STyMo SMPL layout (Root, pelvis, ...)"
    rots, root_pos = unpack(joints, data)
    idx = np.clip(np.round(times / ft).astype(int), 0, len(data) - 1)
    rots, root_pos = rots[idx], root_pos[idx]
    pel_local = data[idx, 6:9]  # pelvis position channels, right after Root's 6
    root_rot = R.from_quat(rots[:, 0])
    pelvis_world = root_pos + root_rot.apply(pel_local)
    global_orient = (root_rot * R.from_quat(rots[:, 1])).as_rotvec()
    body_pose = np.concatenate([R.from_quat(rots[:, names.index(n)]).as_rotvec() for n in SMPL_ORDER[1:]], axis=1)
    duration = len(data) * ft
    return global_orient, body_pose, pelvis_world, duration


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("clips", nargs="+", help="path.bvh[:label]")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--spacing", type=float, default=1.4, help="metres between figures")
    ap.add_argument("--seconds", type=float, default=None, help="render only the first N seconds")
    ap.add_argument("--size", default="1920x1080")
    ap.add_argument("--cam_distance", type=float, default=4.0, help="camera distance factor (GVHMR beta)")
    a = ap.parse_args()

    clips = [(c.rsplit(":", 1) if ":" in c and not c.endswith(".bvh") else (c, Path(c).stem)) for c in a.clips]
    clips = [(str(Path(p).resolve()) if not os.path.isabs(p) else p, lbl) for p, lbl in clips]
    os.chdir(GVHMR)  # GVHMR loads body-model files by relative path
    durations = [load_smpl_params(p, np.zeros(1))[3] for p, _ in clips]
    dur = min(durations) if a.seconds is None else min(min(durations), a.seconds)
    times = np.arange(0, dur, 1.0 / a.fps)
    L = len(times)

    smplx = make_smplx("supermotion").cuda()
    smplx2smpl = torch.load("hmr4d/utils/body_model/smplx2smpl_sparse.pt").cuda()
    faces = make_smplx("smpl").faces

    all_verts = []
    for k, (p, _) in enumerate(clips):
        go, bp, pel, _ = load_smpl_params(p, times)
        out = smplx(
            global_orient=torch.tensor(go).float().cuda(),
            body_pose=torch.tensor(bp).float().cuda(),
            betas=torch.zeros(L, 10).cuda(),
            transl=torch.zeros(L, 3).cuda(),
        )
        verts = torch.stack([smplx2smpl @ v for v in out.vertices])  # (L, 6890, 3)
        # put the model's pelvis where the BVH pelvis is
        shift = torch.tensor(pel).float().cuda() - out.joints[:, 0]
        verts = verts + shift[:, None]
        verts[..., 0] += (k - (len(clips) - 1) / 2) * a.spacing  # side by side along x
        all_verts.append(verts)
    verts = torch.stack(all_verts, 1)  # (L, N, V, 3)
    verts[..., 1] -= verts[..., 1].min()  # lowest point of any body over the clip on the floor

    W, H = (int(v) for v in a.size.split("x"))
    _, _, K = create_camera_sensor(W, H, 24)
    renderer = Renderer(W, H, device="cuda", faces=faces, K=K)
    flat = verts.reshape(L, -1, 3)
    cam_R, cam_T, lights = get_global_cameras_static(flat.cpu(), beta=a.cam_distance, cam_height_degree=12,
                                                     target_center_height=1.0, vec_rot=0)
    scale, cx, cz = get_ground_params_from_points(flat.mean(1), flat)
    renderer.set_ground(scale * 1.5, cx, cz)
    colors = torch.tensor(COLORS[: len(clips)]).float().cuda()

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    writer = get_writer(str(Path(a.out).resolve()), fps=a.fps, crf=20)
    legend = "   ".join(f"{['green', 'blue', 'purple', 'orange'][k]} = {lbl}" for k, (_, lbl) in enumerate(clips))
    for i in tqdm(range(L), desc="render"):
        img = renderer.render_with_ground(verts[i], colors, renderer.create_camera(cam_R[i], cam_T[i]), lights)
        img = np.ascontiguousarray(img)
        fs = min(1.1, 1.1 * 75 / max(len(legend), 1))
        cv2.putText(img, legend, (24, 48), cv2.FONT_HERSHEY_SIMPLEX, fs, (30, 30, 30), 2, cv2.LINE_AA)
        writer.write_frame(img)
    writer.close()
    print(f"wrote {a.out}: {L} frames, {dur:.1f} s")


if __name__ == "__main__":
    main()
