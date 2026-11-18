"""Write a FineDance clip (native 315-dim: 3 transl + 52 joints x 6D) as an SMPL-22 BVH.

Uses GVHMR's BVH writer (tools/demo/export_bvh.py), so run it in the `gvhmr` env:
  conda run -n gvhmr python headless/finedance_to_bvh.py \
      ~/databases/finedance/motion/094.npy out.bvh --start 10 --seconds 45
Output is in centimetres like the Step Up BVHs; pass it through smpl_to_stymo.py next.

6D layout: rows of the rotation matrix (pytorch3d convention). With columns, the
knees bend backwards on average; --check-6d prints the mean knee flexion for both.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

GVHMR = Path.home() / "projects/GVHMR"
sys.path.insert(0, str(GVHMR))
sys.path.insert(0, str(GVHMR / "tools/demo"))
from export_bvh import build_bvh  # noqa: E402

FPS = 30.0


def sixd_to_matrix(x, rows=True):
    a1, a2 = x[..., :3], x[..., 3:6]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 /= np.linalg.norm(b2, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-2 if rows else -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npy")
    ap.add_argument("out")
    ap.add_argument("--start", type=float, default=0.0, help="seconds")
    ap.add_argument("--seconds", type=float, default=None)
    a = ap.parse_args()

    m = np.load(a.npy).astype(np.float64)
    assert m.shape[1] == 315, m.shape
    s = int(a.start * FPS)
    e = len(m) if a.seconds is None else s + int(a.seconds * FPS)
    m = m[s:e]
    transl = m[:, :3]
    six = m[:, 3:].reshape(len(m), 52, 6)[:, :22]

    for rows in (True, False):
        knee = Rotation.from_matrix(sixd_to_matrix(six[:, 4], rows)).as_rotvec()[:, 0]
        print(f"6D as {'rows' if rows else 'cols'}: mean left-knee flexion {np.degrees(knee.mean()):+.1f} deg")

    aa = Rotation.from_matrix(sixd_to_matrix(six).reshape(-1, 3, 3)).as_rotvec().reshape(len(m), 66)
    params = {
        "betas": torch.zeros(len(m), 10),
        "global_orient": torch.from_numpy(aa[:, :3]),
        "body_pose": torch.from_numpy(aa[:, 3:]),
        # FineDance transl is the pelvis position; build_bvh adds the rest pelvis back
        "transl": torch.from_numpy(transl),
    }
    Path(a.out).write_text(build_bvh(params, FPS, scale=100.0))
    print(f"wrote {a.out}: {len(m)} frames ({len(m) / FPS:.1f} s)")


if __name__ == "__main__":
    main()
