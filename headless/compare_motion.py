"""Compare two BVHs joint by joint: how much does B differ from A? (stymo env)
Splits the per-joint rotation difference into a constant offset (posture) and a
time-varying part (dynamics). Usage: python compare_motion.py A.bvh B.bvh [label]"""
import sys

import numpy as np
import pymotion.rotations.quat as quat
from pymotion.io.bvh import BVH


def load(p):
    b = BVH()
    b.load(p)
    rots, pos, *_ = b.get_data()
    return b, rots, pos


def angle_deg(q):  # geodesic angle of unit quaternions [..., 4] (w first)
    w = np.clip(np.abs(q[..., 0]), 0.0, 1.0)
    return np.degrees(2.0 * np.arccos(w))


a_bvh, ra, pa = load(sys.argv[1])
_, rb, pb = load(sys.argv[2])
label = sys.argv[3] if len(sys.argv) > 3 else ""
n = min(len(ra), len(rb))
ra, rb, pa, pb = ra[:n], rb[:n], pa[:n], pb[:n]
names = a_bvh.data["names"]

diff = quat.mul(quat.inverse(ra), rb)  # per-frame local rotation difference A -> B
total = angle_deg(diff)  # [frames, joints]
# constant part: average difference rotation; dynamic part: what's left per frame
mean_q = diff.mean(axis=0)
mean_q /= np.linalg.norm(mean_q, axis=-1, keepdims=True)
static = angle_deg(mean_q)
dynamic = angle_deg(quat.mul(quat.inverse(mean_q)[None], diff))

print(f"\n=== {label}  ({n} frames)")
print(f"overall mean |rot diff|: {total.mean():.2f} deg   static offset: {static.mean():.2f} deg   "
      f"time-varying: {dynamic.mean():.2f} deg   max: {total.max():.1f} deg")
print(f"pelvis height diff: mean {np.mean(pb[:, 1, 1] - pa[:, 1, 1]) * 100:+.1f} cm")
order = np.argsort(-total.mean(axis=0))[:8]
print("top joints:  " + "  ".join(f"{names[j]} {total[:, j].mean():.1f}({static[j]:.1f}/{dynamic[:, j].mean():.1f})"
                                  for j in order))
# is the dynamic difference ~constant over time, or does it change between sections?
seg = np.array_split(total.mean(axis=1), 6)
print("mean diff per sixth of clip: " + " ".join(f"{s.mean():.1f}" for s in seg))
