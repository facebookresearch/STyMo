"""Convert a GVHMR/SMPL-22 BVH into the layout STyMo expects.

STyMo needs a ground-level `Root` joint (y = 0, yaw-only rotation) above the
pelvis, a pelvis with its own position channels, and metres. SMPL BVHs from
GVHMR have the pelvis as root, in centimetres. This script:
  - adds `Root` at the pelvis x/z on the floor, heading from the hip line
    (smoothed), rotation about Y only;
  - re-expresses the pelvis position and rotation relative to `Root`;
  - shifts the floor so the lowest foot point over the clip is at y = 0;
  - converts cm -> m (--scale, default 0.01).
Joint rotations below the pelvis are copied unchanged.

Usage:
  python headless/smpl_to_stymo.py in.bvh out.bvh [--scale 0.01] [--smooth 4]
  python headless/smpl_to_stymo.py in.bvh out.bvh --check   # FK round-trip check
"""
import argparse
import re

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as R

AXES = {"Xrotation": "X", "Yrotation": "Y", "Zrotation": "Z"}


def parse_bvh(path):
    text = open(path).read()
    head, motion = text.split("MOTION")
    joints = []  # dicts: name, parent, offset, channels, end_site
    stack = []
    lines = [l.strip() for l in head.splitlines() if l.strip()]
    i = 0
    pending_end = False
    while i < len(lines):
        l = lines[i]
        if l.startswith(("ROOT", "JOINT")):
            joints.append(dict(name=l.split()[1], parent=stack[-1] if stack else -1,
                               offset=None, channels=[], end_site=None))
        elif l.startswith("End Site"):
            pending_end = True
        elif l == "{":
            stack.append(len(joints) - 1) if not pending_end else stack.append("end")
        elif l == "}":
            top = stack.pop()
            if top == "end":
                pending_end = False
        elif l.startswith("OFFSET"):
            off = np.array([float(v) for v in l.split()[1:4]])
            if pending_end:
                joints[stack[-2]]["end_site"] = off
            else:
                joints[-1]["offset"] = off
        elif l.startswith("CHANNELS"):
            joints[-1]["channels"] = l.split()[2:]
        i += 1
    mlines = motion.strip().splitlines()
    frame_time = float(mlines[1].split(":")[1])
    data = np.array([[float(v) for v in l.split()] for l in mlines[2:] if l.strip()])
    return joints, frame_time, data


def unpack(joints, data):
    """Return local quats (F,J,4 xyzw scipy), root position (F,3)."""
    F, J = data.shape[0], len(joints)
    rots = np.zeros((F, J, 4))
    root_pos = None
    c = 0
    for j, jt in enumerate(joints):
        ch = jt["channels"]
        vals = data[:, c:c + len(ch)]
        c += len(ch)
        pos_idx = [k for k, n in enumerate(ch) if n.endswith("position")]
        rot_idx = [k for k, n in enumerate(ch) if n.endswith("rotation")]
        if pos_idx:
            p = np.zeros((F, 3))
            for k in pos_idx:
                p[:, "XYZ".index(ch[k][0])] = vals[:, k]
            if j == 0:
                root_pos = p
        order = "".join(AXES[ch[k]] for k in rot_idx)
        rots[:, j] = R.from_euler(order, vals[:, rot_idx], degrees=True).as_quat()
    return rots, root_pos


def fk(joints, rots, root_pos, offsets=None):
    F, J = rots.shape[:2]
    offsets = offsets if offsets is not None else np.stack([jt["offset"] for jt in joints])
    gpos = np.zeros((F, J, 3))
    grot = [None] * J
    for j, jt in enumerate(joints):
        local = R.from_quat(rots[:, j])
        if jt["parent"] == -1:
            gpos[:, j] = root_pos
            grot[j] = local
        else:
            p = jt["parent"]
            off = offsets[j] if offsets.ndim == 2 else offsets[:, j]
            gpos[:, j] = gpos[:, p] + grot[p].apply(off)
            grot[j] = grot[p] * local
    return gpos, grot


def write_bvh(path, joints, frame_time, data):
    out = ["HIERARCHY"]

    def rec(j, depth):
        t = "\t" * depth
        jt = joints[j]
        out.append(f"{t}{'ROOT' if jt['parent'] == -1 else 'JOINT'} {jt['name']}")
        out.append(f"{t}{{")
        out.append(f"{t}\tOFFSET " + " ".join(f"{v:.6f}" for v in jt["offset"]))
        out.append(f"{t}\tCHANNELS {len(jt['channels'])} " + " ".join(jt["channels"]))
        kids = [k for k, c in enumerate(joints) if c["parent"] == j]
        for k in kids:
            rec(k, depth + 1)
        if not kids:
            es = jt["end_site"] if jt["end_site"] is not None else np.zeros(3)
            out.extend([f"{t}\tEnd Site", f"{t}\t{{",
                    f"{t}\t\tOFFSET " + " ".join(f"{v:.6f}" for v in es), f"{t}\t}}"])
        out.append(f"{t}}}")

    rec(0, 0)
    out += ["MOTION", f"Frames: {data.shape[0]}", f"Frame Time: {frame_time:.8f}"]
    out += [" ".join(f"{v:.6f}" for v in row) for row in data]
    open(path, "w").write("\n".join(out) + "\n")


def convert(src, dst, scale=0.01, smooth=4.0):
    joints, ft, data = parse_bvh(src)
    assert joints[0]["name"] == "pelvis", "expected an SMPL BVH with pelvis as root"
    rots, root_pos = unpack(joints, data)
    gpos, grot = fk(joints, rots, root_pos)
    gpos *= scale
    names = [j["name"] for j in joints]

    # floor: lowest point of ankles/feet (and their end sites) over the clip
    foot_ids = [names.index(n) for n in ("left_ankle", "right_ankle", "left_foot", "right_foot")]
    ys = [gpos[:, k, 1] for k in foot_ids]
    for n in ("left_foot", "right_foot"):
        k = names.index(n)
        es = joints[k]["end_site"]
        if es is not None:
            ys.append(gpos[:, k, 1] + grot[k].apply(es * scale)[:, 1])
    floor = np.min(np.stack(ys))
    gpos[..., 1] -= floor

    # heading from hip line (left - right) x up -> forward, yaw about Y
    across = gpos[:, names.index("left_hip")] - gpos[:, names.index("right_hip")]
    fwd = np.cross(across, np.array([0.0, 1.0, 0.0]))
    yaw = np.unwrap(np.arctan2(fwd[:, 0], fwd[:, 2]))
    if smooth > 0:
        yaw = gaussian_filter1d(yaw, smooth, mode="nearest")
    root_rot = R.from_euler("Y", yaw[:, None])
    pelvis = gpos[:, 0]
    root_xyz = pelvis.copy()
    root_xyz[:, 1] = 0.0

    pel_local_pos = root_rot.inv().apply(pelvis - root_xyz)
    pel_local_rot = root_rot.inv() * grot[0]

    order_ch = ["Zrotation", "Xrotation", "Yrotation"]
    new_joints = [dict(name="Root", parent=-1, offset=np.zeros(3),
                       channels=["Xposition", "Yposition", "Zposition"] + order_ch, end_site=None)]
    for jt in joints:
        nj = dict(jt)
        nj["parent"] = jt["parent"] + 1
        nj["offset"] = jt["offset"] * scale
        nj["end_site"] = None if jt["end_site"] is None else jt["end_site"] * scale
        new_joints.append(nj)
    new_joints[1]["offset"] = np.array([0.0, pel_local_pos[:, 1].mean(), 0.0])
    new_joints[1]["channels"] = ["Xposition", "Yposition", "Zposition"] + order_ch

    cols = [root_xyz, root_rot.as_euler("ZXY", degrees=True),
            pel_local_pos, pel_local_rot.as_euler("ZXY", degrees=True)]
    # remaining joints: copy their original rotation columns
    c = len(joints[0]["channels"])
    cols.append(data[:, c:])
    for jt in joints[1:]:
        assert jt["channels"] == order_ch, f"unexpected channel order on {jt['name']}"
    write_bvh(dst, new_joints, ft, np.concatenate(cols, axis=1))
    return gpos


def check(src, dst, scale):
    gpos_ref = convert(src, dst, scale)  # floor-shifted, scaled reference
    joints, ft, data = parse_bvh(dst)
    rots, root_pos = unpack(joints, data)
    offs = np.tile(np.stack([j["offset"] for j in joints]), (data.shape[0], 1, 1))
    pel = 3 + 3  # pelvis position columns after Root's 6 channels
    offs[:, 1] = data[:, pel:pel + 3]
    gpos, _ = fk(joints, rots, root_pos, offs)
    err = np.abs(gpos[:, 1:] - gpos_ref).max()
    print(f"max FK position error: {err:.2e} m; root y range "
          f"{root_pos[:,1].min():.3f}..{root_pos[:,1].max():.3f}; "
          f"pelvis height mean {offs[:,1,1].mean():.3f} m")
    return err


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--scale", type=float, default=0.01)
    ap.add_argument("--smooth", type=float, default=4.0, help="gaussian sigma (frames) for heading")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.check:
        assert check(a.src, a.dst, a.scale) < 1e-4
    else:
        convert(a.src, a.dst, a.scale, a.smooth)
        print("wrote", a.dst)
