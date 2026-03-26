"""Create hurdle jump constraints from the current generated motion."""
import os
os.environ["TEXT_ENCODER_MODE"] = "dummy"

import json
import numpy as np
import torch
from kimodo import load_model
from scipy.spatial.transform import Rotation

model = load_model("Kimodo-SOMA-RP-v1", device="cuda:0")
skel = model.skeleton

# Generate a base motion to get realistic poses
torch.manual_seed(55)
output = model(
    prompts=[""],
    num_frames=[150],
    num_denoising_steps=100,
    num_samples=1,
    cfg_type="nocfg",
    post_processing=False,
    return_numpy=True,
)

local_rots = output["local_rot_mats"][0]   # [150, 77, 3, 3]
root_pos = output["root_positions"][0]      # [150, 3]

# We'll create 3 keyframes for a hurdle jump:
# Frame 30: running stance (use a natural standing pose from frame 10)
# Frame 75: mid-air hurdle pose (modify a pose to lift legs)
# Frame 120: landing (back to standing)

def rot_mat_to_axis_angle(R):
    r = Rotation.from_matrix(R)
    return r.as_rotvec().tolist()

def make_keyframe(frame_idx, local_rots_frame, root_position, skel):
    """Create a full-body keyframe constraint dict from motion data."""
    joints_aa = []
    for j in range(77):
        aa = rot_mat_to_axis_angle(local_rots_frame[j])
        joints_aa.append(aa)

    return {
        "local_joints_rot": joints_aa,
        "root_positions": root_position if isinstance(root_position, list) else root_position.tolist(),
    }

# --- Keyframe 1: Frame 30 - running forward (use frame 10 pose as-is) ---
kf1 = make_keyframe(30, local_rots[10], [0.0, float(root_pos[10, 1]), 1.0], skel)

# --- Keyframe 2: Frame 75 - mid-air hurdle jump ---
# Start from frame 40 pose and modify to raise legs
hurdle_rots = local_rots[40].copy()

# Rotate both hip joints forward (lift thighs up)
# SOMA skeleton: find hip joint indices
s77 = skel.somaskel77 if hasattr(skel, 'somaskel77') else skel
print("Hip joints:", s77.hip_joint_names if hasattr(s77, 'hip_joint_names') else "N/A")
print("Parent indices (first 10):", s77.parents[:10].tolist() if hasattr(s77, 'parents') else "N/A")

# For SOMA77 skeleton, typical hip joints are around index 1 (L_Hip) and 5 (R_Hip)
# We'll rotate them to lift legs up
hip_rotation = Rotation.from_euler('x', -70, degrees=True).as_matrix()  # lift thigh forward
knee_rotation = Rotation.from_euler('x', 60, degrees=True).as_matrix()  # bend knee

# Left leg
hurdle_rots[1] = hip_rotation @ hurdle_rots[1]    # L_Hip - lift forward
hurdle_rots[2] = knee_rotation @ hurdle_rots[2]    # L_Knee - bend

# Right leg
hurdle_rots[5] = hip_rotation @ hurdle_rots[5]     # R_Hip - lift forward
hurdle_rots[6] = knee_rotation @ hurdle_rots[6]    # R_Knee - bend

kf2 = make_keyframe(75, hurdle_rots, [0.0, float(root_pos[10, 1]) + 0.4, 2.5], skel)

# --- Keyframe 3: Frame 120 - landing ---
kf3 = make_keyframe(120, local_rots[10], [0.0, float(root_pos[10, 1]), 4.0], skel)

# Build constraint list
constraints = [
    {
        "type": "fullbody",
        "frame_indices": [30, 75, 120],
        "local_joints_rot": [kf1["local_joints_rot"], kf2["local_joints_rot"], kf3["local_joints_rot"]],
        "root_positions": [kf1["root_positions"], kf2["root_positions"], kf3["root_positions"]],
    }
]

out_path = "demo_outputs/hurdle_constraints.json"
with open(out_path, "w") as f:
    json.dump(constraints, f)

print(f"Saved hurdle constraints to {out_path}")
print(f"Keyframes at frames: 30 (run), 75 (jump), 120 (land)")
