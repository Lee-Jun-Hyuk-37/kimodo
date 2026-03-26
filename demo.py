"""
Kimodo Demo Script
==================
Text-driven and constraint-based 3D human motion generation by NVIDIA.

Prerequisites:
  conda activate kimodo
  pip install -e ".[all]"

Usage:
  # Constraint-only mode (no Llama access needed):
  set TEXT_ENCODER_MODE=dummy
  python demo.py                    # run all demos
  python demo.py --demo 1           # run specific demo

  # Full mode (after Llama access is approved):
  set KIMODO_QUANTIZE=4bit
  python demo.py                    # text prompts will work

Available models:
  - Kimodo-SOMA-RP-v1   : human 77-joint skeleton (default)
  - Kimodo-G1-RP-v1     : Unitree G1 robot
  - Kimodo-SMPLX-RP-v1  : SMPL-X 22-joint body
"""

import argparse
import json
import os

import numpy as np
import torch

from kimodo import load_model
from kimodo.constraints import load_constraints_lst

OUTPUT_DIR = "demo_outputs"
EXAMPLES_ROOT = os.path.join(os.path.dirname(__file__), "kimodo", "assets", "demo", "examples", "kimodo-soma-rp")
IS_DUMMY = os.environ.get("TEXT_ENCODER_MODE", "").lower() == "dummy"


def setup(model_name="Kimodo-SOMA-RP-v1", device=None):
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[Setup] Device: {device}, Model: {model_name}")
    if IS_DUMMY:
        print("[Setup] TEXT_ENCODER_MODE=dummy -> text prompts will be ignored")
    model = load_model(model_name, device=device)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return model, device


def save_output(output, name, export_bvh=False, model=None, device="cuda:0"):
    path = os.path.join(OUTPUT_DIR, f"{name}.npz")
    single = {
        k: (v[0] if hasattr(v, "shape") and len(v.shape) > 0 and v.shape[0] == 1 else v)
        for k, v in output.items()
    }
    np.savez(path, **single)
    n_frames = single["posed_joints"].shape[0]
    n_joints = single["posed_joints"].shape[1]
    print(f"  -> Saved: {path}  ({n_frames} frames, {n_joints} joints)")

    if export_bvh and model is not None:
        from kimodo.exports.bvh import save_motion_bvh
        from kimodo.skeleton import global_rots_to_local_rots, SOMASkeleton30

        skeleton = model.skeleton
        if isinstance(skeleton, SOMASkeleton30):
            skeleton = skeleton.somaskel77.to(device)

        bvh_path = os.path.join(OUTPUT_DIR, f"{name}.bvh")
        joints_pos = torch.from_numpy(single["posed_joints"]).to(device)
        joints_rot = torch.from_numpy(single["global_rot_mats"]).to(device)
        local_rots = global_rots_to_local_rots(joints_rot, skeleton)
        root_pos = joints_pos[:, skeleton.root_idx, :]
        save_motion_bvh(bvh_path, local_rots, root_pos, skeleton=skeleton, fps=model.fps)
        print(f"  -> BVH:   {bvh_path}")

    return path


def load_example(example_name, model):
    """Load constraints from a pre-built example folder."""
    example_dir = os.path.join(EXAMPLES_ROOT, example_name)
    constraints_path = os.path.join(example_dir, "constraints.json")
    meta_path = os.path.join(example_dir, "meta.json")

    constraint_lst = []
    if os.path.exists(constraints_path):
        constraint_lst = load_constraints_lst(constraints_path, model.skeleton)

    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)

    return constraint_lst, meta


# ============================================================
# Demo 1: Unconditional motion generation (no text, no constraints)
# ============================================================
def demo_1_unconditional(model, device):
    print("\n" + "=" * 60)
    print("Demo 1: Unconditional Motion Generation")
    print("=" * 60)
    print("  No text, no constraints -> model generates a random motion")

    output = model(
        prompts=[""],
        num_frames=[150],
        num_denoising_steps=100,
        num_samples=1,
        cfg_type="nocfg",
        post_processing=False,
        return_numpy=True,
    )
    save_output(output, "01_unconditional", export_bvh=True, model=model, device=device)


# ============================================================
# Demo 2: Multiple random samples (variation)
# ============================================================
def demo_2_variations(model, device):
    print("\n" + "=" * 60)
    print("Demo 2: Multiple Variations")
    print("=" * 60)
    print("  Same settings, different random seeds -> diverse motions")

    for i, seed in enumerate([42, 123, 777]):
        torch.manual_seed(seed)
        output = model(
            prompts=[""],
            num_frames=[150],
            num_denoising_steps=100,
            num_samples=1,
            cfg_type="nocfg",
            post_processing=False,
            return_numpy=True,
        )
        save_output(output, f"02_variation_seed{seed}", export_bvh=True, model=model, device=device)


# ============================================================
# Demo 3: Root 2D waypoints (sparse path control)
# ============================================================
def demo_3_root_waypoints(model, device):
    print("\n" + "=" * 60)
    print("Demo 3: Root 2D Waypoints")
    print("=" * 60)
    print("  Control WHERE the character moves using sparse waypoints")

    constraint_lst, meta = load_example("06_root_waypoints", model)
    num_frames = int(meta.get("duration", 6.0) * model.fps)

    print(f"  Waypoints: {len(constraint_lst)} constraint sets")
    print(f"  Duration: {meta.get('duration', 6.0)}s ({num_frames} frames)")

    output = model(
        prompts=[""],
        num_frames=[num_frames],
        num_denoising_steps=100,
        num_samples=1,
        constraint_lst=constraint_lst,
        cfg_type="nocfg",
        post_processing=False,
        return_numpy=True,
    )
    save_output(output, "03_root_waypoints", export_bvh=True, model=model, device=device)


# ============================================================
# Demo 4: Root 2D dense path (continuous path control)
# ============================================================
def demo_4_root_path(model, device):
    print("\n" + "=" * 60)
    print("Demo 4: Root 2D Dense Path")
    print("=" * 60)
    print("  Control the exact trajectory the character follows")

    constraint_lst, meta = load_example("05_root_path", model)
    num_frames = int(meta.get("duration", 10.0) * model.fps)

    print(f"  Dense path constraint: every frame specified")
    print(f"  Duration: {meta.get('duration', 10.0)}s ({num_frames} frames)")

    output = model(
        prompts=[""],
        num_frames=[num_frames],
        num_denoising_steps=100,
        num_samples=1,
        constraint_lst=constraint_lst,
        cfg_type="nocfg",
        post_processing=False,
        return_numpy=True,
    )
    save_output(output, "04_root_path", export_bvh=True, model=model, device=device)


# ============================================================
# Demo 5: Full body keyframes
# ============================================================
def demo_5_fullbody_keyframes(model, device):
    print("\n" + "=" * 60)
    print("Demo 5: Full Body Keyframes")
    print("=" * 60)
    print("  Pin exact poses at specific frames, model interpolates between them")

    constraint_lst, meta = load_example("03_full_body_keyframes", model)
    num_frames = int(meta.get("duration", 5.0) * model.fps)

    for c in constraint_lst:
        print(f"  Constraint type: {c}")

    output = model(
        prompts=[""],
        num_frames=[num_frames],
        num_denoising_steps=100,
        num_samples=1,
        constraint_lst=constraint_lst,
        cfg_type="nocfg",
        post_processing=False,
        return_numpy=True,
    )
    save_output(output, "05_fullbody_keyframes", export_bvh=True, model=model, device=device)


# ============================================================
# Demo 6: End-effector constraints
# ============================================================
def demo_6_end_effector(model, device):
    print("\n" + "=" * 60)
    print("Demo 6: End-Effector Constraints")
    print("=" * 60)
    print("  Fix hand/foot positions at specific frames")

    constraint_lst, meta = load_example("04_ee_constraint", model)
    num_frames = int(meta.get("duration", 5.0) * model.fps)

    for c in constraint_lst:
        print(f"  Constraint type: {c}")

    output = model(
        prompts=[""],
        num_frames=[num_frames],
        num_denoising_steps=100,
        num_samples=1,
        constraint_lst=constraint_lst,
        cfg_type="nocfg",
        post_processing=False,
        return_numpy=True,
    )
    save_output(output, "06_end_effector", export_bvh=True, model=model, device=device)


# ============================================================
# Demo 7: Mixed constraints
# ============================================================
def demo_7_mixed_constraints(model, device):
    print("\n" + "=" * 60)
    print("Demo 7: Mixed Constraints")
    print("=" * 60)
    print("  Combine multiple constraint types in one generation")

    constraint_lst, meta = load_example("07_mixed_constraints", model)
    num_frames = int(meta.get("duration", 5.0) * model.fps)

    for c in constraint_lst:
        print(f"  Constraint type: {c}")

    output = model(
        prompts=[""],
        num_frames=[num_frames],
        num_denoising_steps=100,
        num_samples=1,
        constraint_lst=constraint_lst,
        cfg_type="nocfg",
        post_processing=False,
        return_numpy=True,
    )
    save_output(output, "07_mixed_constraints", export_bvh=True, model=model, device=device)


# ============================================================
# Demo 8: Programmatic waypoints (custom path)
# ============================================================
def demo_8_custom_path(model, device):
    print("\n" + "=" * 60)
    print("Demo 8: Custom Programmatic Path (Circle)")
    print("=" * 60)
    print("  Create constraints from code: walk in a circle")

    num_frames = 300  # 10 seconds
    n_waypoints = 8
    radius = 1.5

    frame_indices = [int(i * (num_frames - 1) / (n_waypoints - 1)) for i in range(n_waypoints)]
    angles = [2 * np.pi * i / (n_waypoints - 1) for i in range(n_waypoints)]
    waypoints = [[float(radius * np.sin(a)), float(radius * np.cos(a) - radius)] for a in angles]

    print(f"  {n_waypoints} waypoints in a circle (radius={radius}m)")
    for i, (fi, wp) in enumerate(zip(frame_indices, waypoints)):
        print(f"    Frame {fi:3d}: ({wp[0]:+.2f}, {wp[1]:+.2f})")

    constraint_data = [{
        "type": "root2d",
        "frame_indices": frame_indices,
        "smooth_root_2d": waypoints,
    }]
    constraint_lst = load_constraints_lst(constraint_data, model.skeleton)

    output = model(
        prompts=[""],
        num_frames=[num_frames],
        num_denoising_steps=100,
        num_samples=1,
        constraint_lst=constraint_lst,
        cfg_type="nocfg",
        post_processing=False,
        return_numpy=True,
    )
    save_output(output, "08_circle_path", export_bvh=True, model=model, device=device)


# ============================================================
# Demo 9: Programmatic waypoints (zigzag)
# ============================================================
def demo_9_zigzag(model, device):
    print("\n" + "=" * 60)
    print("Demo 9: Custom Programmatic Path (Zigzag)")
    print("=" * 60)
    print("  Walk in a zigzag pattern")

    num_frames = 240  # 8 seconds
    waypoints = [
        [0.0, 0.0],
        [1.0, 1.0],
        [-1.0, 2.0],
        [1.0, 3.0],
        [-1.0, 4.0],
        [0.0, 5.0],
    ]
    n = len(waypoints)
    frame_indices = [int(i * (num_frames - 1) / (n - 1)) for i in range(n)]

    print(f"  {n} waypoints in zigzag")

    constraint_data = [{
        "type": "root2d",
        "frame_indices": frame_indices,
        "smooth_root_2d": waypoints,
    }]
    constraint_lst = load_constraints_lst(constraint_data, model.skeleton)

    output = model(
        prompts=[""],
        num_frames=[num_frames],
        num_denoising_steps=100,
        num_samples=1,
        constraint_lst=constraint_lst,
        cfg_type="nocfg",
        post_processing=False,
        return_numpy=True,
    )
    save_output(output, "09_zigzag_path", export_bvh=True, model=model, device=device)


# ============================================================
# Main
# ============================================================
DEMOS = {
    1: ("Unconditional Generation", demo_1_unconditional),
    2: ("Multiple Variations (seeds)", demo_2_variations),
    3: ("Root 2D Waypoints", demo_3_root_waypoints),
    4: ("Root 2D Dense Path", demo_4_root_path),
    5: ("Full Body Keyframes", demo_5_fullbody_keyframes),
    6: ("End-Effector Constraints", demo_6_end_effector),
    7: ("Mixed Constraints", demo_7_mixed_constraints),
    8: ("Custom Circle Path", demo_8_custom_path),
    9: ("Custom Zigzag Path", demo_9_zigzag),
}


def main():
    parser = argparse.ArgumentParser(description="Kimodo Demo Runner")
    parser.add_argument("--demo", type=int, default=None, help="Run specific demo (1-9), or all if omitted")
    parser.add_argument("--model", type=str, default="Kimodo-SOMA-RP-v1", help="Model name")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda:0, cpu)")
    args = parser.parse_args()

    print("=" * 60)
    print("  Kimodo Motion Generation Demo")
    print()
    if IS_DUMMY:
        print("  MODE: constraint-only (TEXT_ENCODER_MODE=dummy)")
        print("  Text prompts are ignored. Use constraints to control motion.")
    else:
        print("  MODE: full (text + constraints)")
    print()
    print("  Available demos:")
    for k, (name, _) in DEMOS.items():
        print(f"    {k}. {name}")
    print("=" * 60)

    model, device = setup(args.model, args.device)

    if args.demo is not None:
        if args.demo not in DEMOS:
            print(f"Invalid demo number. Choose 1-{len(DEMOS)}")
            return
        name, fn = DEMOS[args.demo]
        fn(model, device)
    else:
        for k, (name, fn) in DEMOS.items():
            fn(model, device)

    print("\n" + "=" * 60)
    print(f"All outputs saved to ./{OUTPUT_DIR}/")
    print()
    print("Output NPZ keys:")
    print("  posed_joints    [T, 77, 3]   - joint world positions")
    print("  global_rot_mats [T, 77, 3, 3] - global rotation matrices")
    print("  local_rot_mats  [T, 77, 3, 3] - local rotation matrices")
    print("  foot_contacts   [T, 4]        - foot contact labels")
    print("  root_positions  [T, 3]        - root trajectory")
    print()
    print("BVH files can be viewed in Blender or any BVH viewer.")
    if IS_DUMMY:
        print()
        print("To enable text prompts later:")
        print("  1. Get Llama-3 access: https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct")
        print("  2. Run without TEXT_ENCODER_MODE=dummy, with KIMODO_QUANTIZE=4bit instead")
    print("=" * 60)


if __name__ == "__main__":
    main()
