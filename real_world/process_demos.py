"""
Process raw RGB-D demonstrations into PEFM NPZ format.

Converts human demonstration recordings (from demo_capture.py) into
the NPZ format expected by PEFM's dataset loader.

Usage:
    # With human parsing (Grounded SAM + HaMeR)
    python -m real_world.process_demos \
        --input_dir raw_demos \
        --output_dir processed \
        --object_prompt "cup. block." \
        --use_human_parsing

    # Simple mode (depth filtering only, no ML models)
    python -m real_world.process_demos \
        --input_dir raw_demos \
        --output_dir processed

    # With human parsing + save overlay videos
    python -m real_world.process_demos \
        --input_dir raw_demos \
        --output_dir processed \
        --use_human_parsing --save_vis

Output format:
    processed/
        pcs/
            episode_000_t0000.npz  # {pc, eef_pos, action}
            ...
        vis/                        # (when --save_vis)
            episode_000.mp4         # overlay of masks + hand keypoints
            ...
"""

import os
import glob
import argparse
import json
import numpy as np
import cv2
from tqdm import tqdm


# Hand keypoint skeleton for drawing (21 points: wrist + 5 fingers x 4)
HAND_SKELETON_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]


def write_episode_vis(
    episode_dir: str,
    frames: list,
    output_path: str,
    fps: int = 6,
) -> None:
    """Write a video overlaying parsing results (object masks + hand keypoints) on color frames."""
    color_files = sorted(glob.glob(os.path.join(episode_dir, "color_*.png")))
    if not color_files:
        return

    first_idx = frames[0]["frame_idx"]
    first_img = cv2.imread(color_files[first_idx])
    if first_img is None:
        return
    H, W = first_img.shape[:2]

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, float(fps), (W, H))

    mask_color = (0, 180, 255)
    kp_color = (0, 255, 0)
    skeleton_color = (0, 200, 255)

    for frame in frames:
        idx = frame["frame_idx"]
        if idx >= len(color_files):
            continue
        color = cv2.imread(color_files[idx])
        if color is None:
            continue
        overlay = color.copy()

        vis_masks = frame.get("vis_masks")
        if vis_masks:
            combined = np.zeros((H, W), dtype=np.uint8)
            for m in vis_masks.values():
                m = np.asarray(m)
                if m.ndim == 3:
                    m = m[0]
                if m.shape[:2] == (H, W):
                    combined = np.maximum(combined, (m.astype(np.uint8) * 255))
            if combined.max() > 0:
                mask_bgr = np.zeros_like(overlay)
                mask_bgr[combined > 0] = mask_color
                cv2.addWeighted(mask_bgr, 0.35, overlay, 1.0, 0, overlay)

        kp_2d = frame.get("vis_keypoints_2d")
        if kp_2d is not None and len(kp_2d) >= 21:
            pts = np.asarray(kp_2d[:21], dtype=np.int32)
            for i, j in HAND_SKELETON_EDGES:
                if 0 <= i < len(pts) and 0 <= j < len(pts):
                    pt1 = (int(pts[i, 0]), int(pts[i, 1]))
                    pt2 = (int(pts[j, 0]), int(pts[j, 1]))
                    cv2.line(overlay, pt1, pt2, skeleton_color, 2)
            for k in range(min(21, len(pts))):
                cx, cy = int(pts[k, 0]), int(pts[k, 1])
                if 0 <= cx < W and 0 <= cy < H:
                    cv2.circle(overlay, (cx, cy), 4, kp_color, -1)

        writer.write(overlay)

    writer.release()


def load_intrinsics(intrinsics_path: str) -> dict:
    """
    Load camera intrinsics from JSON file.

    Args:
        intrinsics_path: Path to intrinsics.json (saved by demo_capture.py)

    Returns:
        intrinsics: dict with fx, fy, ppx, ppy, width, height
    """
    with open(intrinsics_path, 'r') as f:
        intrinsics = json.load(f)
    return intrinsics


def process_with_human_parsing(
    episode_dir: str,
    intrinsics: dict,
    object_prompt: str,
    target_fps: int,
    num_points: int,
    device: str,
    save_vis: bool = False,
    use_mock_hand_detector: bool = False,
):
    """
    Process episode using full human parsing pipeline.

    Uses Grounded SAM for object detection and HaMeR for hand detection
    (or mock hand detector if use_mock_hand_detector=True or HaMeR unavailable).
    """
    from real_world.human_parsing import HumanDemoParser

    parser = HumanDemoParser(
        object_prompt=object_prompt,
        device=device,
        num_points=num_points,
        use_mock_hand_detector=use_mock_hand_detector,
    )

    frames = parser.parse_episode(
        episode_dir,
        intrinsics,
        target_fps=target_fps,
        save_vis=save_vis,
    )

    return frames


def process_simple(
    episode_dir: str,
    intrinsics: dict,
    target_fps: int,
    num_points: int,
    depth_min: float,
    depth_max: float,
):
    """
    Process episode using simple depth filtering.

    Does NOT extract hand poses - uses placeholder EEF states.
    Useful for testing or when ML models are not available.
    """
    from real_world.human_parsing.demo_parser import SimpleDemoParser

    parser = SimpleDemoParser(
        num_points=num_points,
        depth_min=depth_min,
        depth_max=depth_max,
    )

    frames = parser.parse_episode(
        episode_dir,
        intrinsics,
        target_fps=target_fps,
    )

    return frames


def save_frames_as_npz(
    frames: list,
    output_dir: str,
    episode_name: str,
    num_eef: int = 1,
):
    """
    Save parsed frames as NPZ files in PEFM format.

    Args:
        frames: list of dicts with 'pc', 'eef_pos', 'action'
        output_dir: output directory (pcs/ subdirectory)
        episode_name: name for this episode (e.g., 'episode_000')
        num_eef: number of end-effectors (1 for single arm)
    """
    os.makedirs(output_dir, exist_ok=True)

    for t, frame in enumerate(frames):
        # Format filename to match PEFM dataset expectations
        # Pattern: episode_name_tXXXX.npz
        npz_path = os.path.join(output_dir, f"{episode_name}_t{t:04d}.npz")

        # Ensure correct shapes for PEFM dataset loader
        pc = frame['pc']  # (num_points, 3)
        eef_pos = frame['eef_pos'].reshape(num_eef, -1)  # (num_eef, eef_dim)
        action = frame['action']  # (action_dim,) flattened

        np.savez(
            npz_path,
            pc=pc.astype(np.float32),
            eef_pos=eef_pos.astype(np.float32),
            action=action.astype(np.float32),
        )


def main():
    parser = argparse.ArgumentParser(
        description='Process raw RGB-D demos into PEFM NPZ format'
    )

    # Input/output
    parser.add_argument(
        '--input_dir', required=True,
        help='Directory containing raw demos (with intrinsics.json and episode_* subdirs)'
    )
    parser.add_argument(
        '--output_dir', required=True,
        help='Output directory (will create pcs/ subdirectory)'
    )

    # Processing mode
    parser.add_argument(
        '--use_human_parsing', action='store_true',
        help='Use full human parsing pipeline (Grounded SAM + HaMeR)'
    )
    parser.add_argument(
        '--object_prompt', default='object',
        help='Text prompt for object detection (e.g., "cup. block.")'
    )

    # Frame rate
    parser.add_argument(
        '--target_fps', type=int, default=3,
        help='Output frame rate (default: 3, EquiBot uses 3Hz)'
    )
    parser.add_argument(
        '--input_fps', type=int, default=6,
        help='Input recording frame rate (default: 6, demo_capture.py default)'
    )

    # Point cloud
    parser.add_argument(
        '--num_points', type=int, default=1024,
        help='Number of points per point cloud (default: 1024)'
    )

    # Simple mode options
    parser.add_argument(
        '--depth_min', type=float, default=0.2,
        help='Minimum depth in meters (simple mode, default: 0.2)'
    )
    parser.add_argument(
        '--depth_max', type=float, default=1.5,
        help='Maximum depth in meters (simple mode, default: 1.5)'
    )

    # Device
    parser.add_argument(
        '--device', default='cuda',
        help='Device for ML models (default: cuda)'
    )

    # Filtering
    parser.add_argument(
        '--min_frames', type=int, default=5,
        help='Minimum frames per episode to include (default: 5)'
    )
    parser.add_argument(
        '--save_vis', action='store_true',
        help='Save overlay videos (masks + hand keypoints) to output_dir/vis/ (only with --use_human_parsing)'
    )
    parser.add_argument(
        '--mock_hands', action='store_true',
        help='Use mock hand detector (e.g. when HaMeR is not installed on M1); still runs object tracking'
    )

    args = parser.parse_args()

    # Load intrinsics
    intrinsics_path = os.path.join(args.input_dir, 'intrinsics.json')
    if not os.path.exists(intrinsics_path):
        raise FileNotFoundError(
            f"intrinsics.json not found in {args.input_dir}. "
            "Run demo_capture.py first to record demos with camera intrinsics."
        )

    intrinsics = load_intrinsics(intrinsics_path)
    print(f"Loaded intrinsics: fx={intrinsics['fx']:.1f}, fy={intrinsics['fy']:.1f}")

    # Find episode directories
    episode_dirs = sorted(glob.glob(os.path.join(args.input_dir, 'episode_*')))
    if len(episode_dirs) == 0:
        raise FileNotFoundError(
            f"No episode_* directories found in {args.input_dir}"
        )

    print(f"Found {len(episode_dirs)} episodes")

    # Setup output directory
    output_pcs = os.path.join(args.output_dir, 'pcs')
    os.makedirs(output_pcs, exist_ok=True)
    output_vis = os.path.join(args.output_dir, 'vis')
    if args.save_vis and args.use_human_parsing:
        os.makedirs(output_vis, exist_ok=True)

    # Copy intrinsics to output
    output_intrinsics = os.path.join(args.output_dir, 'intrinsics.json')
    with open(output_intrinsics, 'w') as f:
        json.dump(intrinsics, f, indent=2)

    # Process each episode
    processed_count = 0
    skipped_count = 0
    total_frames = 0

    for ep_dir in tqdm(episode_dirs, desc='Processing episodes'):
        ep_name = os.path.basename(ep_dir)

        try:
            if args.use_human_parsing:
                frames = process_with_human_parsing(
                    ep_dir,
                    intrinsics,
                    args.object_prompt,
                    args.target_fps,
                    args.num_points,
                    args.device,
                    save_vis=args.save_vis,
                    use_mock_hand_detector=args.mock_hands,
                )
            else:
                frames = process_simple(
                    ep_dir,
                    intrinsics,
                    args.target_fps,
                    args.num_points,
                    args.depth_min,
                    args.depth_max,
                )

            if len(frames) < args.min_frames:
                print(f"  Skipping {ep_name}: only {len(frames)} frames")
                skipped_count += 1
                continue

            # Save as NPZ
            save_frames_as_npz(frames, output_pcs, ep_name)
            if args.save_vis and args.use_human_parsing and frames and 'vis_masks' in frames[0]:
                write_episode_vis(
                    ep_dir,
                    frames,
                    os.path.join(output_vis, f"{ep_name}.mp4"),
                    args.input_fps,
                )
            processed_count += 1
            total_frames += len(frames)

        except Exception as e:
            print(f"  Error processing {ep_name}: {e}")
            skipped_count += 1

    # Summary
    print(f"\nProcessing complete:")
    print(f"  Processed: {processed_count} episodes")
    print(f"  Skipped: {skipped_count} episodes")
    print(f"  Total frames: {total_frames}")
    print(f"  Output: {output_pcs}")
    if args.save_vis and args.use_human_parsing and processed_count > 0:
        print(f"  Vis:   {output_vis}")

    # Save metadata
    metadata = {
        'num_episodes': processed_count,
        'total_frames': total_frames,
        'target_fps': args.target_fps,
        'num_points': args.num_points,
        'use_human_parsing': args.use_human_parsing,
        'object_prompt': args.object_prompt if args.use_human_parsing else None,
    }
    metadata_path = os.path.join(args.output_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"  Metadata saved to: {metadata_path}")


if __name__ == '__main__':
    main()
