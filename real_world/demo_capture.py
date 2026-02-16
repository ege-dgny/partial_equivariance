import pyrealsense2 as rs
import numpy as np
import cv2
import os
import time
import json
import argparse
from typing import Optional


def fix_permissions(path):
    sudo_uid = os.environ.get('SUDO_UID')
    sudo_gid = os.environ.get('SUDO_GID')
    if sudo_uid is not None and sudo_gid is not None:
        os.chown(path, int(sudo_uid), int(sudo_gid))


def _default_data_rw_dir() -> str:
    """
    Default output root for real-world demos.

    Resolves to: Partial_Equivariance/data_rw (one level above this file).
    """
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data_rw"))


def _next_episode_idx(base_output_dir: str) -> int:
    """Pick the next non-existing episode index to avoid overwriting."""
    idx = 0
    while os.path.exists(os.path.join(base_output_dir, f"episode_{idx:03d}")):
        idx += 1
    return idx


def record_demos(base_output_dir: str, fps: int = 6, data_rw_dir: Optional[str] = None):
    """
    Record raw RGB-D demos and save them under:
      <data_rw_dir>/<task_name>/

    This directory will contain:
      - intrinsics.json
      - episode_000/{color_*.png, depth_*.png}
      - episode_001/...
    """
    if data_rw_dir is None:
        data_rw_dir = _default_data_rw_dir()

    # Backward-compatible API: the first argument is treated as the task name.
    task_name = base_output_dir
    output_dir = os.path.abspath(os.path.join(data_rw_dir, task_name))

    os.makedirs(output_dir, exist_ok=True)
    fix_permissions(output_dir)
    print(f"Saving real-world demos to: {output_dir}")

    # --- Hardware reset ---
    ctx = rs.context()
    for dev in ctx.devices:
        print(f"Device: {dev.get_info(rs.camera_info.name)}")
        # dev.hardware_reset()
    # print("Waiting for camera to reboot...")
    # time.sleep(5)

    # --- Configure streams ---
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, fps)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, fps)

    profile = pipeline.start(config)

    # --- Depth-color alignment ---
    align = rs.align(rs.stream.color)

    # --- Save intrinsics (once) ---
    color_stream = profile.get_stream(rs.stream.color)
    intr = color_stream.as_video_stream_profile().get_intrinsics()
    intrinsics_data = {
        "width": intr.width,
        "height": intr.height,
        "fx": intr.fx,
        "fy": intr.fy,
        "ppx": intr.ppx,
        "ppy": intr.ppy,
        "model": str(intr.model),
        "coeffs": intr.coeffs
    }
    intrinsics_path = os.path.join(output_dir, "intrinsics.json")
    with open(intrinsics_path, "w") as f:
        json.dump(intrinsics_data, f, indent=2)
    fix_permissions(intrinsics_path)
    print(f"Saved intrinsics to {intrinsics_path}")

    # --- Recording state ---
    episode_idx = _next_episode_idx(output_dir)
    frame_idx = 0
    recording = False

    print("\n=== PEFM Demo Recorder ===")
    print("SPACE  = start/stop episode")
    print("Q      = quit\n")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)

            depth_frame = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            # Raw data
            depth_image = np.asanyarray(depth_frame.get_data())   # uint16 mm
            color_image = np.asanyarray(color_frame.get_data())   # uint8 BGR

            # --- Display ---
            depth_vis = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03),
                cv2.COLORMAP_JET
            )
            display = color_image.copy()
            status = f"REC EP {episode_idx} | Frame {frame_idx}" if recording else "PAUSED"
            color = (0, 0, 255) if recording else (0, 200, 0)
            cv2.putText(display, status, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            if recording:
                cv2.rectangle(display, (0, 0), (639, 479), (0, 0, 255), 3)

            cv2.imshow("Color", display)
            cv2.imshow("Depth", depth_vis)

            # --- Save if recording ---
            if recording:
                ep_dir = os.path.join(output_dir, f"episode_{episode_idx:03d}")

                depth_path = os.path.join(ep_dir, f"depth_{frame_idx:05d}.png")
                color_path = os.path.join(ep_dir, f"color_{frame_idx:05d}.png")

                ok_depth = cv2.imwrite(depth_path, depth_image)   # RAW 16-bit
                ok_color = cv2.imwrite(color_path, color_image)
                if not (ok_depth and ok_color):
                    print(
                        f"[WARN] Failed to write frame {frame_idx} "
                        f"(depth_ok={ok_depth}, color_ok={ok_color}) to {ep_dir}"
                    )
                fix_permissions(depth_path)
                fix_permissions(color_path)

                frame_idx += 1

            # --- Keyboard ---
            key = cv2.waitKey(1) & 0xFF

            if key == ord(' '):
                if not recording:
                    ep_dir = os.path.join(
                        output_dir, f"episode_{episode_idx:03d}"
                    )
                    os.makedirs(ep_dir, exist_ok=True)
                    fix_permissions(ep_dir)
                    frame_idx = 0
                    recording = True
                    print(f">> Recording episode {episode_idx}...")
                else:
                    recording = False
                    print(f">> Episode {episode_idx} done: {frame_idx} frames")
                    episode_idx += 1

            elif key == ord('q'):
                if recording:
                    print(f">> Episode {episode_idx} done: {frame_idx} frames")
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        total = episode_idx + (1 if recording else 0)
        print(f"\nFinished. {total} episodes in {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record raw RealSense RGB-D demos.")
    parser.add_argument(
        "--task_name",
        type=str,
        default="book_shelf",
        help="Task name (demos saved under data_rw/<task_name>/).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=6,
        help="Capture FPS for both color and depth streams.",
    )
    parser.add_argument(
        "--data_rw_dir",
        type=str,
        default=None,
        help="Override output root (default: Partial_Equivariance/data_rw).",
    )
    args = parser.parse_args()

    record_demos(base_output_dir=args.task_name, fps=args.fps, data_rw_dir=args.data_rw_dir)
