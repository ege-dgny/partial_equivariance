"""
Lightweight environment visualizer for PEFM envs.

Usage (run from Partial_Equivariance/ after `pip install -e pefm_envs/`):

  # Franka (PyBullet GUI)
  python -m pefm_envs.visualize_env --suite franka --task peg_insert --vis

  # Franka (no GUI, render frames with OpenCV)
  python -m pefm_envs.visualize_env --suite franka --task pick_place --render

  # Mobile (always DIRECT; render frames with OpenCV)
  python -m pefm_envs.visualize_env --suite mobile --task pour --render

This script only resets the environment and visualizes the setup; it does not
run demo generation or save dataset files.
"""

from __future__ import annotations

import argparse
import time
from types import SimpleNamespace
from typing import Any, Dict, Type

import numpy as np


def _get_franka_env_class(task: str):
    from pefm_envs.sim_franka.centering_env import CenteringEnv
    from pefm_envs.sim_franka.orient_place_env import OrientPlaceEnv
    from pefm_envs.sim_franka.peg_insert_env import PegInsertEnv
    from pefm_envs.sim_franka.pick_place_env import PickPlaceEnv
    from pefm_envs.sim_franka.stack_env import StackEnv

    table: Dict[str, Type] = {
        "pick_place": PickPlaceEnv,
        "peg_insert": PegInsertEnv,
        "centering": CenteringEnv,
        "orient_place": OrientPlaceEnv,
        "stack": StackEnv,
    }
    if task not in table:
        raise ValueError(f"Unknown franka task '{task}'. Choices: {sorted(table)}")
    return table[task]


def _get_mobile_env_class(task: str):
    from pefm_envs.sim_mobile.compass_closing_env import CompassClosingEnv
    from pefm_envs.sim_mobile.insertion_env import InsertionEnv
    from pefm_envs.sim_mobile.pouring_env import PouringEnv

    table: Dict[str, Type] = {
        "pour": PouringEnv,
        "insert": InsertionEnv,
        "compass_close": CompassClosingEnv,
    }
    if task not in table:
        raise ValueError(f"Unknown mobile task '{task}'. Choices: {sorted(table)}")
    return table[task]


def _make_args_franka(cli: argparse.Namespace) -> SimpleNamespace:
    # FrankaEnv reads these fields in __init__.
    return SimpleNamespace(
        dof=cli.dof,
        num_eef=1,
        max_episode_length=cli.max_episode_length,
        seed=cli.seed,
        freq=cli.freq,
        ac_noise=cli.ac_noise,
        randomize_rotation=cli.randomize_rotation,
        randomize_scale=cli.randomize_scale,
        uniform_scaling=cli.uniform_scaling,
        scale_low=cli.scale_low,
        scale_high=cli.scale_high,
        scale_aspect_limit=cli.scale_aspect_limit,
        vis=cli.vis,
        demo_mode=False,
    )


def _make_args_mobile(cli: argparse.Namespace) -> SimpleNamespace:
    # BaseEnv expects these fields (even if many are unused for a given task).
    return SimpleNamespace(
        num_eef=2,
        dof=cli.dof,
        max_episode_length=cli.max_episode_length,
        seed=cli.seed,
        freq=cli.freq,
        ac_noise=cli.ac_noise,
        randomize_rotation=cli.randomize_rotation,
        randomize_scale=cli.randomize_scale,
        uniform_scaling=cli.uniform_scaling,
        scale_low=cli.scale_low,
        scale_high=cli.scale_high,
        scale_aspect_limit=cli.scale_aspect_limit,
        randomize_position=False,
        rand_pos_scale=0.0,
        vis=False,  # sim_mobile env uses DIRECT in current code
        cam_resolution=cli.cam_resolution,
    )


def _maybe_print_setup_info(env: Any):
    ang = None
    if hasattr(env, "_object_rotation"):
        try:
            ang = float(np.array(env._object_rotation)[-1])
        except Exception:
            ang = None
    if ang is not None:
        print(f"[visualize_env] object yaw (rad): {ang:.3f} | deg: {np.degrees(ang):.1f}")
    if hasattr(env, "rigid_ids"):
        print(f"[visualize_env] rigid objects: {len(getattr(env, 'rigid_ids', []))}")


def _render_loop(env: Any, fps: float, resolution: int):
    try:
        import cv2  # optional dependency
    except Exception as e:
        raise RuntimeError(
            "OpenCV (cv2) is required for --render. "
            "Install it (e.g. `pip install opencv-python`) or run with --vis (Franka only)."
        ) from e

    delay_ms = max(1, int(1000 / max(fps, 1e-6)))
    win = "pefm_envs visualize"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    print("[visualize_env] Rendering. Press 'q' to quit.")
    while True:
        # Render RGB only; point cloud extraction is much heavier.
        if hasattr(env, "render_dual"):
            frame = env.render_dual(resolution=resolution)
        else:
            out = env.render(return_depth=False, return_pc=False, resolution=resolution)
            frame = out["images"][0][..., :3]

        # Convert RGB->BGR for OpenCV
        bgr = frame[..., ::-1].copy()
        cv2.imshow(win, bgr)

        key = cv2.waitKey(delay_ms) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()


def _idle_loop_franka(env: Any, fps: float):
    dt = 1.0 / max(fps, 1e-6)
    print("[visualize_env] GUI idle. Ctrl+C to quit.")
    try:
        while True:
            env.sim.stepSimulation()
            time.sleep(dt)
    except KeyboardInterrupt:
        pass


def main():
    parser = argparse.ArgumentParser(description="Visualize PEFM environments (no demos).")
    parser.add_argument("--suite", choices=["franka", "mobile"], default="franka")
    parser.add_argument(
        "--task",
        type=str,
        default="peg_insert",
        help="Task name (depends on suite).",
    )

    # Common sim/env args
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_episode_length", type=int, default=1)
    parser.add_argument("--dof", type=int, default=7)
    parser.add_argument("--freq", type=int, default=5)
    parser.add_argument("--ac_noise", type=float, default=0.0)
    parser.add_argument("--randomize_rotation", action="store_true")
    parser.add_argument("--randomize_scale", action="store_true")
    parser.add_argument("--uniform_scaling", action="store_true")
    parser.add_argument("--scale_low", type=float, default=1.0)
    parser.add_argument("--scale_high", type=float, default=1.0)
    parser.add_argument("--scale_aspect_limit", type=float, default=100.0)

    # Visualization mode
    parser.add_argument(
        "--vis",
        action="store_true",
        help="Use PyBullet GUI (Franka suite only).",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render frames to an OpenCV window (works for both suites).",
    )
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--cam_resolution", type=int, default=240)

    cli = parser.parse_args()

    if cli.suite == "franka":
        EnvCls = _get_franka_env_class(cli.task)
        args = _make_args_franka(cli)
    else:
        EnvCls = _get_mobile_env_class(cli.task)
        args = _make_args_mobile(cli)
        if cli.vis:
            print(
                "[visualize_env] Note: sim_mobile envs run in DIRECT in current code; "
                "--vis has no effect. Use --render to see frames."
            )

    rng = np.random.RandomState(cli.seed)
    env = EnvCls(args, rng)
    env.reset()
    _maybe_print_setup_info(env)

    if cli.render:
        _render_loop(env, fps=cli.fps, resolution=cli.cam_resolution)
    elif cli.suite == "franka" and cli.vis:
        _idle_loop_franka(env, fps=240.0)
    else:
        print(
            "[visualize_env] Setup initialized. Nothing else to do.\n"
            "Tip: pass --render to see frames, or --vis (Franka) for GUI."
        )


if __name__ == "__main__":
    main()

