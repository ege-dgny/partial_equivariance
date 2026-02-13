"""
Real-world data collection and robot deployment for PEFM.

This package provides:
1. Human Demo Parsing - Extract object point clouds + hand poses from RGB-D video
2. Data Processing - Convert demos to PEFM-compatible NPZ format
3. Robot Interface - Real Franka Panda control (placeholder)
4. Real Inference - Closed-loop policy execution (placeholder)

Based on EquiBot paper Section F.1 for human demo infrastructure.

Usage:
    # Parse human demonstrations
    from real_world.human_parsing import HumanDemoParser
    parser = HumanDemoParser(object_prompt="cup")
    frames = parser.parse_episode(episode_dir, intrinsics)

    # Process demos to NPZ format
    python -m real_world.process_demos --input_dir raw_demos --output_dir processed

    # Record raw RGB-D demos
    python -m real_world.demo_capture
"""

from .utils.point_cloud import (
    depth_to_pointcloud,
    filter_workspace,
    downsample_pointcloud,
    transform_to_robot_frame,
    load_intrinsics,
)

__all__ = [
    "depth_to_pointcloud",
    "filter_workspace",
    "downsample_pointcloud",
    "transform_to_robot_frame",
    "load_intrinsics",
]
