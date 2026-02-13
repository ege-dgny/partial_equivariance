"""
Point cloud utilities for real-world PEFM.

Provides depth-to-pointcloud conversion, workspace filtering, and downsampling.
Reference: pefm_envs/sim_mobile/utils/project.py for vectorized_unproject pattern.
"""

import numpy as np


def depth_to_pointcloud(
    depth_img,
    intrinsics,
    depth_scale=1000.0,
    filter_mask=None,
    max_depth=2.0,
):
    """
    Convert depth image to point cloud using camera intrinsics.

    Uses vectorized unprojection for efficiency.

    Args:
        depth_img: (H, W) uint16 depth image (typically in mm)
        intrinsics: dict with keys 'fx', 'fy', 'ppx', 'ppy'
                    OR (3, 3) intrinsic matrix K
        depth_scale: conversion factor (1000 for mm→meters)
        filter_mask: optional (H, W) bool mask, keep only True pixels
        max_depth: maximum depth to include (meters), filters far points

    Returns:
        pc: (N, 3) point cloud in camera frame
    """
    # Parse intrinsics
    if isinstance(intrinsics, dict):
        fx = intrinsics['fx']
        fy = intrinsics['fy']
        cx = intrinsics['ppx']
        cy = intrinsics['ppy']
    else:
        # Assume (3,3) matrix
        fx = intrinsics[0, 0]
        fy = intrinsics[1, 1]
        cx = intrinsics[0, 2]
        cy = intrinsics[1, 2]

    # Convert depth to meters
    depth = depth_img.astype(np.float32) / depth_scale

    # Create pixel coordinate grids
    H, W = depth.shape
    u = np.arange(W, dtype=np.float32)
    v = np.arange(H, dtype=np.float32)
    u, v = np.meshgrid(u, v)

    # Apply mask if provided
    if filter_mask is not None:
        valid = filter_mask & (depth > 0) & (depth < max_depth)
    else:
        valid = (depth > 0) & (depth < max_depth)

    # Get valid pixel coordinates and depths
    u_valid = u[valid]
    v_valid = v[valid]
    z_valid = depth[valid]

    # Unproject: (u, v, z) -> (x, y, z)
    x = (u_valid - cx) * z_valid / fx
    y = (v_valid - cy) * z_valid / fy

    # Stack into point cloud
    pc = np.stack([x, y, z_valid], axis=-1)

    return pc


def filter_workspace(pc, bounds=None):
    """
    Filter point cloud to workspace bounds.

    Args:
        pc: (N, 3) point cloud
        bounds: dict with keys 'x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max'
                OR None for default Franka workspace

    Returns:
        filtered_pc: (M, 3) filtered point cloud
    """
    if bounds is None:
        # Default Franka workspace (meters from robot base)
        bounds = {
            'x_min': 0.2,
            'x_max': 0.8,
            'y_min': -0.5,
            'y_max': 0.5,
            'z_min': -0.02,
            'z_max': 0.5,
        }

    mask = (
        (pc[:, 0] >= bounds['x_min']) & (pc[:, 0] <= bounds['x_max']) &
        (pc[:, 1] >= bounds['y_min']) & (pc[:, 1] <= bounds['y_max']) &
        (pc[:, 2] >= bounds['z_min']) & (pc[:, 2] <= bounds['z_max'])
    )

    return pc[mask]


def downsample_pointcloud(pc, num_points=1024, method='random'):
    """
    Downsample point cloud to fixed number of points.

    Args:
        pc: (N, 3) input point cloud
        num_points: target number of points
        method: 'random' for random subsampling, 'uniform' for uniform stride

    Returns:
        downsampled: (num_points, 3) point cloud
    """
    N = pc.shape[0]

    if N == 0:
        # Return zeros if empty
        return np.zeros((num_points, 3), dtype=np.float32)

    if N >= num_points:
        if method == 'random':
            indices = np.random.choice(N, num_points, replace=False)
        else:  # uniform
            step = N // num_points
            indices = np.arange(0, N, step)[:num_points]
        return pc[indices].astype(np.float32)
    else:
        # Pad by repeating points if not enough
        if method == 'random':
            indices = np.random.choice(N, num_points, replace=True)
        else:
            # Repeat uniformly
            repeats = (num_points // N) + 1
            indices = np.tile(np.arange(N), repeats)[:num_points]
        return pc[indices].astype(np.float32)


def transform_to_robot_frame(pc, T_robot_camera):
    """
    Transform point cloud from camera frame to robot base frame.

    Args:
        pc: (N, 3) point cloud in camera frame
        T_robot_camera: (4, 4) transformation matrix from camera to robot frame

    Returns:
        pc_robot: (N, 3) point cloud in robot frame
    """
    # Homogeneous coordinates
    N = pc.shape[0]
    pc_homo = np.concatenate([pc, np.ones((N, 1))], axis=1)  # (N, 4)

    # Apply transformation
    pc_robot_homo = (T_robot_camera @ pc_homo.T).T  # (N, 4)

    return pc_robot_homo[:, :3].astype(np.float32)


def load_intrinsics(intrinsics_path):
    """
    Load camera intrinsics from JSON file (saved by demo_capture.py).

    Args:
        intrinsics_path: path to intrinsics.json

    Returns:
        intrinsics: dict with fx, fy, ppx, ppy, width, height
    """
    import json
    with open(intrinsics_path, 'r') as f:
        intrinsics = json.load(f)
    return intrinsics


def create_intrinsic_matrix(intrinsics):
    """
    Create 3x3 intrinsic matrix from intrinsics dict.

    Args:
        intrinsics: dict with fx, fy, ppx, ppy

    Returns:
        K: (3, 3) intrinsic matrix
    """
    fx = intrinsics['fx']
    fy = intrinsics['fy']
    cx = intrinsics['ppx']
    cy = intrinsics['ppy']

    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1],
    ], dtype=np.float32)

    return K
