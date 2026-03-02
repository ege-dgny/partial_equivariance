"""
Complete Human Demo Parsing Pipeline.

Based on EquiBot paper Section F.1:
- Combines object tracking, hand detection, and alignment
- Parses raw RGB-D video recordings into training data
- Outputs object point clouds + human "end-effector" poses

Input: Raw RGB-D episode directories (from demo_capture.py)
Output: Parsed frames with 'pc', 'eef_pos', 'action' keys ready for PEFM training
"""

import os
import glob
import numpy as np
import cv2
from typing import Dict, List, Optional
import warnings

from .object_tracker import ObjectTracker
from .hand_detector import HandDetector, MockHandDetector
from .alignment import HandPointCloudAligner


class HumanDemoParser:
    """
    Complete pipeline for parsing human demonstrations.

    Workflow:
    1. Load RGB-D frames from episode directory
    2. Detect and track objects using Grounded SAM + DEVA
    3. Detect hands using HaMeR
    4. Align hand keypoints to point cloud frame
    5. Extract "end-effector" pose from hand
    6. Compute actions as state differences

    Usage:
        parser = HumanDemoParser(object_prompt="cup. block.")
        frames = parser.parse_episode(episode_dir, intrinsics)
        # frames is a list of dicts with 'pc', 'eef_pos', 'action' keys
    """

    def __init__(
        self,
        object_prompt: str = "object",
        device: str = "cuda",
        num_points: int = 1024,
        use_mock_hand_detector: bool = False,
    ):
        self.object_prompt = object_prompt
        self.device = device
        self.num_points = num_points

        self.object_tracker = ObjectTracker(
            text_prompt=object_prompt,
            device=device,
        )

        if use_mock_hand_detector:
            self.hand_detector = MockHandDetector(device=device)
        else:
            self.hand_detector = HandDetector(device=device)

        self.aligner = HandPointCloudAligner()

    def parse_episode(
        self,
        episode_dir: str,
        intrinsics: dict,
        target_fps: int = 3,
        input_fps: int = 6,
        save_vis: bool = False,
    ) -> List[Dict]:
        """
        Parse single episode from raw RGB-D recordings.

        Args:
            episode_dir: Path to episode directory with depth_*.png and color_*.png
            intrinsics: Camera intrinsics dict with fx, fy, ppx, ppy
            target_fps: Output frame rate (EquiBot uses 3Hz)
            input_fps: Input recording frame rate (demo_capture.py uses 6Hz)
            save_vis: If True, add 'vis_masks' and 'vis_keypoints_2d' to each frame
                for later video overlay (used by process_demos --save_vis).

        Returns:
            frames: List of dicts, each containing:
                - 'pc': (num_points, 3) object point cloud
                - 'eef_pos': (13,) end-effector state
                - 'action': (7,) action (computed from state differences)
                - 'frame_idx': original frame index
                - 'vis_masks', 'vis_keypoints_2d': (only when save_vis=True)
        """
        depth_files = sorted(glob.glob(os.path.join(episode_dir, "depth_*.png")))
        color_files = sorted(glob.glob(os.path.join(episode_dir, "color_*.png")))

        if len(depth_files) == 0:
            warnings.warn(f"No depth files found in {episode_dir}")
            return []

        if len(depth_files) != len(color_files):
            warnings.warn(
                f"Mismatched depth ({len(depth_files)}) and color ({len(color_files)}) files"
            )
            n_frames = min(len(depth_files), len(color_files))
            depth_files = depth_files[:n_frames]
            color_files = color_files[:n_frames]

        skip = max(1, input_fps // target_fps)

        self.object_tracker.reset_tracking()

        frames = []
        prev_masks = None

        for i in range(0, len(depth_files), skip):
            frame_data = self._process_frame(
                depth_files[i],
                color_files[i],
                intrinsics,
                prev_masks,
                save_vis=save_vis,
            )

            if frame_data is not None:
                frame_data['frame_idx'] = i
                frames.append(frame_data)
                prev_masks = frame_data.get('_masks')

        frames = self._compute_actions(frames, input_fps)

        for frame in frames:
            frame.pop('_masks', None)

        return frames

    def _process_frame(
        self,
        depth_path: str,
        color_path: str,
        intrinsics: dict,
        prev_masks: Optional[Dict] = None,
        save_vis: bool = False,
    ) -> Optional[Dict]:
        """
        Process single frame.

        Returns None if hand detection fails (frame is dropped).
        The caller tracks original frame indices so that velocity
        computation can account for temporal gaps.
        """
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        color = cv2.imread(color_path)

        if depth is None or color is None:
            warnings.warn(f"Failed to load {depth_path} or {color_path}")
            return None

        # 1. Object detection and tracking
        try:
            object_pc, masks = self.object_tracker.track(
                color, depth, intrinsics, prev_masks
            )
        except Exception as e:
            warnings.warn(f"Object tracking failed: {e}")
            object_pc = np.zeros((0, 3), dtype=np.float32)
            masks = {}

        # 2. Hand detection
        try:
            hands = self.hand_detector.detect_hands(color)
        except Exception as e:
            warnings.warn(f"Hand detection failed: {e}")
            hands = []

        if len(hands) == 0:
            return None

        # 3. Align hands to point cloud frame
        try:
            aligned_hands = self.aligner.align(
                object_pc, hands, color, depth, intrinsics
            )
        except Exception as e:
            warnings.warn(f"Hand alignment failed: {e}")
            aligned_hands = hands

        if len(aligned_hands) == 0:
            return None

        # 4. Extract EEF pose from primary hand
        primary_hand = aligned_hands[0]
        try:
            eef_pos = self.aligner.extract_eef_pose(primary_hand)
        except Exception as e:
            warnings.warn(f"EEF extraction failed: {e}")
            return None

        # 5. Downsample point cloud
        object_pc = self._downsample_pc(object_pc)

        out = {
            'pc': object_pc.astype(np.float32),
            'eef_pos': eef_pos.astype(np.float32),
            '_masks': masks,
        }
        if save_vis:
            out['vis_masks'] = masks
            out['vis_keypoints_2d'] = primary_hand['keypoints_2d'].astype(np.float32).copy()
        return out

    def _downsample_pc(self, pc: np.ndarray) -> np.ndarray:
        N = pc.shape[0]
        if N == 0:
            return np.zeros((self.num_points, 3), dtype=np.float32)
        if N >= self.num_points:
            indices = np.random.choice(N, self.num_points, replace=False)
        else:
            indices = np.random.choice(N, self.num_points, replace=True)
        return pc[indices].astype(np.float32)

    # ------------------------------------------------------------------
    # Action computation
    # ------------------------------------------------------------------

    def _compute_actions(
        self,
        frames: List[Dict],
        input_fps: int,
    ) -> List[Dict]:
        """
        Compute actions from consecutive state differences.

        Action format: [grip, vx, vy, vz, drx, dry, drz]

        Uses actual frame_idx difference to compute correct time deltas
        even when intermediate frames were dropped.
        """
        for i in range(len(frames)):
            if i < len(frames) - 1:
                curr = frames[i]['eef_pos']
                nxt = frames[i + 1]['eef_pos']

                # Actual time delta (accounts for dropped frames)
                frame_gap = frames[i + 1]['frame_idx'] - frames[i]['frame_idx']
                dt = frame_gap / input_fps
                if dt < 1e-6:
                    dt = 1.0 / input_fps

                # Position velocity
                pos_diff = nxt[:3] - curr[:3]
                vel = pos_diff / dt

                # Orientation velocity from x_dir / z_dir change
                ori_vel = self._compute_orientation_velocity(
                    curr[3:6], curr[6:9], nxt[3:6], nxt[6:9], dt
                )

                grip = nxt[-1]

                action = np.concatenate(
                    [[grip], vel, ori_vel]
                ).astype(np.float32)
            else:
                action = np.zeros(7, dtype=np.float32)

            frames[i]['action'] = action

        return frames

    @staticmethod
    def _compute_orientation_velocity(
        x_dir_curr: np.ndarray,
        z_dir_curr: np.ndarray,
        x_dir_next: np.ndarray,
        z_dir_next: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """
        Compute angular velocity (drx, dry, drz) from two orientation
        frames defined by (x_dir, z_dir).

        Constructs full rotation matrices, computes the relative rotation,
        and converts to a rotation vector divided by dt.
        """
        def _build_frame(x, z):
            x = x / (np.linalg.norm(x) + 1e-8)
            z = z / (np.linalg.norm(z) + 1e-8)
            y = np.cross(z, x)
            y = y / (np.linalg.norm(y) + 1e-8)
            x = np.cross(y, z)
            return np.stack([x, y, z], axis=1)  # (3, 3) columns = axes

        def _rotmat_to_rotvec(R):
            cos_angle = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
            angle = np.arccos(cos_angle)
            if angle < 1e-6:
                return np.zeros(3, dtype=np.float32)
            axis = np.array([
                R[2, 1] - R[1, 2],
                R[0, 2] - R[2, 0],
                R[1, 0] - R[0, 1],
            ]) / (2.0 * np.sin(angle) + 1e-8)
            return (axis * angle).astype(np.float32)

        R_curr = _build_frame(x_dir_curr, z_dir_curr)
        R_next = _build_frame(x_dir_next, z_dir_next)
        R_rel = R_next @ R_curr.T
        rotvec = _rotmat_to_rotvec(R_rel)
        return rotvec / dt


def parse_human_demo(
    episode_dir: str,
    intrinsics: dict,
    object_prompt: str = "object",
    target_fps: int = 3,
) -> List[Dict]:
    """Convenience function for parsing a human demonstration."""
    parser = HumanDemoParser(object_prompt=object_prompt)
    return parser.parse_episode(episode_dir, intrinsics, target_fps=target_fps)


class SimpleDemoParser:
    """
    Simplified demo parser without ML models.

    Uses simple depth-based filtering instead of object detection.
    Useful for testing or when ML models are not installed.
    """

    def __init__(
        self,
        num_points: int = 1024,
        depth_min: float = 0.2,
        depth_max: float = 1.5,
        workspace_bounds: Optional[dict] = None,
    ):
        self.num_points = num_points
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.workspace_bounds = workspace_bounds

    def parse_episode(
        self,
        episode_dir: str,
        intrinsics: dict,
        target_fps: int = 3,
        input_fps: int = 6,
    ) -> List[Dict]:
        """
        Parse episode with simple depth filtering.

        Note: This does NOT extract hand poses — eef_pos will be placeholder.
        Use this only for testing point cloud processing.
        """
        from real_world.utils.point_cloud import (
            depth_to_pointcloud,
            filter_workspace,
            downsample_pointcloud,
        )

        depth_files = sorted(glob.glob(os.path.join(episode_dir, "depth_*.png")))
        skip = max(1, input_fps // target_fps)

        frames = []

        for i in range(0, len(depth_files), skip):
            depth = cv2.imread(depth_files[i], cv2.IMREAD_UNCHANGED)
            if depth is None:
                continue

            pc = depth_to_pointcloud(
                depth, intrinsics,
                depth_scale=1000.0,
                max_depth=self.depth_max,
            )
            pc = pc[pc[:, 2] >= self.depth_min]

            if self.workspace_bounds is not None:
                pc = filter_workspace(pc, self.workspace_bounds)

            pc = downsample_pointcloud(pc, self.num_points)

            eef_pos = np.zeros(13, dtype=np.float32)
            eef_pos[5] = -1  # z_dir pointing down
            eef_pos[11] = -1  # gravity pointing down

            frames.append({
                'pc': pc,
                'eef_pos': eef_pos,
                'frame_idx': i,
            })

        for i in range(len(frames)):
            frames[i]['action'] = np.zeros(7, dtype=np.float32)

        return frames
