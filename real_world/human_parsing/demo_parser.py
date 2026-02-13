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
        """
        Initialize demo parser.

        Args:
            object_prompt: Text description for object detection
            device: Device for ML models ('cuda' or 'cpu')
            num_points: Number of points in output point clouds
            use_mock_hand_detector: Use mock detector for testing
        """
        self.object_prompt = object_prompt
        self.device = device
        self.num_points = num_points

        # Initialize components
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
    ) -> List[Dict]:
        """
        Parse single episode from raw RGB-D recordings.

        Args:
            episode_dir: Path to episode directory with depth_*.png and color_*.png
            intrinsics: Camera intrinsics dict with fx, fy, ppx, ppy
            target_fps: Output frame rate (EquiBot uses 3Hz)
            input_fps: Input recording frame rate (demo_capture.py uses 6Hz)

        Returns:
            frames: List of dicts, each containing:
                - 'pc': (num_points, 3) object point cloud
                - 'eef_pos': (13,) end-effector state
                - 'action': (7,) action (computed from state differences)
                - 'frame_idx': original frame index
        """
        # Load frame files
        depth_files = sorted(glob.glob(os.path.join(episode_dir, "depth_*.png")))
        color_files = sorted(glob.glob(os.path.join(episode_dir, "color_*.png")))

        if len(depth_files) == 0:
            warnings.warn(f"No depth files found in {episode_dir}")
            return []

        if len(depth_files) != len(color_files):
            warnings.warn(
                f"Mismatched depth ({len(depth_files)}) and color ({len(color_files)}) files"
            )
            # Use minimum
            n_frames = min(len(depth_files), len(color_files))
            depth_files = depth_files[:n_frames]
            color_files = color_files[:n_frames]

        # Compute frame skip for downsampling
        skip = max(1, input_fps // target_fps)

        # Reset tracker state
        self.object_tracker.reset_tracking()

        frames = []
        prev_masks = None

        for i in range(0, len(depth_files), skip):
            frame_data = self._process_frame(
                depth_files[i],
                color_files[i],
                intrinsics,
                prev_masks,
            )

            if frame_data is not None:
                frame_data['frame_idx'] = i
                frames.append(frame_data)
                prev_masks = frame_data.get('_masks')

        # Compute actions from state differences
        frames = self._compute_actions(frames, target_fps)

        # Remove internal mask storage
        for frame in frames:
            frame.pop('_masks', None)

        return frames

    def _process_frame(
        self,
        depth_path: str,
        color_path: str,
        intrinsics: dict,
        prev_masks: Optional[Dict] = None,
    ) -> Optional[Dict]:
        """
        Process single frame.

        Args:
            depth_path: Path to depth image (uint16 PNG)
            color_path: Path to color image (BGR PNG)
            intrinsics: Camera intrinsics
            prev_masks: Masks from previous frame for tracking

        Returns:
            frame_data: dict with 'pc', 'eef_pos', '_masks', or None if failed
        """
        # Load images
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)  # uint16
        color = cv2.imread(color_path)  # BGR

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
            # No hands detected - skip frame
            return None

        # 3. Align hands to point cloud frame
        try:
            aligned_hands = self.aligner.align(
                object_pc, hands, color, depth, intrinsics
            )
        except Exception as e:
            warnings.warn(f"Hand alignment failed: {e}")
            aligned_hands = hands  # Use unaligned

        # 4. Extract EEF pose from primary hand
        if len(aligned_hands) == 0:
            return None

        primary_hand = aligned_hands[0]  # Use first detected hand
        try:
            eef_pos = self.aligner.extract_eef_pose(primary_hand)
        except Exception as e:
            warnings.warn(f"EEF extraction failed: {e}")
            return None

        # 5. Downsample point cloud
        object_pc = self._downsample_pc(object_pc)

        return {
            'pc': object_pc.astype(np.float32),
            'eef_pos': eef_pos.astype(np.float32),
            '_masks': masks,  # Store for tracking continuity
        }

    def _downsample_pc(self, pc: np.ndarray) -> np.ndarray:
        """
        Downsample point cloud to fixed size.

        Args:
            pc: (N, 3) input point cloud

        Returns:
            downsampled: (num_points, 3) point cloud
        """
        N = pc.shape[0]

        if N == 0:
            return np.zeros((self.num_points, 3), dtype=np.float32)

        if N >= self.num_points:
            indices = np.random.choice(N, self.num_points, replace=False)
            return pc[indices].astype(np.float32)
        else:
            # Pad by repeating points
            indices = np.random.choice(N, self.num_points, replace=True)
            return pc[indices].astype(np.float32)

    def _compute_actions(
        self,
        frames: List[Dict],
        fps: int,
    ) -> List[Dict]:
        """
        Compute actions from state differences.

        Action format: [grip, vx, vy, vz, drx, dry, drz]
        - grip: gripper state (0 or 1)
        - vx, vy, vz: position velocities
        - drx, dry, drz: orientation velocities (simplified to zeros for now)

        Args:
            frames: list of frame dicts with 'eef_pos'
            fps: frame rate for velocity computation

        Returns:
            frames: updated with 'action' key
        """
        for i in range(len(frames)):
            if i < len(frames) - 1:
                curr_state = frames[i]['eef_pos']
                next_state = frames[i + 1]['eef_pos']

                # Position difference (indices 0:3)
                pos_diff = next_state[:3] - curr_state[:3]

                # Velocity = diff * fps
                vel = pos_diff * fps

                # Orientation velocities (simplified - use zeros for now)
                # Could compute from x_dir, z_dir changes if needed
                ori_vel = np.zeros(3, dtype=np.float32)

                # Gripper state from next frame
                grip = next_state[-1]

                # Assemble action: [grip, vx, vy, vz, drx, dry, drz]
                action = np.concatenate([[grip], vel, ori_vel]).astype(np.float32)
            else:
                # Last frame: zero action
                action = np.zeros(7, dtype=np.float32)

            frames[i]['action'] = action

        return frames


def parse_human_demo(
    episode_dir: str,
    intrinsics: dict,
    object_prompt: str = "object",
    target_fps: int = 3,
) -> List[Dict]:
    """
    Convenience function for parsing human demonstration.

    Args:
        episode_dir: Path to episode directory
        intrinsics: Camera intrinsics
        object_prompt: Text description for object detection
        target_fps: Output frame rate

    Returns:
        frames: List of parsed frame dicts
    """
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
        """
        Initialize simple parser.

        Args:
            num_points: Number of output points
            depth_min: Minimum depth in meters
            depth_max: Maximum depth in meters
            workspace_bounds: Optional workspace filtering bounds
        """
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

        Note: This does NOT extract hand poses - eef_pos will be placeholder.
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

            # Convert to point cloud with depth filtering
            pc = depth_to_pointcloud(
                depth, intrinsics,
                depth_scale=1000.0,
                max_depth=self.depth_max,
            )

            # Filter by minimum depth
            pc = pc[pc[:, 2] >= self.depth_min]

            # Optional workspace filtering
            if self.workspace_bounds is not None:
                pc = filter_workspace(pc, self.workspace_bounds)

            # Downsample
            pc = downsample_pointcloud(pc, self.num_points)

            # Placeholder EEF pose (13 zeros)
            eef_pos = np.zeros(13, dtype=np.float32)
            eef_pos[5] = -1  # z_dir pointing down
            eef_pos[11] = -1  # gravity pointing down

            frames.append({
                'pc': pc,
                'eef_pos': eef_pos,
                'frame_idx': i,
            })

        # Compute placeholder actions
        for i in range(len(frames)):
            frames[i]['action'] = np.zeros(7, dtype=np.float32)

        return frames
