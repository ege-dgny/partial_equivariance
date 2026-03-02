"""
Hand-to-Pointcloud Alignment Module.

Based on EquiBot paper Section F.1:
- Aligns hand keypoints from HaMeR's coordinate frame to the point cloud frame
- Uses Kabsch algorithm (SVD-based) to fit rigid transform
- Extracts "end-effector" pose from thumb and index finger positions

This alignment is CRITICAL because:
- Hand detection (HaMeR) outputs in its own coordinate frame
- Point cloud is in camera frame
- Without alignment, hand poses won't match object positions
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


class HandPointCloudAligner:
    """
    Aligns hand keypoints to point cloud coordinate frame.

    The alignment process:
    1. Find 2D-3D correspondences (hand keypoints in image -> depth at those pixels)
    2. Fit rigid transform (R, t) from HaMeR frame to camera/PC frame using Kabsch algorithm
    3. Apply transform to all hand keypoints

    Usage:
        aligner = HandPointCloudAligner()
        aligned_hands = aligner.align(object_pc, hands, rgb, depth, intrinsics)
        for hand in aligned_hands:
            eef_pos = aligner.extract_eef_pose(hand)
    """

    def __init__(
        self,
        min_correspondences: int = 4,
        gripper_close_threshold: float = 0.04,
        depth_patch_radius: int = 2,
        gravity_in_camera: Optional[np.ndarray] = None,
    ):
        """
        Args:
            min_correspondences: Minimum number of valid 2D-3D matches for alignment
            gripper_close_threshold: Distance in meters below which gripper is "closed"
            depth_patch_radius: Half-size of neighborhood patch for depth sampling
                (radius=2 gives a 5x5 patch). Reduces noise from single-pixel lookup.
            gravity_in_camera: (3,) gravity direction expressed in camera frame.
                Default [0, 1, 0] assumes camera is level and looking forward
                (Y-down in OpenCV convention). Set to [0, 0, 1] if camera looks
                straight down. Must be a unit vector.
        """
        self.min_correspondences = min_correspondences
        self.gripper_close_threshold = gripper_close_threshold
        self.depth_patch_radius = depth_patch_radius

        if gravity_in_camera is not None:
            g = np.asarray(gravity_in_camera, dtype=np.float32)
            self.gravity_in_camera = g / (np.linalg.norm(g) + 1e-8)
        else:
            self.gravity_in_camera = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    def align(
        self,
        object_pc: np.ndarray,
        hands: List[Dict],
        rgb_image: np.ndarray,
        depth_image: np.ndarray,
        intrinsics: dict,
    ) -> List[Dict]:
        """
        Align hand 3D keypoints to point cloud coordinates.

        Args:
            object_pc: (N, 3) object point cloud in camera frame
            hands: list of detected hands from HandDetector
            rgb_image: (H, W, 3) original RGB image
            depth_image: (H, W) depth image (uint16 in mm), aligned to RGB
            intrinsics: camera intrinsics with 'fx', 'fy', 'ppx', 'ppy'

        Returns:
            aligned_hands: list of hands with transformed keypoints_3d
                Each hand dict now also contains:
                - 'keypoints_3d': transformed (21, 3) keypoints in camera frame
                - 'transform_R': (3, 3) rotation matrix
                - 'transform_t': (3,) translation vector
                - 'alignment_success': bool indicating if alignment succeeded
                - 'alignment_rmsd': float, RMSD of the fit (lower is better)
        """
        aligned_hands = []

        for hand in hands:
            kp_2d = hand['keypoints_2d'].astype(np.float32)  # (21, 2)
            kp_3d_hamer = hand['keypoints_3d'].astype(np.float32)  # (21, 3)

            kp_3d_camera, valid_indices = self._sample_depth_at_keypoints(
                kp_2d, depth_image, intrinsics
            )

            if len(valid_indices) < self.min_correspondences:
                aligned_hand = hand.copy()
                aligned_hand['alignment_success'] = False
                aligned_hand['transform_R'] = np.eye(3, dtype=np.float32)
                aligned_hand['transform_t'] = np.zeros(3, dtype=np.float32)
                aligned_hand['alignment_rmsd'] = float('inf')
                aligned_hands.append(aligned_hand)
                continue

            kp_3d_hamer_valid = kp_3d_hamer[valid_indices]

            R, t = self._fit_rigid_transform(kp_3d_hamer_valid, kp_3d_camera)
            kp_3d_aligned = (R @ kp_3d_hamer.T).T + t

            rmsd = self.compute_alignment_error(
                kp_3d_hamer_valid, kp_3d_camera, R, t
            )

            aligned_hand = hand.copy()
            aligned_hand['keypoints_3d'] = kp_3d_aligned
            aligned_hand['transform_R'] = R
            aligned_hand['transform_t'] = t
            aligned_hand['alignment_success'] = True
            aligned_hand['alignment_rmsd'] = rmsd
            aligned_hands.append(aligned_hand)

        return aligned_hands

    def _sample_depth_at_keypoints(
        self,
        keypoints_2d: np.ndarray,
        depth_image: np.ndarray,
        intrinsics: dict,
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Sample depth at 2D keypoint locations using a local neighborhood
        median and unproject to 3D.

        Uses a (2*radius+1)^2 patch around each keypoint and takes the
        median of non-zero depths.  This is much more robust than single-pixel
        lookup, especially at finger boundaries where depth is noisy.

        Returns:
            points_3d: (M, 3) valid 3D points in camera frame
            valid_indices: list of M indices into keypoints_2d
        """
        H, W = depth_image.shape
        fx, fy = intrinsics['fx'], intrinsics['fy']
        cx, cy = intrinsics['ppx'], intrinsics['ppy']
        r = self.depth_patch_radius

        points_3d = []
        valid_indices = []

        for i, (u, v) in enumerate(keypoints_2d):
            u_int, v_int = int(round(u)), int(round(v))
            if not (0 <= u_int < W and 0 <= v_int < H):
                continue

            v_lo = max(0, v_int - r)
            v_hi = min(H, v_int + r + 1)
            u_lo = max(0, u_int - r)
            u_hi = min(W, u_int + r + 1)

            patch = depth_image[v_lo:v_hi, u_lo:u_hi].astype(np.float64)
            valid_depths = patch[patch > 0]

            if len(valid_depths) == 0:
                continue

            depth_mm = float(np.median(valid_depths))
            z = depth_mm / 1000.0

            x = (u - cx) * z / fx
            y = (v - cy) * z / fy

            points_3d.append([x, y, z])
            valid_indices.append(i)

        if len(points_3d) == 0:
            return np.zeros((0, 3), dtype=np.float32), []
        return np.array(points_3d, dtype=np.float32), valid_indices

    def _fit_rigid_transform(
        self,
        src_points: np.ndarray,
        dst_points: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit rigid transform (R, t) from src to dst points using Kabsch algorithm.

        Transform is applied as: dst = R @ src + t
        """
        assert src_points.shape == dst_points.shape
        assert src_points.shape[0] >= 3, "Need at least 3 points for rigid transform"

        src_centroid = src_points.mean(axis=0)
        dst_centroid = dst_points.mean(axis=0)

        src_centered = src_points - src_centroid
        dst_centered = dst_points - dst_centroid

        H = src_centered.T @ dst_centered
        U, S, Vt = np.linalg.svd(H)

        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        t = dst_centroid - R @ src_centroid
        return R.astype(np.float32), t.astype(np.float32)

    def extract_eef_pose(self, aligned_hand: Dict) -> np.ndarray:
        """
        Extract "end-effector" pose from aligned hand.

        Uses thumb tip + index tip to define gripper pose:
        - Position: midpoint between thumb and index tips
        - Orientation: x_dir from thumb-to-index projection on the
          plane perpendicular to gravity;  z_dir from wrist-to-fingertip
          direction projected onto the gravity axis.
        - Gripper state: based on thumb-index distance

        NOTE: The resulting x_dir/z_dir represent the human *hand*
        orientation, NOT a Franka EEF frame.  Downstream training
        must use the same convention at inference time (e.g., map
        the robot gripper frame into the same thumb/index convention
        via a fixed transform).

        Returns:
            eef_pos: (13,) array matching PEFM state format:
                [xyz(3), x_dir(3), z_dir(3), gravity(3), grip(1)]
        """
        keypoints_3d = aligned_hand['keypoints_3d']

        thumb_tip = keypoints_3d[4]
        index_tip = keypoints_3d[8]
        wrist = keypoints_3d[0]
        middle_mcp = keypoints_3d[9]

        pos = (thumb_tip + index_tip) / 2.0

        # --- z_dir: direction the palm faces (roughly wrist → middle_mcp)
        palm_vec = middle_mcp - wrist
        palm_norm = np.linalg.norm(palm_vec)
        if palm_norm > 1e-6:
            z_dir = palm_vec / palm_norm
        else:
            z_dir = -self.gravity_in_camera.copy()

        # --- x_dir: perpendicular to z_dir, along thumb→index axis
        finger_vec = index_tip - thumb_tip
        x_dir = finger_vec - np.dot(finger_vec, z_dir) * z_dir
        x_norm = np.linalg.norm(x_dir)
        if x_norm > 1e-6:
            x_dir = (x_dir / x_norm).astype(np.float32)
        else:
            # Fallback: pick any direction perpendicular to z_dir
            arb = np.array([1, 0, 0], dtype=np.float32)
            if abs(np.dot(arb, z_dir)) > 0.9:
                arb = np.array([0, 1, 0], dtype=np.float32)
            x_dir = np.cross(z_dir, arb)
            x_dir = (x_dir / (np.linalg.norm(x_dir) + 1e-8)).astype(np.float32)

        # --- Gripper state
        finger_dist = np.linalg.norm(thumb_tip - index_tip)
        grip = 1.0 if finger_dist < self.gripper_close_threshold else 0.0

        gravity = self.gravity_in_camera.copy()

        eef_pos = np.concatenate([
            pos.astype(np.float32),
            x_dir.astype(np.float32),
            z_dir.astype(np.float32),
            gravity,
            [np.float32(grip)],
        ])

        return eef_pos

    def compute_alignment_error(
        self,
        src_points: np.ndarray,
        dst_points: np.ndarray,
        R: np.ndarray,
        t: np.ndarray,
    ) -> float:
        """Compute RMSD between transformed src points and dst points."""
        transformed = (R @ src_points.T).T + t
        diff = transformed - dst_points
        rmsd = np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))
        return float(rmsd)


def align_hands_to_pointcloud(
    object_pc: np.ndarray,
    hands: List[Dict],
    rgb_image: np.ndarray,
    depth_image: np.ndarray,
    intrinsics: dict,
) -> List[Dict]:
    """Convenience function for hand alignment."""
    aligner = HandPointCloudAligner()
    return aligner.align(object_pc, hands, rgb_image, depth_image, intrinsics)
