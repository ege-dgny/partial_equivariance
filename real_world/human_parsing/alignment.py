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
    1. Find 2D-3D correspondences (hand keypoints in image → depth at those pixels)
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
    ):
        """
        Initialize aligner.

        Args:
            min_correspondences: Minimum number of valid 2D-3D matches for alignment
            gripper_close_threshold: Distance in meters below which gripper is "closed"
        """
        self.min_correspondences = min_correspondences
        self.gripper_close_threshold = gripper_close_threshold

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
        """
        aligned_hands = []

        for hand in hands:
            # Get 2D and 3D keypoints
            kp_2d = hand['keypoints_2d'].astype(np.float32)  # (21, 2)
            kp_3d_hamer = hand['keypoints_3d'].astype(np.float32)  # (21, 3) in HaMeR frame

            # Find 2D-3D correspondences by sampling depth at keypoint locations
            kp_3d_camera, valid_indices = self._sample_depth_at_keypoints(
                kp_2d, depth_image, intrinsics
            )

            # Check if we have enough correspondences
            if len(valid_indices) < self.min_correspondences:
                # Not enough correspondences - return unaligned hand
                aligned_hand = hand.copy()
                aligned_hand['alignment_success'] = False
                aligned_hand['transform_R'] = np.eye(3)
                aligned_hand['transform_t'] = np.zeros(3)
                aligned_hands.append(aligned_hand)
                continue

            # Get corresponding HaMeR keypoints
            kp_3d_hamer_valid = kp_3d_hamer[valid_indices]

            # Fit rigid transform: HaMeR → camera frame
            R, t = self._fit_rigid_transform(kp_3d_hamer_valid, kp_3d_camera)

            # Transform all keypoints to camera frame
            kp_3d_aligned = (R @ kp_3d_hamer.T).T + t

            # Create aligned hand dict
            aligned_hand = hand.copy()
            aligned_hand['keypoints_3d'] = kp_3d_aligned
            aligned_hand['transform_R'] = R
            aligned_hand['transform_t'] = t
            aligned_hand['alignment_success'] = True
            aligned_hands.append(aligned_hand)

        return aligned_hands

    def _sample_depth_at_keypoints(
        self,
        keypoints_2d: np.ndarray,
        depth_image: np.ndarray,
        intrinsics: dict,
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Sample depth at 2D keypoint locations and unproject to 3D.

        Args:
            keypoints_2d: (N, 2) pixel coordinates
            depth_image: (H, W) uint16 depth in mm
            intrinsics: camera intrinsics

        Returns:
            points_3d: (M, 3) valid 3D points in camera frame
            valid_indices: list of M indices into keypoints_2d
        """
        H, W = depth_image.shape
        fx, fy = intrinsics['fx'], intrinsics['fy']
        cx, cy = intrinsics['ppx'], intrinsics['ppy']

        points_3d = []
        valid_indices = []

        for i, (u, v) in enumerate(keypoints_2d):
            # Check if pixel is within image bounds
            u_int, v_int = int(round(u)), int(round(v))
            if not (0 <= u_int < W and 0 <= v_int < H):
                continue

            # Get depth value
            depth_mm = depth_image[v_int, u_int]
            if depth_mm == 0:
                continue  # Invalid depth

            # Convert to meters
            z = depth_mm / 1000.0

            # Unproject to 3D
            x = (u - cx) * z / fx
            y = (v - cy) * z / fy

            points_3d.append([x, y, z])
            valid_indices.append(i)

        return np.array(points_3d, dtype=np.float32), valid_indices

    def _fit_rigid_transform(
        self,
        src_points: np.ndarray,
        dst_points: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit rigid transform (R, t) from src to dst points using Kabsch algorithm.

        The Kabsch algorithm finds the optimal rotation matrix that minimizes
        the RMSD between two sets of paired points.

        Args:
            src_points: (N, 3) source points (HaMeR keypoints)
            dst_points: (N, 3) destination points (camera frame)

        Returns:
            R: (3, 3) rotation matrix
            t: (3,) translation vector

        Transform is applied as: dst = R @ src + t
        """
        assert src_points.shape == dst_points.shape
        assert src_points.shape[0] >= 3, "Need at least 3 points for rigid transform"

        # 1. Compute centroids
        src_centroid = src_points.mean(axis=0)
        dst_centroid = dst_points.mean(axis=0)

        # 2. Center the points
        src_centered = src_points - src_centroid
        dst_centered = dst_points - dst_centroid

        # 3. Compute cross-covariance matrix
        H = src_centered.T @ dst_centered  # (3, 3)

        # 4. SVD decomposition
        U, S, Vt = np.linalg.svd(H)

        # 5. Compute rotation matrix
        R = Vt.T @ U.T

        # 6. Handle reflection case (det(R) = -1)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        # 7. Compute translation
        t = dst_centroid - R @ src_centroid

        return R.astype(np.float32), t.astype(np.float32)

    def extract_eef_pose(self, aligned_hand: Dict) -> np.ndarray:
        """
        Extract "end-effector" pose from aligned hand.

        Uses thumb tip + index tip to define gripper pose:
        - Position: midpoint between thumb and index tips
        - Orientation: derived from finger direction
        - Gripper state: based on thumb-index distance

        Args:
            aligned_hand: dict from align() with keypoints_3d in camera frame

        Returns:
            eef_pos: (13,) array matching PEFM state format:
                [xyz(3), x_dir(3), z_dir(3), gravity(3), grip(1)]
        """
        keypoints_3d = aligned_hand['keypoints_3d']

        # Get fingertip positions (using standard hand keypoint indices)
        thumb_tip = keypoints_3d[4]   # Thumb tip
        index_tip = keypoints_3d[8]   # Index tip
        wrist = keypoints_3d[0]       # Wrist

        # Position: midpoint between thumb and index tips
        pos = (thumb_tip + index_tip) / 2

        # Orientation: construct frame from finger geometry
        # x_dir: along the finger axis (thumb to index direction, normalized)
        finger_vec = index_tip - thumb_tip
        finger_norm = np.linalg.norm(finger_vec)
        if finger_norm > 1e-6:
            finger_vec = finger_vec / finger_norm
        else:
            finger_vec = np.array([1, 0, 0])

        # z_dir: pointing "down" in world frame (gravity direction)
        # For now, use world gravity direction
        z_dir = np.array([0, 0, -1], dtype=np.float32)

        # x_dir: perpendicular to z_dir, in the finger plane
        # Project finger_vec onto plane perpendicular to z_dir
        x_dir = finger_vec - np.dot(finger_vec, z_dir) * z_dir
        x_norm = np.linalg.norm(x_dir)
        if x_norm > 1e-6:
            x_dir = x_dir / x_norm
        else:
            x_dir = np.array([1, 0, 0], dtype=np.float32)

        # Gripper state: based on thumb-index distance
        finger_dist = np.linalg.norm(thumb_tip - index_tip)
        grip = 1.0 if finger_dist < self.gripper_close_threshold else 0.0

        # Gravity vector (constant world frame direction)
        gravity = np.array([0, 0, -1], dtype=np.float32)

        # Assemble 13-dim state vector
        eef_pos = np.concatenate([
            pos.astype(np.float32),      # xyz (3)
            x_dir.astype(np.float32),    # x_dir (3)
            z_dir.astype(np.float32),    # z_dir (3)
            gravity,                      # gravity (3)
            [grip],                       # grip (1)
        ])

        return eef_pos

    def compute_alignment_error(
        self,
        src_points: np.ndarray,
        dst_points: np.ndarray,
        R: np.ndarray,
        t: np.ndarray,
    ) -> float:
        """
        Compute RMSD between transformed src points and dst points.

        Args:
            src_points: (N, 3) source points
            dst_points: (N, 3) target points
            R: (3, 3) rotation matrix
            t: (3,) translation vector

        Returns:
            rmsd: root mean squared distance
        """
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
    """
    Convenience function for hand alignment.

    Args:
        object_pc: (N, 3) object point cloud
        hands: list of detected hands
        rgb_image: RGB image
        depth_image: depth image (uint16 mm)
        intrinsics: camera intrinsics

    Returns:
        aligned_hands: list of aligned hand dicts
    """
    aligner = HandPointCloudAligner()
    return aligner.align(object_pc, hands, rgb_image, depth_image, intrinsics)
