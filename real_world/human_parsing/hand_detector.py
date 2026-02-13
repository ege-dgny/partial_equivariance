"""
Hand Detection using HaMeR (Hand Mesh Recovery).

Based on EquiBot paper Section F.1:
- Uses HaMeR for hand detection and 3D keypoint estimation
- Outputs 21 3D keypoints per hand (wrist + 5 fingers × 4 joints)

Reference: https://github.com/geopavlakos/hamer
"""

import numpy as np
from typing import Dict, List, Optional
import warnings

# Lazy import for heavy dependency
_hamer_model = None


def _load_hamer():
    """Lazy load HaMeR model."""
    global _hamer_model
    if _hamer_model is not None:
        return _hamer_model

    try:
        # HaMeR installation varies - try multiple import paths
        try:
            from hamer.models import HAMER
            from hamer.utils.renderer import Renderer
        except ImportError:
            # Alternative import path
            from hamer import HAMER

        import torch

        # Load model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Try to load from default checkpoint
        import os
        checkpoint_path = os.environ.get(
            'HAMER_CHECKPOINT',
            'weights/hamer_ckpt.pth'
        )

        model = HAMER.from_pretrained(checkpoint_path)
        model.to(device)
        model.eval()

        _hamer_model = {
            'model': model,
            'device': device,
        }
        return _hamer_model

    except ImportError as e:
        warnings.warn(
            f"HaMeR not installed: {e}\n"
            "Install with: pip install hamer\n"
            "Or clone: https://github.com/geopavlakos/hamer"
        )
        return None
    except Exception as e:
        warnings.warn(f"Failed to load HaMeR model: {e}")
        return None


# HaMeR hand keypoint indices
HAND_KEYPOINT_NAMES = [
    'wrist',
    'thumb_cmc', 'thumb_mcp', 'thumb_ip', 'thumb_tip',
    'index_mcp', 'index_pip', 'index_dip', 'index_tip',
    'middle_mcp', 'middle_pip', 'middle_dip', 'middle_tip',
    'ring_mcp', 'ring_pip', 'ring_dip', 'ring_tip',
    'pinky_mcp', 'pinky_pip', 'pinky_dip', 'pinky_tip',
]

# Key fingertip indices
THUMB_TIP_IDX = 4
INDEX_TIP_IDX = 8
MIDDLE_TIP_IDX = 12
RING_TIP_IDX = 16
PINKY_TIP_IDX = 20


class HandDetector:
    """
    HaMeR hand detection and mesh recovery.

    Outputs 3D hand keypoints (21 per hand: wrist + 5 fingers × 4 joints).

    Usage:
        detector = HandDetector()
        hands = detector.detect_hands(rgb_image)
        for hand in hands:
            thumb_tip, index_tip = detector.get_fingertip_positions(hand)
    """

    def __init__(self, device: str = "cuda", confidence_threshold: float = 0.5):
        """
        Initialize hand detector.

        Args:
            device: Device to run model on ('cuda' or 'cpu')
            confidence_threshold: Minimum confidence for hand detection
        """
        self.device = device
        self.confidence_threshold = confidence_threshold
        self._hamer = None

    def _ensure_model_loaded(self):
        """Load model if not already loaded."""
        if self._hamer is None:
            self._hamer = _load_hamer()

    def detect_hands(self, rgb_image: np.ndarray) -> List[Dict]:
        """
        Detect hands and extract keypoints.

        Args:
            rgb_image: (H, W, 3) RGB image (BGR format from cv2 is OK)

        Returns:
            hands: list of dicts, each containing:
                - 'keypoints_3d': (21, 3) 3D keypoints in HaMeR coordinate frame
                - 'keypoints_2d': (21, 2) 2D pixel coordinates
                - 'hand_side': 'left' or 'right'
                - 'confidence': detection confidence
                - 'bbox': (4,) bounding box [x1, y1, x2, y2]
        """
        self._ensure_model_loaded()

        if self._hamer is None:
            # Fallback: return empty results if model not available
            warnings.warn("HaMeR not loaded, returning empty hands list")
            return []

        try:
            return self._detect_with_hamer(rgb_image)
        except Exception as e:
            warnings.warn(f"HaMeR detection failed: {e}")
            return []

    def _detect_with_hamer(self, rgb_image: np.ndarray) -> List[Dict]:
        """
        Run HaMeR model for hand detection.

        This is a simplified implementation - actual HaMeR API may differ.
        """
        import torch
        from PIL import Image

        model = self._hamer['model']
        device = self._hamer['device']

        # Convert to RGB if BGR
        if rgb_image.shape[2] == 3:
            rgb_image = rgb_image[..., ::-1].copy()

        # Preprocess image
        pil_image = Image.fromarray(rgb_image)

        # Run inference
        with torch.no_grad():
            # Note: Actual HaMeR API may differ
            # This is a representative implementation
            results = model.predict(pil_image)

        hands = []
        for result in results:
            # Extract results - API depends on HaMeR version
            if hasattr(result, 'joints_3d'):
                keypoints_3d = result.joints_3d.cpu().numpy()  # (21, 3)
            else:
                keypoints_3d = np.zeros((21, 3), dtype=np.float32)

            if hasattr(result, 'joints_2d'):
                keypoints_2d = result.joints_2d.cpu().numpy()  # (21, 2)
            else:
                keypoints_2d = np.zeros((21, 2), dtype=np.float32)

            if hasattr(result, 'hand_side'):
                hand_side = result.hand_side
            else:
                hand_side = 'right'  # Default

            if hasattr(result, 'confidence'):
                confidence = float(result.confidence)
            else:
                confidence = 1.0

            if hasattr(result, 'bbox'):
                bbox = result.bbox.cpu().numpy()
            else:
                bbox = np.array([0, 0, rgb_image.shape[1], rgb_image.shape[0]])

            if confidence >= self.confidence_threshold:
                hands.append({
                    'keypoints_3d': keypoints_3d,
                    'keypoints_2d': keypoints_2d,
                    'hand_side': hand_side,
                    'confidence': confidence,
                    'bbox': bbox,
                })

        return hands

    def get_fingertip_positions(
        self, hand: Dict
    ) -> tuple:
        """
        Extract thumb tip and index tip positions.

        Args:
            hand: dict from detect_hands()

        Returns:
            thumb_tip: (3,) 3D position of thumb tip
            index_tip: (3,) 3D position of index tip
        """
        keypoints_3d = hand['keypoints_3d']
        thumb_tip = keypoints_3d[THUMB_TIP_IDX]
        index_tip = keypoints_3d[INDEX_TIP_IDX]
        return thumb_tip, index_tip

    def get_all_fingertips(self, hand: Dict) -> Dict[str, np.ndarray]:
        """
        Extract all fingertip positions.

        Args:
            hand: dict from detect_hands()

        Returns:
            fingertips: dict mapping finger name to (3,) position
        """
        keypoints_3d = hand['keypoints_3d']
        return {
            'thumb': keypoints_3d[THUMB_TIP_IDX],
            'index': keypoints_3d[INDEX_TIP_IDX],
            'middle': keypoints_3d[MIDDLE_TIP_IDX],
            'ring': keypoints_3d[RING_TIP_IDX],
            'pinky': keypoints_3d[PINKY_TIP_IDX],
        }

    def get_wrist_position(self, hand: Dict) -> np.ndarray:
        """Get wrist position."""
        return hand['keypoints_3d'][0]

    def estimate_gripper_width(self, hand: Dict) -> float:
        """
        Estimate gripper width from thumb-index distance.

        Returns:
            width: distance between thumb tip and index tip in meters
        """
        thumb_tip, index_tip = self.get_fingertip_positions(hand)
        return float(np.linalg.norm(thumb_tip - index_tip))

    def is_grasping(self, hand: Dict, threshold: float = 0.04) -> bool:
        """
        Check if hand is in grasping pose.

        Args:
            hand: dict from detect_hands()
            threshold: distance threshold in meters (default 4cm)

        Returns:
            True if thumb-index distance is below threshold
        """
        return self.estimate_gripper_width(hand) < threshold


class MockHandDetector(HandDetector):
    """
    Mock hand detector for testing without HaMeR installed.

    Returns synthetic hand keypoints based on simple heuristics.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._hamer = {'mock': True}  # Prevent loading real model

    def detect_hands(self, rgb_image: np.ndarray) -> List[Dict]:
        """Return mock hand detection."""
        H, W = rgb_image.shape[:2]

        # Generate mock keypoints in center of image
        center = np.array([W / 2, H / 2])

        # Generate 2D keypoints around center
        keypoints_2d = np.zeros((21, 2), dtype=np.float32)
        keypoints_2d[0] = center  # wrist
        for i in range(5):  # 5 fingers
            for j in range(4):  # 4 joints per finger
                idx = 1 + i * 4 + j
                angle = (i - 2) * 0.3  # Spread fingers
                dist = 20 + j * 15  # Increasing distance
                keypoints_2d[idx] = center + np.array([
                    np.sin(angle) * dist,
                    -dist  # Pointing up
                ])

        # Generate 3D keypoints (mock depth)
        keypoints_3d = np.zeros((21, 3), dtype=np.float32)
        keypoints_3d[:, :2] = (keypoints_2d - center) / 1000  # Scale to meters
        keypoints_3d[:, 2] = 0.5  # Mock depth at 0.5m

        return [{
            'keypoints_3d': keypoints_3d,
            'keypoints_2d': keypoints_2d,
            'hand_side': 'right',
            'confidence': 0.9,
            'bbox': np.array([W/4, H/4, 3*W/4, 3*H/4]),
        }]
