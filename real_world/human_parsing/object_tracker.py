"""
Object Detection and Tracking using Grounded SAM + DEVA.

Based on EquiBot paper Section F.1:
- Uses Grounding DINO for open-vocabulary object detection
- Uses SAM (Segment Anything Model) for instance segmentation
- Uses DEVA for video object tracking with consistent IDs

References:
- GroundingDINO: https://github.com/IDEA-Research/GroundingDINO
- SAM: https://github.com/facebookresearch/segment-anything
- DEVA: https://github.com/hkchengrex/Tracking-Anything-with-DEVA
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import warnings

# Lazy imports for heavy dependencies
_grounding_dino = None
_sam = None
_deva = None


def _load_grounding_dino():
    """Lazy load Grounding DINO model."""
    global _grounding_dino
    if _grounding_dino is not None:
        return _grounding_dino

    try:
        from groundingdino.util.inference import load_model, predict
        import groundingdino.datasets.transforms as T
        from groundingdino.util import box_ops

        # Load model with default weights
        # Users should set GROUNDING_DINO_CONFIG and GROUNDING_DINO_CHECKPOINT env vars
        import os
        config_path = os.environ.get(
            'GROUNDING_DINO_CONFIG',
            'GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py'
        )
        checkpoint_path = os.environ.get(
            'GROUNDING_DINO_CHECKPOINT',
            'weights/groundingdino_swint_ogc.pth'
        )

        model = load_model(config_path, checkpoint_path)
        _grounding_dino = {
            'model': model,
            'predict': predict,
            'transforms': T,
            'box_ops': box_ops,
        }
        return _grounding_dino

    except ImportError as e:
        warnings.warn(
            f"Grounding DINO not installed: {e}\n"
            "Install with: pip install groundingdino-py\n"
            "Or clone: https://github.com/IDEA-Research/GroundingDINO"
        )
        return None


def _load_sam():
    """Lazy load SAM model."""
    global _sam
    if _sam is not None:
        return _sam

    try:
        from segment_anything import sam_model_registry, SamPredictor

        import os
        checkpoint_path = os.environ.get(
            'SAM_CHECKPOINT',
            'weights/sam_vit_h_4b8939.pth'
        )
        model_type = os.environ.get('SAM_MODEL_TYPE', 'vit_h')

        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        sam.cuda()
        predictor = SamPredictor(sam)

        _sam = {'predictor': predictor}
        return _sam

    except ImportError as e:
        warnings.warn(
            f"SAM not installed: {e}\n"
            "Install with: pip install segment-anything\n"
            "Or clone: https://github.com/facebookresearch/segment-anything"
        )
        return None


def _load_deva():
    """Lazy load DEVA tracker."""
    global _deva
    if _deva is not None:
        return _deva

    try:
        # DEVA integration - this is a placeholder for the actual import
        # The exact import depends on how DEVA is installed
        from deva.inference.inference_core import DEVAInferenceCore
        from deva.inference.eval_args import add_common_eval_args

        # Default config - users should customize
        _deva = {'core': DEVAInferenceCore}
        return _deva

    except ImportError as e:
        warnings.warn(
            f"DEVA not installed: {e}\n"
            "Install with: pip install deva-track\n"
            "Or clone: https://github.com/hkchengrex/Tracking-Anything-with-DEVA"
        )
        return None


class ObjectTracker:
    """
    Grounded SAM + DEVA for object detection and tracking.

    Outputs segmented point cloud containing only objects of interest.

    Usage:
        tracker = ObjectTracker(text_prompt="cup. block. peg.")
        object_pc, masks = tracker.detect_and_segment(rgb, depth, intrinsics)

        # For video tracking:
        for frame in video:
            object_pc, masks = tracker.track(rgb, depth, intrinsics, prev_masks)
    """

    def __init__(
        self,
        text_prompt: str = "object",
        device: str = "cuda",
        box_threshold: float = 0.3,
        text_threshold: float = 0.25,
    ):
        """
        Initialize object tracker.

        Args:
            text_prompt: Text description of objects to detect (e.g., "cup. block.")
            device: Device to run models on ('cuda' or 'cpu')
            box_threshold: Confidence threshold for bounding box detection
            text_threshold: Confidence threshold for text matching
        """
        self.text_prompt = text_prompt
        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

        # Lazy-load models on first use
        self._grounding_dino = None
        self._sam = None
        self._deva = None

        # Tracking state
        self._prev_masks = None
        self._tracking_initialized = False

    def _ensure_models_loaded(self):
        """Load models if not already loaded."""
        if self._grounding_dino is None:
            self._grounding_dino = _load_grounding_dino()
        if self._sam is None:
            self._sam = _load_sam()

    def _ensure_deva_loaded(self):
        """Load DEVA tracker if not already loaded."""
        if self._deva is None:
            self._deva = _load_deva()

    def detect_and_segment(
        self,
        rgb_image: np.ndarray,
        depth_image: np.ndarray,
        intrinsics: dict,
    ) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
        """
        Single-frame detection and segmentation.

        Args:
            rgb_image: (H, W, 3) RGB image (BGR format from cv2 is OK)
            depth_image: (H, W) depth image (uint16 in mm)
            intrinsics: dict with 'fx', 'fy', 'ppx', 'ppy'

        Returns:
            object_pc: (N, 3) segmented point cloud of objects
            masks: dict mapping object_id -> (H, W) binary mask
        """
        self._ensure_models_loaded()

        if self._grounding_dino is None or self._sam is None:
            # Fallback: return empty results if models not available
            warnings.warn("Models not loaded, returning empty point cloud")
            return np.zeros((0, 3), dtype=np.float32), {}

        # 1. Run Grounding DINO for detection
        boxes, labels, scores = self._detect_objects(rgb_image)

        if len(boxes) == 0:
            return np.zeros((0, 3), dtype=np.float32), {}

        # 2. Run SAM for segmentation
        masks = self._segment_from_boxes(rgb_image, boxes)

        # 3. Convert masked depth to point cloud
        object_pc = self._masked_depth_to_pc(depth_image, masks, intrinsics)

        return object_pc, masks

    def track(
        self,
        rgb_image: np.ndarray,
        depth_image: np.ndarray,
        intrinsics: dict,
        prev_masks: Optional[Dict[int, np.ndarray]] = None,
    ) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
        """
        Track objects across frames using DEVA.

        Args:
            rgb_image: (H, W, 3) RGB image
            depth_image: (H, W) depth image
            intrinsics: camera intrinsics
            prev_masks: masks from previous frame (None for first frame)

        Returns:
            object_pc: (N, 3) tracked object point cloud
            masks: updated masks with consistent IDs
        """
        if prev_masks is None:
            # First frame: detect and segment
            return self.detect_and_segment(rgb_image, depth_image, intrinsics)

        self._ensure_deva_loaded()

        if self._deva is None:
            # Fallback: re-detect each frame if DEVA not available
            warnings.warn("DEVA not loaded, falling back to per-frame detection")
            return self.detect_and_segment(rgb_image, depth_image, intrinsics)

        # Track masks using DEVA
        masks = self._track_masks(rgb_image, prev_masks)

        # Convert to point cloud
        object_pc = self._masked_depth_to_pc(depth_image, masks, intrinsics)

        return object_pc, masks

    def _detect_objects(
        self, rgb_image: np.ndarray
    ) -> Tuple[np.ndarray, List[str], np.ndarray]:
        """
        Detect objects using Grounding DINO.

        Returns:
            boxes: (N, 4) bounding boxes in xyxy format
            labels: list of N label strings
            scores: (N,) confidence scores
        """
        import torch
        from PIL import Image

        # Convert to PIL Image
        if rgb_image.shape[2] == 3:
            # Assume BGR from cv2, convert to RGB
            rgb_image = rgb_image[..., ::-1]
        pil_image = Image.fromarray(rgb_image)

        # Apply transforms
        transform = self._grounding_dino['transforms'].Compose([
            self._grounding_dino['transforms'].RandomResize([800], max_size=1333),
            self._grounding_dino['transforms'].ToTensor(),
            self._grounding_dino['transforms'].Normalize([0.485, 0.456, 0.406],
                                                         [0.229, 0.224, 0.225]),
        ])
        image_transformed, _ = transform(pil_image, None)

        # Run detection
        boxes, logits, phrases = self._grounding_dino['predict'](
            model=self._grounding_dino['model'],
            image=image_transformed,
            caption=self.text_prompt,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
        )

        # Convert boxes to image coordinates
        H, W = rgb_image.shape[:2]
        boxes = boxes * torch.tensor([W, H, W, H])
        boxes = boxes.cpu().numpy()
        scores = logits.cpu().numpy()

        return boxes, phrases, scores

    def _segment_from_boxes(
        self, rgb_image: np.ndarray, boxes: np.ndarray
    ) -> Dict[int, np.ndarray]:
        """
        Generate segmentation masks from bounding boxes using SAM.

        Returns:
            masks: dict mapping object_id -> (H, W) binary mask
        """
        predictor = self._sam['predictor']

        # Set image
        if rgb_image.shape[2] == 3:
            rgb_image = rgb_image[..., ::-1]  # BGR to RGB
        predictor.set_image(rgb_image)

        masks = {}
        for i, box in enumerate(boxes):
            # Predict mask from box
            mask, score, _ = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=box,
                multimask_output=False,
            )
            masks[i] = mask[0]  # Take first mask

        return masks

    def _track_masks(
        self, rgb_image: np.ndarray, prev_masks: Dict[int, np.ndarray]
    ) -> Dict[int, np.ndarray]:
        """
        Track masks across frames using DEVA.

        This is a simplified implementation - full DEVA integration
        would require more setup.
        """
        # Placeholder: for now, just re-segment
        # Full DEVA would propagate masks with consistent IDs
        self._ensure_models_loaded()

        if self._grounding_dino is None or self._sam is None:
            return prev_masks

        boxes, labels, scores = self._detect_objects(rgb_image)
        if len(boxes) == 0:
            return {}

        return self._segment_from_boxes(rgb_image, boxes)

    def _masked_depth_to_pc(
        self,
        depth_image: np.ndarray,
        masks: Dict[int, np.ndarray],
        intrinsics: dict,
    ) -> np.ndarray:
        """
        Convert depth to point cloud, keeping only masked pixels.

        Args:
            depth_image: (H, W) uint16 depth in mm
            masks: dict of object_id -> (H, W) binary mask
            intrinsics: camera intrinsics

        Returns:
            pc: (N, 3) point cloud
        """
        from real_world.utils.point_cloud import depth_to_pointcloud

        # Combine all masks
        combined_mask = np.zeros(depth_image.shape, dtype=bool)
        for mask in masks.values():
            combined_mask |= mask

        # Convert to point cloud with mask
        pc = depth_to_pointcloud(
            depth_image,
            intrinsics,
            depth_scale=1000.0,
            filter_mask=combined_mask,
        )

        return pc

    def reset_tracking(self):
        """Reset tracking state for new episode."""
        self._prev_masks = None
        self._tracking_initialized = False
