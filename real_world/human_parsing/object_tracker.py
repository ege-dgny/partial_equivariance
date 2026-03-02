"""
Object Detection and Tracking using Grounded SAM + DEVA.

Based on EquiBot paper Section F.1:
- Uses Grounding DINO for open-vocabulary object detection
- Uses SAM (Segment Anything Model) for instance segmentation
- Uses DEVA for video object tracking with consistent IDs (optional)

When DEVA is not installed the tracker falls back to independent
per-frame detection via Grounded SAM.  This is noisier but functional.

References:
- GroundingDINO: https://github.com/IDEA-Research/GroundingDINO
- SAM: https://github.com/facebookresearch/segment-anything
- DEVA: https://github.com/hkchengrex/Tracking-Anything-with-DEVA
"""

import os
import numpy as np
from typing import Dict, List, Optional, Tuple
import warnings

# Repo root (Partial_Equivariance): real_world/human_parsing/object_tracker.py -> 3 levels up
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def _default_grounding_dino_paths():
    """Resolve config and checkpoint to third_party/ and weights/ when present."""
    config_in_third = os.path.join(
        _REPO_ROOT, "third_party", "GroundingDINO",
        "groundingdino", "config", "GroundingDINO_SwinT_OGC.py"
    )
    checkpoint_in_weights = os.path.join(_REPO_ROOT, "weights", "groundingdino_swint_ogc.pth")
    config = config_in_third if os.path.isfile(config_in_third) else None
    checkpoint = checkpoint_in_weights if os.path.isfile(checkpoint_in_weights) else None
    return config, checkpoint


def _load_grounding_dino(device: str = "cpu"):
    """Load Grounding DINO model onto *device*."""
    try:
        from groundingdino.util.inference import load_model, predict
        import groundingdino.datasets.transforms as T
        from groundingdino.util.box_ops import box_cxcywh_to_xyxy

        config_path = os.environ.get("GROUNDING_DINO_CONFIG")
        checkpoint_path = os.environ.get("GROUNDING_DINO_CHECKPOINT")
        if not config_path or not checkpoint_path:
            default_config, default_ckpt = _default_grounding_dino_paths()
            config_path = config_path or default_config or "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
            checkpoint_path = checkpoint_path or default_ckpt or "weights/groundingdino_swint_ogc.pth"
        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"Grounding DINO config not found: {config_path}")
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Grounding DINO checkpoint not found: {checkpoint_path}")

        model = load_model(config_path, checkpoint_path)
        model = model.to(device)

        return {
            'model': model,
            'predict': predict,
            'transforms': T,
            'box_cxcywh_to_xyxy': box_cxcywh_to_xyxy,
        }

    except ImportError as e:
        warnings.warn(
            f"Grounding DINO not installed: {e}\n"
            "Install with: pip install groundingdino-py\n"
            "Or clone: https://github.com/IDEA-Research/GroundingDINO"
        )
        return None


def _load_sam(device: str = "cpu"):
    """Load SAM model onto *device*."""
    try:
        from segment_anything import sam_model_registry, SamPredictor

        checkpoint_path = os.environ.get("SAM_CHECKPOINT")
        if not checkpoint_path:
            default_ckpt = os.path.join(_REPO_ROOT, "weights", "sam_vit_h_4b8939.pth")
            checkpoint_path = default_ckpt if os.path.isfile(default_ckpt) else "weights/sam_vit_h_4b8939.pth"
        model_type = os.environ.get('SAM_MODEL_TYPE', 'vit_h')

        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        sam.to(device)
        predictor = SamPredictor(sam)

        return {'predictor': predictor}

    except ImportError as e:
        warnings.warn(
            f"SAM not installed: {e}\n"
            "Install with: pip install git+https://github.com/facebookresearch/segment-anything.git\n"
        )
        return None


class ObjectTracker:
    """
    Grounded SAM + (optional) DEVA for object detection and tracking.

    Outputs segmented point cloud containing only objects of interest.

    Usage:
        tracker = ObjectTracker(text_prompt="cup. block. peg.")
        object_pc, masks = tracker.detect_and_segment(rgb, depth, intrinsics)

        # For video tracking (falls back to per-frame detection if DEVA is missing):
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
        self.text_prompt = text_prompt
        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

        self._grounding_dino = None
        self._sam = None

        self._prev_masks = None
        self._tracking_initialized = False

    def _ensure_models_loaded(self):
        if self._grounding_dino is None:
            self._grounding_dino = _load_grounding_dino(self.device)
        if self._sam is None:
            self._sam = _load_sam(self.device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_and_segment(
        self,
        rgb_image: np.ndarray,
        depth_image: np.ndarray,
        intrinsics: dict,
    ) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
        """
        Single-frame detection and segmentation.

        Args:
            rgb_image: (H, W, 3) BGR image from cv2
            depth_image: (H, W) uint16 depth in mm
            intrinsics: dict with 'fx', 'fy', 'ppx', 'ppy'

        Returns:
            object_pc: (N, 3) segmented point cloud
            masks: dict mapping object_id -> (H, W) binary mask
        """
        self._ensure_models_loaded()

        if self._grounding_dino is None or self._sam is None:
            warnings.warn("Models not loaded, returning empty point cloud")
            return np.zeros((0, 3), dtype=np.float32), {}

        boxes, labels, scores = self._detect_objects(rgb_image)

        if len(boxes) == 0:
            return np.zeros((0, 3), dtype=np.float32), {}

        masks = self._segment_from_boxes(rgb_image, boxes)
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
        Track objects across frames.

        Currently re-runs Grounded SAM per frame (no temporal DEVA).
        Object IDs may not be consistent across frames.
        """
        # TODO: integrate DEVA for temporally consistent mask propagation.
        #  - Initialize DEVA with masks from the first frame.
        #  - On subsequent frames, propagate masks and only re-detect
        #    every N frames to pick up new objects.
        return self.detect_and_segment(rgb_image, depth_image, intrinsics)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _detect_objects(
        self, rgb_image: np.ndarray
    ) -> Tuple[np.ndarray, List[str], np.ndarray]:
        """
        Detect objects using Grounding DINO.

        Returns:
            boxes: (N, 4) bounding boxes in **xyxy pixel** format
            labels: list of N label strings
            scores: (N,) confidence scores
        """
        import torch
        from PIL import Image

        rgb_for_pil = rgb_image[..., ::-1].copy()  # BGR -> RGB, contiguous
        pil_image = Image.fromarray(rgb_for_pil)

        transform = self._grounding_dino['transforms'].Compose([
            self._grounding_dino['transforms'].RandomResize([800], max_size=1333),
            self._grounding_dino['transforms'].ToTensor(),
            self._grounding_dino['transforms'].Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            ),
        ])
        image_transformed, _ = transform(pil_image, None)

        boxes, logits, phrases = self._grounding_dino['predict'](
            model=self._grounding_dino['model'],
            image=image_transformed,
            caption=self.text_prompt,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
        )

        if len(boxes) == 0:
            return np.empty((0, 4)), [], np.empty(0)

        # boxes from predict() are normalized cxcywh -> convert to xyxy
        boxes_xyxy = self._grounding_dino['box_cxcywh_to_xyxy'](boxes)

        H, W = rgb_image.shape[:2]
        boxes_xyxy = boxes_xyxy * torch.tensor(
            [W, H, W, H], device=boxes_xyxy.device, dtype=boxes_xyxy.dtype
        )
        boxes_np = boxes_xyxy.cpu().numpy()
        scores_np = logits.cpu().numpy()

        return boxes_np, phrases, scores_np

    def _segment_from_boxes(
        self, rgb_image: np.ndarray, boxes: np.ndarray
    ) -> Dict[int, np.ndarray]:
        """Generate segmentation masks from xyxy bounding boxes using SAM."""
        predictor = self._sam['predictor']

        rgb_for_sam = rgb_image[..., ::-1].copy()  # BGR -> RGB, contiguous
        predictor.set_image(rgb_for_sam)

        masks = {}
        for i, box in enumerate(boxes):
            mask, score, _ = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=box,
                multimask_output=False,
            )
            masks[i] = mask[0]

        return masks

    def _masked_depth_to_pc(
        self,
        depth_image: np.ndarray,
        masks: Dict[int, np.ndarray],
        intrinsics: dict,
    ) -> np.ndarray:
        from real_world.utils.point_cloud import depth_to_pointcloud

        combined_mask = np.zeros(depth_image.shape, dtype=bool)
        for mask in masks.values():
            combined_mask |= mask

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
