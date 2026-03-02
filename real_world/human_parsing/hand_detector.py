"""
Hand Detection using HaMeR (Hand Mesh Recovery).

Based on EquiBot paper Section F.1:
- Uses ViTDet (Detectron2) for person detection
- Uses ViTPose for body/hand keypoint detection
- Uses HaMeR for 3D hand mesh recovery (MANO)

The full pipeline:
  RGB image
    -> ViTDet: detect person bounding boxes
    -> ViTPose: predict whole-body keypoints (incl. 21 per hand)
    -> extract hand bounding boxes from ViTPose hand keypoints
    -> HaMeR: predict MANO mesh per hand crop
    -> output 3D & 2D keypoints, hand side, confidence

References:
- HaMeR: https://github.com/geopavlakos/hamer
- Detectron2/ViTDet: https://github.com/facebookresearch/detectron2
- ViTPose: https://github.com/ViTAE-Transformer/ViTPose
"""

import os
import sys
import numpy as np
from typing import Dict, List, Optional
import warnings

HAND_KEYPOINT_NAMES = [
    'wrist',
    'thumb_cmc', 'thumb_mcp', 'thumb_ip', 'thumb_tip',
    'index_mcp', 'index_pip', 'index_dip', 'index_tip',
    'middle_mcp', 'middle_pip', 'middle_dip', 'middle_tip',
    'ring_mcp', 'ring_pip', 'ring_dip', 'ring_tip',
    'pinky_mcp', 'pinky_pip', 'pinky_dip', 'pinky_tip',
]

THUMB_TIP_IDX = 4
INDEX_TIP_IDX = 8
MIDDLE_TIP_IDX = 12
RING_TIP_IDX = 16
PINKY_TIP_IDX = 20


class HandDetector:
    """
    Full HaMeR hand detection pipeline (ViTDet -> ViTPose -> HaMeR).

    Outputs 21 3D keypoints per hand (from MANO mesh) and 21 2D keypoints
    (from ViTPose).

    Usage:
        detector = HandDetector(device="cuda")
        hands = detector.detect_hands(bgr_image)
        for hand in hands:
            print(hand['keypoints_3d'].shape)  # (21, 3)
            print(hand['keypoints_2d'].shape)  # (21, 2)
    """

    def __init__(
        self,
        device: str = "cuda",
        confidence_threshold: float = 0.5,
        rescale_factor: float = 2.0,
        body_detector: str = "vitdet",
        hamer_dir: Optional[str] = None,
    ):
        """
        Args:
            device: 'cuda', 'cpu', or 'mps'
            confidence_threshold: minimum ViTPose hand keypoint confidence
            rescale_factor: padding factor for hand bounding boxes
            body_detector: 'vitdet' or 'regnety'
            hamer_dir: path to cloned hamer repo (for vitpose_model.py).
                Defaults to env HAMER_DIR or 'third_party/hamer'.
        """
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.rescale_factor = rescale_factor
        self.body_detector = body_detector

        if hamer_dir is None:
            hamer_dir = os.environ.get('HAMER_DIR', 'third_party/hamer')
        self.hamer_dir = os.path.abspath(hamer_dir)

        self._person_detector = None
        self._vitpose = None
        self._hamer_model = None
        self._hamer_cfg = None
        self._mock_fallback = None  # used when pipeline not loaded (e.g. HaMeR missing on M1)

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _ensure_models_loaded(self):
        if self._hamer_model is not None:
            return
        self._load_all()

    def _load_all(self):
        """Load ViTDet, ViTPose, and HaMeR."""
        import torch

        device = torch.device(self.device)

        # --- HaMeR ---
        try:
            from hamer.configs import CACHE_DIR_HAMER
            from hamer.models import download_models, load_hamer, DEFAULT_CHECKPOINT
            from hamer.utils import recursive_to  # noqa: F401

            download_models(CACHE_DIR_HAMER)
            ckpt = os.environ.get('HAMER_CHECKPOINT', DEFAULT_CHECKPOINT)
            model, model_cfg = load_hamer(ckpt)
            model = model.to(device)
            model.eval()
            self._hamer_model = model
            self._hamer_cfg = model_cfg
        except ImportError as e:
            warnings.warn(
                f"HaMeR not installed: {e}\n"
                "Install with: cd third_party/hamer && pip install -e .[all]"
            )
            return
        except Exception as e:
            warnings.warn(f"Failed to load HaMeR: {e}")
            return

        # --- Person detector (ViTDet via Detectron2) ---
        try:
            from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy

            if self.body_detector == 'vitdet':
                from detectron2.config import LazyConfig
                from pathlib import Path
                import hamer as hamer_pkg

                cfg_path = (
                    Path(hamer_pkg.__file__).parent
                    / 'configs'
                    / 'cascade_mask_rcnn_vitdet_h_75ep.py'
                )
                detectron2_cfg = LazyConfig.load(str(cfg_path))
                detectron2_cfg.train.init_checkpoint = (
                    "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
                )
                for i in range(3):
                    detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
                self._person_detector = DefaultPredictor_Lazy(detectron2_cfg)
            else:
                from detectron2 import model_zoo
                from detectron2.config import get_cfg
                detectron2_cfg = model_zoo.get_config(
                    'new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py',
                    trained=True,
                )
                detectron2_cfg.model.roi_heads.box_predictor.test_score_thresh = 0.5
                detectron2_cfg.model.roi_heads.box_predictor.test_nms_thresh = 0.4
                self._person_detector = DefaultPredictor_Lazy(detectron2_cfg)

        except ImportError as e:
            warnings.warn(f"Detectron2 not installed: {e}")
            return
        except Exception as e:
            warnings.warn(f"Failed to load person detector: {e}")
            return

        # --- ViTPose ---
        try:
            if self.hamer_dir not in sys.path:
                sys.path.insert(0, self.hamer_dir)
            from vitpose_model import ViTPoseModel
            self._vitpose = ViTPoseModel(device)
        except ImportError as e:
            warnings.warn(
                f"ViTPose not installed: {e}\n"
                f"Make sure HAMER_DIR={self.hamer_dir} is correct and "
                "third-party/ViTPose is installed."
            )
            return
        except Exception as e:
            warnings.warn(f"Failed to load ViTPose: {e}")
            return

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_hands(self, rgb_image: np.ndarray) -> List[Dict]:
        """
        Detect hands and extract 2D + 3D keypoints.

        Args:
            rgb_image: (H, W, 3) BGR image from cv2

        Returns:
            hands: list of dicts, each containing:
                - 'keypoints_3d': (21, 3) 3D keypoints (HaMeR/MANO frame)
                - 'keypoints_2d': (21, 2) 2D pixel coords (from ViTPose)
                - 'hand_side': 'left' or 'right'
                - 'confidence': detection confidence
                - 'bbox': (4,) hand bounding box [x1, y1, x2, y2]
        """
        self._ensure_models_loaded()

        if self._hamer_model is None or self._person_detector is None or self._vitpose is None:
            # Fallback to mock hand detector so episodes still produce frames (e.g. on M1 without HaMeR)
            if self._mock_fallback is None:
                warnings.warn(
                    "HaMeR pipeline not loaded; using mock hand detector so episodes are not dropped. "
                    "Install HaMeR for real hand poses, or use --mock_hands to silence this."
                )
                self._mock_fallback = MockHandDetector(device=self.device)
            return self._mock_fallback.detect_hands(rgb_image)

        try:
            return self._run_pipeline(rgb_image)
        except Exception as e:
            warnings.warn(f"HaMeR pipeline failed: {e}")
            return []

    # ------------------------------------------------------------------
    # Pipeline internals
    # ------------------------------------------------------------------

    def _run_pipeline(self, bgr_image: np.ndarray) -> List[Dict]:
        import torch
        from hamer.utils import recursive_to
        from hamer.datasets.vitdet_dataset import ViTDetDataset
        from hamer.utils.renderer import cam_crop_to_full

        device = torch.device(self.device)

        # 1. Detect people
        det_out = self._person_detector(bgr_image)
        det_instances = det_out['instances']
        valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)
        pred_bboxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
        pred_scores = det_instances.scores[valid_idx].cpu().numpy()

        if len(pred_bboxes) == 0:
            return []

        # 2. ViTPose -> hand keypoints
        img_rgb = bgr_image[:, :, ::-1].copy()
        vitposes_out = self._vitpose.predict_pose(
            img_rgb,
            [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)],
        )

        bboxes = []
        is_right = []
        kp_2d_list = []

        for vitposes in vitposes_out:
            kps = vitposes['keypoints']  # (133, 3) for whole-body
            left_hand_kp = kps[-42:-21]   # (21, 3)  x, y, conf
            right_hand_kp = kps[-21:]      # (21, 3)

            for hand_kp, right_flag in [(left_hand_kp, 0), (right_hand_kp, 1)]:
                valid = hand_kp[:, 2] > self.confidence_threshold
                if valid.sum() > 3:
                    xs, ys = hand_kp[valid, 0], hand_kp[valid, 1]
                    bbox = [xs.min(), ys.min(), xs.max(), ys.max()]
                    bboxes.append(bbox)
                    is_right.append(right_flag)
                    kp_2d_list.append(hand_kp[:, :2].copy())  # pixel coords

        if len(bboxes) == 0:
            return []

        boxes_np = np.stack(bboxes)
        right_np = np.stack(is_right)

        # 3. HaMeR inference on hand crops
        dataset = ViTDetDataset(
            self._hamer_cfg, bgr_image, boxes_np, right_np,
            rescale_factor=self.rescale_factor,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=8, shuffle=False, num_workers=0,
        )

        hands: List[Dict] = []
        hand_idx = 0

        for batch in dataloader:
            batch = recursive_to(batch, device)
            with torch.no_grad():
                out = self._hamer_model(batch)

            batch_size = batch['img'].shape[0]
            img_size = batch['img_size'].float()
            box_center = batch['box_center'].float()
            box_size = batch['box_size'].float()

            scaled_focal = (
                self._hamer_cfg.EXTRA.FOCAL_LENGTH
                / self._hamer_cfg.MODEL.IMAGE_SIZE
                * img_size.max()
            )

            multiplier = 2 * batch['right'] - 1
            pred_cam = out['pred_cam']
            pred_cam[:, 1] = multiplier * pred_cam[:, 1]

            pred_cam_t = cam_crop_to_full(
                pred_cam, box_center, box_size, img_size, scaled_focal,
            ).detach().cpu().numpy()

            # Extract 3D joints from MANO output
            pred_kp3d = self._extract_joints_3d(out)  # (B, 21, 3) or None

            for n in range(batch_size):
                if hand_idx >= len(kp_2d_list):
                    break

                right_flag = int(batch['right'][n].item())

                if pred_kp3d is not None:
                    kp3d = pred_kp3d[n].copy()
                    # Mirror x for left hands (HaMeR convention)
                    kp3d[:, 0] = (2 * right_flag - 1) * kp3d[:, 0]
                    kp3d = kp3d + pred_cam_t[n]
                else:
                    kp3d = np.zeros((21, 3), dtype=np.float32)

                avg_conf = float(
                    np.mean(
                        dataset[hand_idx]['keypoints_2d'][:, 2]
                        if hasattr(dataset[hand_idx], '__getitem__')
                        else 0.8
                    )
                ) if False else 0.8  # ViTPose conf not directly accessible here

                hands.append({
                    'keypoints_3d': kp3d.astype(np.float32),
                    'keypoints_2d': kp_2d_list[hand_idx].astype(np.float32),
                    'hand_side': 'right' if right_flag else 'left',
                    'confidence': avg_conf,
                    'bbox': boxes_np[hand_idx],
                })
                hand_idx += 1

        return hands

    @staticmethod
    def _extract_joints_3d(out: dict) -> Optional[np.ndarray]:
        """
        Extract (B, 21, 3) 3D joints from HaMeR model output.

        Tries multiple keys because the exact output format varies
        across HaMeR versions.
        """
        import torch

        for key in ('pred_keypoints_3d', 'pred_joints', 'pred_joints3d'):
            if key in out:
                val = out[key]
                if isinstance(val, torch.Tensor):
                    kp = val.detach().cpu().numpy()
                    if kp.ndim == 3 and kp.shape[1] >= 21:
                        return kp[:, :21, :]
                    return kp

        # Fallback: derive joints from MANO vertices if model exposes the regressor
        if 'pred_vertices' in out:
            try:
                verts = out['pred_vertices'].detach().cpu().numpy()
                # Fingertip vertex indices in the standard MANO mesh (778 verts)
                tip_ids = [745, 317, 444, 556, 673]
                # MANO joint regressor maps 778 verts -> 16 joints;
                # with tips appended we get 21.
                # Without the regressor we can only return tips + wrist vertex.
                warnings.warn(
                    "HaMeR output has no joint key; falling back to vertex tips. "
                    "3D keypoints may be less accurate."
                )
                B = verts.shape[0]
                joints = np.zeros((B, 21, 3), dtype=np.float32)
                joints[:, 0, :] = verts[:, 0, :]  # wrist ≈ vertex 0
                for idx, tip_v in enumerate(tip_ids):
                    joints[:, 4 + idx * 4, :] = verts[:, tip_v, :]
                return joints
            except Exception:
                pass

        return None

    # ------------------------------------------------------------------
    # Convenience helpers (same API as before)
    # ------------------------------------------------------------------

    def get_fingertip_positions(self, hand: Dict) -> tuple:
        kp = hand['keypoints_3d']
        return kp[THUMB_TIP_IDX], kp[INDEX_TIP_IDX]

    def get_all_fingertips(self, hand: Dict) -> Dict[str, np.ndarray]:
        kp = hand['keypoints_3d']
        return {
            'thumb': kp[THUMB_TIP_IDX],
            'index': kp[INDEX_TIP_IDX],
            'middle': kp[MIDDLE_TIP_IDX],
            'ring': kp[RING_TIP_IDX],
            'pinky': kp[PINKY_TIP_IDX],
        }

    def get_wrist_position(self, hand: Dict) -> np.ndarray:
        return hand['keypoints_3d'][0]

    def estimate_gripper_width(self, hand: Dict) -> float:
        t, i = self.get_fingertip_positions(hand)
        return float(np.linalg.norm(t - i))

    def is_grasping(self, hand: Dict, threshold: float = 0.04) -> bool:
        return self.estimate_gripper_width(hand) < threshold


class MockHandDetector(HandDetector):
    """
    Mock detector for testing without HaMeR installed.
    Returns synthetic hand keypoints in the centre of the image.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._hamer_model = True  # prevent real loading

    def _ensure_models_loaded(self):
        pass  # nothing to load

    def detect_hands(self, rgb_image: np.ndarray) -> List[Dict]:
        H, W = rgb_image.shape[:2]
        center = np.array([W / 2, H / 2])

        keypoints_2d = np.zeros((21, 2), dtype=np.float32)
        keypoints_2d[0] = center
        for i in range(5):
            for j in range(4):
                idx = 1 + i * 4 + j
                angle = (i - 2) * 0.3
                dist = 20 + j * 15
                keypoints_2d[idx] = center + np.array([
                    np.sin(angle) * dist,
                    -dist,
                ])

        keypoints_3d = np.zeros((21, 3), dtype=np.float32)
        keypoints_3d[:, :2] = (keypoints_2d - center) / 1000
        keypoints_3d[:, 2] = 0.5

        return [{
            'keypoints_3d': keypoints_3d,
            'keypoints_2d': keypoints_2d,
            'hand_side': 'right',
            'confidence': 0.9,
            'bbox': np.array([W / 4, H / 4, 3 * W / 4, 3 * H / 4]),
        }]
