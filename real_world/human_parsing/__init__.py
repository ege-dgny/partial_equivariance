"""
Human demo parsing infrastructure for PEFM.

Based on EquiBot paper Section F.1:
1. Object Detection & Tracking: Grounded SAM + DEVA
2. Hand Detection: HaMeR
3. Alignment Module: Hand-to-pointcloud coordinate alignment
"""

from .object_tracker import ObjectTracker
from .hand_detector import HandDetector
from .alignment import HandPointCloudAligner
from .demo_parser import HumanDemoParser

__all__ = [
    "ObjectTracker",
    "HandDetector",
    "HandPointCloudAligner",
    "HumanDemoParser",
]
