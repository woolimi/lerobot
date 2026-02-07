# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Singleton YOLO26 segmentation service. Load model once globally; thread-safe inference; config-driven.
"""

import logging
import os
import threading
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from lerobot.common.vision_config import VisionConfig

logger = logging.getLogger(__name__)

_instance: "YOLO26Service | None" = None
_lock = threading.Lock()


class YOLO26Service:
    """
    Singleton YOLO26 segmentation service.
    Load model once from VisionConfig; thread-safe segment(image_np) -> mask.
    """

    def __init__(self, vision_config: "VisionConfig") -> None:
        self._vision_config = vision_config
        self._model = None
        self._infer_lock = threading.Lock()

    def _load_model(self) -> object:
        path = Path(self._vision_config.segmentation_model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"YOLO model not found: {self._vision_config.segmentation_model_path}"
            )
        try:
            os.environ.setdefault("YOLO_VERBOSE", "false")
            logging.getLogger("ultralytics").setLevel(logging.WARNING)
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError(
                "YOLO26Service requires ultralytics. Install with: pip install ultralytics"
            ) from e
        model = YOLO(str(path))
        logger.info(
            "YOLO26Service loaded from %s (conf=%.2f, iou=%.2f)",
            path,
            self._vision_config.segmentation_confidence,
            self._vision_config.segmentation_iou_threshold,
        )
        return model

    def segment(self, image_np: np.ndarray) -> np.ndarray:
        """
        Run segmentation; returns binary mask (H, W) uint8, 255=object, 0=background.
        On failure (e.g. model missing), caller should fallback to raw image.
        """
        with self._infer_lock:
            if self._model is None:
                self._model = self._load_model()

        out = np.asarray(image_np, dtype=np.uint8).copy(order="C")
        h, w = out.shape[0], out.shape[1]

        conf = self._vision_config.segmentation_confidence
        iou = self._vision_config.segmentation_iou_threshold

        results = self._model.predict(
            out,
            conf=conf,
            iou=iou,
            max_det=300,
            verbose=False,
            stream=False,
        )

        mask_combined = np.zeros((h, w), dtype=np.uint8)
        if not results or results[0].masks is None or len(results[0].masks) == 0:
            return mask_combined

        result = results[0]
        masks = result.masks
        if hasattr(masks.data, "cpu"):
            masks_np = masks.data.cpu().numpy()
        else:
            masks_np = np.asarray(masks.data)

        for i in range(masks_np.shape[0]):
            m = masks_np[i]
            if m.shape[0] != h or m.shape[1] != w:
                try:
                    import cv2
                    m = cv2.resize(
                        m.astype(np.uint8),
                        (w, h),
                        interpolation=cv2.INTER_NEAREST,
                    )
                except ImportError:
                    y_idx = (
                        np.arange(h) * m.shape[0] / h
                    ).astype(np.int32).clip(0, m.shape[0] - 1)
                    x_idx = (
                        np.arange(w) * m.shape[1] / w
                    ).astype(np.int32).clip(0, m.shape[1] - 1)
                    m = m[np.ix_(y_idx, x_idx)]
            mask_combined = np.maximum(
                mask_combined, (m > 0.5).astype(np.uint8) * 255
            )

        return mask_combined


def get_yolo26_service(vision_config: "VisionConfig") -> YOLO26Service:
    """Get or create the singleton YOLO26Service for the given config."""
    global _instance
    with _lock:
        if _instance is None:
            _instance = YOLO26Service(vision_config)
        return _instance
