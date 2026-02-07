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
YOLO26 segmentation wrapper for IBR (Image Background Removal).

Uses Ultralytics YOLO segmentation (YOLOv8 / YOLO26-seg). Lazy model loading,
auto device selection, and thread-safe inference.
"""

import logging
import threading
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class YOLO26Segmenter:
    """
    Thread-safe YOLO segmentation wrapper. Returns a single merged object mask
    (uint8 0 or 255) per image.
    """

    def __init__(self, model_path: str, device: str = "auto") -> None:
        self.model_path = model_path
        self.device = device
        self._model = None
        self._lock = threading.Lock()

    def _get_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
        return "cpu"

    def _load_model(self) -> object:
        path = Path(self.model_path)
        if not path.exists():
            raise FileNotFoundError(f"YOLO model not found: {self.model_path}")
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError(
                "YOLO26Segmenter requires ultralytics. Install with: pip install ultralytics"
            ) from e
        device = self._get_device()
        model = YOLO(str(path))
        logger.info("YOLO segmentation model loaded from %s (device=%s)", self.model_path, device)
        return model

    def warmup(self) -> None:
        """Load model and run a dummy inference for faster first real call."""
        with self._lock:
            if self._model is None:
                self._model = self._load_model()
        # Dummy run to warm up GPU/engine (predict_mask acquires lock internally)
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        self.predict_mask(dummy)

    def predict_mask(self, image_np: np.ndarray) -> np.ndarray:
        """
        Run segmentation and return a single-channel binary mask.

        Args:
            image_np: HWC uint8 numpy array (BGR or RGB).

        Returns:
            mask: uint8 array (H, W), 255 = object, 0 = background.
        """
        with self._lock:
            if self._model is None:
                self._model = self._load_model()

        out = np.asarray(image_np, dtype=np.uint8).copy(order="C")
        h, w = out.shape[0], out.shape[1]

        results = self._model.predict(
            out,
            conf=0.1,
            iou=0.5,
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
