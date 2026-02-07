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
IBR (Image Background Removal) observation processor using YOLO segmentation.

Supports Ultralytics segmentation models: YOLOv8 (yolov8n-seg.pt, yolov8s-seg.pt)
and YOLO26 2026 (yolo26n-seg.pt, yolo26s-seg.pt 등). 동일 predict API 사용.
Produces per-pixel binary masks (object=255, background=0) and optionally IBR images.

Requires: pip install ultralytics
"""

from dataclasses import dataclass
from typing import Any
import threading

import numpy as np

from .pipeline import ObservationProcessorStep, ProcessorStepRegistry
from .core import RobotObservation


def _is_image_array(value: Any) -> bool:
    """True if value is a numpy image (H, W, 3)."""
    if not isinstance(value, np.ndarray) or value.ndim != 3 or value.shape[2] != 3:
        return False
    return True


@ProcessorStepRegistry.register("yolo_ibr")
@dataclass
class YOLOIBRObservationProcessorStep(ObservationProcessorStep):
    """
    IBR step: YOLOv8 segmentation → binary mask (object=255, background=0).
    Optionally IBR image (original with background removed).
    Adds `<camera>_mask` and, unless mask_only, `<camera>_ibr`.

    - mask_only=True: 마스킹 화면만 출력해 부하·전송량 감소.
    - every_n_frames>1: N프레임마다만 YOLO 실행, 나머지는 이전 마스크 재사용해 부하 감소.
    - confidence를 낮추면 더 많은 객체 탐지(대신 오탐 증가 가능).
    - iou를 낮추면 겹친 객체를 더 많이 남김.
    - 2026 최신: yolo26n-seg.pt, yolo26s-seg.pt (인식·속도 개선). 구: yolov8*-seg.pt.

    Requires: pip install ultralytics
    """

    # yolov8n-seg.pt(구) / yolo26n-seg.pt·yolo26s-seg.pt(2026 최신, pip install -U ultralytics)
    model: str = "yolov8n-seg.pt"
    # 낮출수록 더 많은 객체 탐지. 마스크에 안 보이면 0.05~0.08 로.
    confidence: float = 0.1
    # NMS IoU 임계값. 낮출수록 겹친 객체를 더 많이 남김 (기본 0.5, 기본 YOLO는 0.7).
    iou: float = 0.5
    # 이미지당 최대 탐지 개수. 늘리면 복잡한 장면에서 더 많이 인식.
    max_det: int = 300
    background_color: tuple[int, int, int] = (0, 0, 0)
    # True면 _mask만 추가하고 _ibr 미생성·미전송 → 부하 감소
    mask_only: bool = False
    # 1이면 매 프레임 YOLO 실행. 2 이상이면 N프레임마다만 실행하고 나머지는 이전 결과 재사용.
    every_n_frames: int = 1

    def __post_init__(self) -> None:
        self._yolo = None
        self._cache: dict[str, tuple[np.ndarray, np.ndarray | None]] = {}
        self._call_count: dict[str, int] = {}
        self._cache_lock = threading.Lock()

    def _get_model(self):
        if self._yolo is None:
            try:
                from ultralytics import YOLO
            except ImportError as e:
                raise ImportError(
                    "YOLOIBRObservationProcessorStep requires ultralytics. "
                    "Install with: pip install ultralytics"
                ) from e
            self._yolo = YOLO(self.model)
        return self._yolo

    def _run_segmentation(
        self, img: np.ndarray, *, compute_ibr: bool = True
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """
        Run YOLOv8 segmentation. Returns (binary_mask, ibr_image or None).
        compute_ibr=False면 IBR 이미지 생성 생략(부하 감소).
        """
        out = np.asarray(img, dtype=np.uint8).copy(order="C")
        h, w = out.shape[0], out.shape[1]

        model = self._get_model()
        results = model.predict(
            out,
            conf=self.confidence,
            iou=self.iou,
            max_det=self.max_det,
            verbose=False,
            stream=False,
        )

        mask_combined = np.zeros((h, w), dtype=np.uint8)
        ibr: np.ndarray | None = None
        if not results or results[0].masks is None or len(results[0].masks) == 0:
            if compute_ibr:
                ibr = out.copy()
                ibr[:, :] = self.background_color
            return mask_combined, ibr

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
                    y_idx = (np.arange(h) * m.shape[0] / h).astype(np.int32).clip(0, m.shape[0] - 1)
                    x_idx = (np.arange(w) * m.shape[1] / w).astype(np.int32).clip(0, m.shape[1] - 1)
                    m = m[np.ix_(y_idx, x_idx)]
            mask_combined = np.maximum(mask_combined, (m > 0.5).astype(np.uint8) * 255)

        if compute_ibr:
            ibr = out.copy()
            bg = np.array(self.background_color, dtype=np.uint8)
            ibr[mask_combined == 0] = bg

        return mask_combined, ibr

    def observation(self, observation: RobotObservation) -> RobotObservation:
        processed = observation.copy()
        compute_ibr = not self.mask_only

        for key, value in list(processed.items()):
            if not _is_image_array(value) or ".pos" in key:
                continue

            run_yolo = True
            with self._cache_lock:
                n = self.every_n_frames
                cnt = self._call_count.get(key, 0)
                self._call_count[key] = (cnt + 1) % max(n, 1)
                if n > 1 and cnt != 0:
                    run_yolo = False
                    if key in self._cache:
                        mask, ibr_img = self._cache[key]
                        processed[f"{key}_mask"] = np.expand_dims(mask.copy(), axis=-1)
                        if not self.mask_only and ibr_img is not None:
                            processed[f"{key}_ibr"] = np.ascontiguousarray(ibr_img.copy(), dtype=np.uint8)
                        continue

            mask, ibr_img = self._run_segmentation(value, compute_ibr=compute_ibr)
            with self._cache_lock:
                self._cache[key] = (mask.copy(), ibr_img.copy() if ibr_img is not None else None)

            processed[f"{key}_mask"] = np.expand_dims(mask, axis=-1)
            if not self.mask_only and ibr_img is not None:
                processed[f"{key}_ibr"] = np.ascontiguousarray(ibr_img, dtype=np.uint8)
        return processed

    def transform_features(
        self, features: dict[Any, dict[str, Any]]
    ) -> dict[Any, dict[str, Any]]:
        # IBR outputs are for visualization only; no new dataset features.
        return features
