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
Observation processor that runs YOLO detection and draws bounding boxes on
camera images (for verifying object detection). Does not mask; only draws boxes.

Requires: pip install ultralytics
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from .pipeline import ObservationProcessorStep, ProcessorStepRegistry
from .core import RobotObservation


def _is_image_array(value: Any) -> bool:
    """True if value is a numpy image (H, W, 3)."""
    if not isinstance(value, np.ndarray) or value.ndim != 3 or value.shape[2] != 3:
        return False
    return True


@ProcessorStepRegistry.register("yolo_bbox")
@dataclass
class YOLOBboxObservationProcessorStep(ObservationProcessorStep):
    """
    Draws YOLO detection bounding boxes on observation images (for debugging/verification).
    Does not mask; only overlays rectangles on the image.

    Requires: pip install ultralytics
    """

    model: str = "yolov8n.pt"
    confidence: float = 0.25
    # (R, G, B) for box color; default bright red for visibility
    box_color: tuple[int, int, int] = (255, 0, 0)
    line_thickness: int = 3

    def __post_init__(self) -> None:
        self._yolo = None

    def _get_model(self):
        if self._yolo is None:
            try:
                from ultralytics import YOLO
            except ImportError as e:
                raise ImportError(
                    "YOLOBboxObservationProcessorStep requires ultralytics. "
                    "Install with: pip install ultralytics"
                ) from e
            self._yolo = YOLO(self.model)
        return self._yolo

    def _draw_boxes(self, img: np.ndarray) -> np.ndarray:
        """Run YOLO detection and draw bounding boxes on a copy of the image."""
        out = np.asarray(img, dtype=np.uint8).copy(order="C")
        h, w = out.shape[0], out.shape[1]

        model = self._get_model()
        results = model.predict(
            out,
            conf=self.confidence,
            verbose=False,
            stream=False,
        )
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return out

        result = results[0]
        xyxy = result.boxes.xyxy
        if hasattr(xyxy, "cpu"):
            xyxy = xyxy.cpu().numpy()
        else:
            xyxy = np.asarray(xyxy)

        try:
            import cv2
        except ImportError:
            # fallback: draw by setting pixel borders (slower)
            for i in range(len(xyxy)):
                x1, y1, x2, y2 = (
                    max(0, int(xyxy[i, 0])),
                    max(0, int(xyxy[i, 1])),
                    min(w, int(xyxy[i, 2])),
                    min(h, int(xyxy[i, 3])),
                )
                t = self.line_thickness
                out[y1 : y1 + t, x1:x2] = self.box_color
                out[y2 - t : y2, x1:x2] = self.box_color
                out[y1:y2, x1 : x1 + t] = self.box_color
                out[y1:y2, x2 - t : x2] = self.box_color
            return out

        # Image is typically RGB; pass (R,G,B) so channels match
        color = tuple(int(c) for c in self.box_color)
        for i in range(len(xyxy)):
            x1, y1, x2, y2 = (
                max(0, int(xyxy[i, 0])),
                max(0, int(xyxy[i, 1])),
                min(w, int(xyxy[i, 2])),
                min(h, int(xyxy[i, 3])),
            )
            cv2.rectangle(out, (x1, y1), (x2, y2), color, self.line_thickness)
        return np.ascontiguousarray(out, dtype=np.uint8)

    def observation(self, observation: RobotObservation) -> RobotObservation:
        processed = observation.copy()
        for key, value in list(processed.items()):
            if not _is_image_array(value) or ".pos" in key:
                continue
            processed[key] = self._draw_boxes(value)
        return processed

    def transform_features(
        self, features: dict[Any, dict[str, Any]]
    ) -> dict[Any, dict[str, Any]]:
        return features
