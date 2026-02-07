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
Dual-camera IBR processor: gripper -> IBR, top -> RAW.
"""

import logging
import threading
from typing import TYPE_CHECKING

import numpy as np

from lerobot.vision.camera_roles import CameraRole, get_camera_role
from lerobot.vision.yolo26_service import YOLO26Service, get_yolo26_service

if TYPE_CHECKING:
    from lerobot.common.vision_config import VisionConfig

logger = logging.getLogger(__name__)

BACKGROUND_COLOR = (0, 0, 0)
FALLBACK_WARN_EVERY_N_FRAMES = 30


class DualCameraIBRProcessor:
    """
    Process images per camera role: GRIPPER -> IBR (YOLO26 + mask), TOP -> RAW.
    Frame skip and fallback to raw on segmentation failure.
    """

    def __init__(
        self,
        vision_config: "VisionConfig",
        yolo_service: YOLO26Service | None = None,
    ) -> None:
        self._config = vision_config
        self._service = yolo_service or get_yolo26_service(vision_config)
        self._frame_count = 0
        self._last_mask: np.ndarray | None = None
        self._last_mask_lock = threading.Lock()
        self._fallback_warn_count = 0

    def _should_run_segmentation(self) -> bool:
        skip = max(1, self._config.segmentation_frame_skip)
        self._frame_count += 1
        return (self._frame_count - 1) % skip == 0

    def _apply_mask(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        out = np.asarray(image, dtype=np.uint8).copy()
        bg = np.array(BACKGROUND_COLOR, dtype=np.uint8)
        out[mask == 0] = bg
        return out

    def _apply_brightness_contrast_gamma(
        self,
        image: np.ndarray,
        brightness: float,
        contrast: float,
        gamma: float,
    ) -> np.ndarray:
        """Apply brightness (scale), contrast, and gamma to RGB image. No-op if all are 1.0."""
        if abs(brightness - 1.0) < 1e-6 and abs(contrast - 1.0) < 1e-6 and abs(gamma - 1.0) < 1e-6:
            return image
        out = np.asarray(image, dtype=np.float64).copy()
        if out.ndim != 3 or out.shape[2] != 3:
            return np.asarray(image, dtype=np.uint8)
        # Gamma (on [0,1]): >1 darkens overexposed areas. Avoid 0**gamma when gamma<0 (divide-by-zero).
        if abs(gamma - 1.0) >= 1e-6:
            out_norm = np.clip(out / 255.0, 1e-7, 1.0)
            out = np.power(out_norm, gamma) * 255.0
        # Contrast around midpoint 128
        if abs(contrast - 1.0) >= 1e-6:
            out = (out - 128.0) * contrast + 128.0
        # Brightness scale
        if abs(brightness - 1.0) >= 1e-6:
            out = out * brightness
        out = np.clip(out, 0, 255).astype(np.uint8)
        return out

    def _stabilize_brightness(self, image: np.ndarray, camera_role: CameraRole) -> np.ndarray:
        """Apply optional brightness/contrast/gamma, then CLAHE on luminance (LAB L) if enabled."""
        try:
            import cv2
        except ImportError:
            return image
        if camera_role == CameraRole.TOP:
            brightness = self._config.top_brightness
            contrast = self._config.top_contrast
            gamma = self._config.top_gamma
            enabled = self._config.top_brightness_stabilize
            clip = self._config.top_brightness_clip_limit
            tile = self._config.top_brightness_tile_size
        else:
            brightness = self._config.gripper_brightness
            contrast = self._config.gripper_contrast
            gamma = self._config.gripper_gamma
            enabled = self._config.gripper_brightness_stabilize
            clip = self._config.gripper_brightness_clip_limit
            tile = self._config.gripper_brightness_tile_size

        out = np.asarray(image, dtype=np.uint8).copy()
        if out.ndim != 3 or out.shape[2] != 3:
            return out

        # 1) Manual brightness/contrast/gamma (e.g. darken overexposed top view)
        out = self._apply_brightness_contrast_gamma(out, brightness, contrast, gamma)

        # 2) CLAHE on luminance if enabled
        if not enabled:
            return out
        clip = max(0.0, min(10.0, float(clip)))
        tile = max(2, min(32, int(tile)))
        lab = cv2.cvtColor(out, cv2.COLOR_RGB2LAB)
        l_channel, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
        l_eq = clahe.apply(l_channel)
        lab_eq = cv2.merge([l_eq, a, b])
        out = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)
        return out

    def process(self, image: np.ndarray, camera_role: CameraRole) -> np.ndarray:
        """
        Return IBR image for GRIPPER (if gripper_use_ibr), else RAW.
        For TOP or when IBR disabled for role, return image unchanged.
        """
        use_ibr = (
            (camera_role == CameraRole.GRIPPER and self._config.gripper_use_ibr)
            or (camera_role == CameraRole.TOP and self._config.top_use_ibr)
        )
        if not use_ibr:
            out = np.asarray(image, dtype=np.uint8).copy()
            out = self._stabilize_brightness(out, camera_role)
            return out

        run_yolo = self._should_run_segmentation()
        mask = None

        if run_yolo:
            try:
                mask = self._service.segment(image)
                with self._last_mask_lock:
                    self._last_mask = mask.copy()
            except Exception as e:
                self._fallback_warn_count += 1
                if self._fallback_warn_count % FALLBACK_WARN_EVERY_N_FRAMES == 1:
                    logger.warning(
                        "Segmentation failed (%s), using raw image (warning %d every %d frames)",
                        e,
                        self._fallback_warn_count,
                        FALLBACK_WARN_EVERY_N_FRAMES,
                    )
                return np.asarray(image, dtype=np.uint8).copy()

        if mask is None:
            with self._last_mask_lock:
                mask = self._last_mask

        if mask is None:
            return np.asarray(image, dtype=np.uint8).copy()

        out = self._apply_mask(image, mask)
        out = self._stabilize_brightness(out, camera_role)
        return out

    def process_with_mask(
        self, image: np.ndarray, camera_role: CameraRole
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """
        Same as process() but also return mask for gripper when IBR is used (for optional storage).
        Returns (processed_image, mask or None).
        """
        use_ibr = (
            (camera_role == CameraRole.GRIPPER and self._config.gripper_use_ibr)
            or (camera_role == CameraRole.TOP and self._config.top_use_ibr)
        )
        if not use_ibr:
            out = np.asarray(image, dtype=np.uint8).copy()
            out = self._stabilize_brightness(out, camera_role)
            return out, None

        run_yolo = self._should_run_segmentation()
        mask = None

        if run_yolo:
            try:
                mask = self._service.segment(image)
                with self._last_mask_lock:
                    self._last_mask = mask.copy()
            except Exception as e:
                self._fallback_warn_count += 1
                if self._fallback_warn_count % FALLBACK_WARN_EVERY_N_FRAMES == 1:
                    logger.warning(
                        "Segmentation failed (%s), using raw image (warning %d every %d frames)",
                        e,
                        self._fallback_warn_count,
                        FALLBACK_WARN_EVERY_N_FRAMES,
                    )
                return np.asarray(image, dtype=np.uint8).copy(), None

        if mask is None:
            with self._last_mask_lock:
                mask = self._last_mask

        if mask is None:
            return np.asarray(image, dtype=np.uint8).copy(), None

        out = self._apply_mask(image, mask)
        out = self._stabilize_brightness(out, camera_role)
        return out, mask
