# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use it except in compliance with the License.
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
Real-time brightness preprocessing for overexposed top-view camera images.
Uses luminance channel (Y in YCrCb), gamma correction (gamma > 1) to reduce
brightness only when too bright, with EMA smoothing for temporal consistency.
Optional CLAHE on luminance to recover details without increasing brightness.
"""

from __future__ import annotations

import numpy as np


class BrightnessReducePreprocessor:
    """
    Reduces image brightness only when the image is too bright, using luminance-based
    gamma correction and optional CLAHE. Designed for real-time control with no
    resolution or geometry change.

    - Operates on luminance (Y in YCrCb); gamma > 1 suppresses overexposure.
    - Applies correction only when mean luminance exceeds luminance_threshold.
    - EMA smoothing of the applied gamma for temporal consistency (no flickering).
    - Optional CLAHE with low clipLimit to recover detail without increasing brightness.
    - Default behavior: only reduce brightness, never increase it.
    """

    def __init__(
        self,
        luminance_threshold: float = 165.0,
        gamma: float = 1.4,
        brightness_scale: float = 1.0,
        gamma_ema_alpha: float = 0.85,
        use_clahe: bool = True,
        clahe_clip_limit: float = 2.0,
        clahe_tile_size: int = 8,
    ) -> None:
        """
        Args:
            luminance_threshold: Mean luminance (0–255) above which to reduce brightness.
                Typical overexposed indoor: ~160–180. Only reduction is applied when above this.
            gamma: Gamma correction factor (> 1.0 darkens). Applied to luminance channel only.
            brightness_scale: Optional global scale on luminance after gamma (default 1.0 = no scale).
                Use <= 1.0 to avoid increasing brightness.
            gamma_ema_alpha: EMA smoothing factor for applied gamma (0–1). Higher = more smoothing.
                0.85 gives stable, flicker-free behavior.
            use_clahe: If True, apply CLAHE on luminance (LAB L) after gamma to recover detail.
            clahe_clip_limit: CLAHE clip limit (low value avoids boosting brightness).
            clahe_tile_size: CLAHE grid size (e.g. 8 for 8x8).
        """
        self.luminance_threshold = max(0.0, min(255.0, float(luminance_threshold)))
        self.gamma = max(1.0, float(gamma))
        self.brightness_scale = max(0.01, min(1.0, float(brightness_scale)))
        self.gamma_ema_alpha = max(0.0, min(1.0, float(gamma_ema_alpha)))
        self.use_clahe = bool(use_clahe)
        self.clahe_clip_limit = max(0.0, min(10.0, float(clahe_clip_limit)))
        self.clahe_tile_size = max(2, min(32, int(clahe_tile_size)))

        # EMA state: effective gamma used last frame (1.0 = no correction)
        self._ema_gamma: float = 1.0

    def reset(self) -> None:
        """Reset EMA state for a new session."""
        self._ema_gamma = 1.0

    def _apply_gamma_to_luminance_yuv(self, image: np.ndarray, gamma_val: float) -> np.ndarray:
        """Apply gamma correction to Y channel of YCrCb; leave Cr,Cb unchanged. In-place style, returns new array."""
        try:
            import cv2
        except ImportError:
            return image

        out = np.asarray(image, dtype=np.uint8).copy()
        if out.ndim != 3 or out.shape[2] != 3:
            return out

        # RGB -> YCrCb (OpenCV uses BGR order for color conversion names; we assume input is RGB)
        ycrcb = cv2.cvtColor(out, cv2.COLOR_RGB2YCrCb)
        y, cr, cb = cv2.split(ycrcb)

        # Gamma on Y only: Y_out = (Y/255)^gamma * 255
        y_float = np.float32(y) / 255.0
        np.power(y_float, gamma_val, out=y_float)
        y_new = (y_float * 255.0).clip(0, 255).astype(np.uint8)

        ycrcb_out = cv2.merge([y_new, cr, cb])
        out = cv2.cvtColor(ycrcb_out, cv2.COLOR_YCrCb2RGB)
        return out

    def _apply_clahe_lab(self, image: np.ndarray) -> np.ndarray:
        """Apply CLAHE to L channel of LAB; recover detail without increasing brightness."""
        try:
            import cv2
        except ImportError:
            return image

        out = np.asarray(image, dtype=np.uint8).copy()
        if out.ndim != 3 or out.shape[2] != 3:
            return out

        lab = cv2.cvtColor(out, cv2.COLOR_RGB2LAB)
        l_ch, a, b = cv2.split(lab)
        clip = max(0.0, min(10.0, self.clahe_clip_limit))
        tile = max(2, min(32, self.clahe_tile_size))
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
        l_eq = clahe.apply(l_ch)
        lab_eq = cv2.merge([l_eq, a, b])
        out = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)
        return out

    def process(self, image: np.ndarray) -> np.ndarray:
        """
        Process one frame: reduce brightness only when too bright (luminance > threshold),
        using EMA-smoothed gamma on the luminance channel, then optional CLAHE.

        Args:
            image: RGB image (H, W, 3), uint8.

        Returns:
            Processed RGB image, same shape and dtype. Resolution and geometry unchanged.
        """
        try:
            import cv2
        except ImportError:
            return np.asarray(image, dtype=np.uint8).copy()

        out = np.asarray(image, dtype=np.uint8).copy()
        if out.ndim != 3 or out.shape[2] != 3:
            return out

        # Luminance: use Y from YCrCb
        ycrcb = cv2.cvtColor(out, cv2.COLOR_RGB2YCrCb)
        y_channel = ycrcb[:, :, 0]
        mean_luminance = float(np.mean(y_channel))

        # Only reduce when too bright; target gamma: use configured gamma when above threshold
        if mean_luminance <= self.luminance_threshold:
            target_gamma = 1.0  # no reduction
        else:
            target_gamma = self.gamma

        # EMA smoothing for temporal consistency
        alpha = self.gamma_ema_alpha
        self._ema_gamma = alpha * self._ema_gamma + (1.0 - alpha) * target_gamma

        # Apply gamma only if meaningfully > 1 (avoid unnecessary work)
        if self._ema_gamma > 1.01:
            out = self._apply_gamma_to_luminance_yuv(out, self._ema_gamma)
            if self.brightness_scale < 1.0:
                ycrcb = cv2.cvtColor(out, cv2.COLOR_RGB2YCrCb)
                y, cr, cb = cv2.split(ycrcb)
                y = (np.float32(y) * self.brightness_scale).clip(0, 255).astype(np.uint8)
                out = cv2.cvtColor(cv2.merge([y, cr, cb]), cv2.COLOR_YCrCb2RGB)

        if self.use_clahe:
            out = self._apply_clahe_lab(out)

        return out
