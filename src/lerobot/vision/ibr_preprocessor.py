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
IBR (Image Background Removal) preprocessor using a segmenter and numpy masking.
"""

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from lerobot.vision.yolo26_segmenter import YOLO26Segmenter


class IBRPreprocessor:
    """Applies background removal to an image using a segmentation mask."""

    def __init__(
        self,
        segmenter: "YOLO26Segmenter",
        background_color: tuple[int, int, int] = (0, 0, 0),
    ) -> None:
        self.segmenter = segmenter
        self.background_color = background_color

    def remove_background(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Zero out background pixels (where mask == 0); keep foreground unchanged.

        Args:
            image: HWC uint8.
            mask: HW uint8, 0 or 255.

        Returns:
            HWC uint8 image with background set to background_color.
        """
        out = np.asarray(image, dtype=np.uint8).copy()
        bg = np.array(self.background_color, dtype=np.uint8)
        out[mask == 0] = bg
        return out

    def process(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Run segmentation and return IBR image and mask.

        Args:
            image: HWC uint8.

        Returns:
            processed_image: HWC uint8 with background removed.
            mask: HW uint8, 0 or 255.
        """
        mask = self.segmenter.predict_mask(image)
        processed_image = self.remove_background(image, mask)
        return processed_image, mask
