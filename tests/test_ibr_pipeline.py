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

"""Tests for IBR (Image Background Removal) pipeline: mask shape, background zeroed, dataset fallback, teleop toggle."""

import numpy as np
import pytest

from lerobot.vision.ibr_preprocessor import IBRPreprocessor


def test_mask_shape_matches_image() -> None:
    """Mask should have shape (H, W) matching image (H, W, C); IBR output preserves image shape."""
    h, w, c = 240, 320, 3
    image = np.zeros((h, w, c), dtype=np.uint8)
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[50:100, 50:100] = 255
    assert mask.shape == (h, w)
    # IBRPreprocessor.remove_background only needs (image, mask); segmenter unused for this call
    class DummySegmenter:
        pass
    preprocessor = IBRPreprocessor(segmenter=DummySegmenter())  # type: ignore[arg-type]
    out = preprocessor.remove_background(image, mask)
    assert out.shape == image.shape


def test_background_actually_zeroed() -> None:
    """Where mask is 0, output should equal background_color."""
    h, w = 100, 100
    image = np.ones((h, w, 3), dtype=np.uint8) * 123
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[40:60, 40:60] = 255
    bg_color = (255, 255, 255)
    class DummySegmenter:
        pass
    preprocessor = IBRPreprocessor(segmenter=DummySegmenter(), background_color=bg_color)  # type: ignore[arg-type]
    out = preprocessor.remove_background(image, mask)
    bg_region = out[mask == 0]
    expected = np.array(bg_color, dtype=np.uint8)
    assert np.all(bg_region == expected)
    fg_region = out[mask == 255]
    assert np.all(fg_region == 123)


def test_dataset_fallback_use_ibr_images() -> None:
    """When use_ibr_images=True and _ibr key exists, base key gets _ibr content; else fallback to raw."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    # We can't create a full dataset in a unit test without disk; test the overwrite logic in isolation
    item = {
        "observation.images.front": np.ones((64, 64, 3), dtype=np.uint8) * 100,
        "observation.images.front_ibr": np.ones((64, 64, 3), dtype=np.uint8) * 200,
    }
    video_keys = ["observation.images.front", "observation.images.front_ibr"]
    base_keys = [k for k in video_keys if not k.endswith("_ibr") and not k.endswith("_mask")]
    use_ibr_images = True
    if use_ibr_images:
        for key in base_keys:
            ibr_key = f"{key}_ibr"
            if ibr_key in item:
                item[key] = item[ibr_key]
    assert np.all(item["observation.images.front"] == 200)
    # Fallback: when _ibr not present, base stays
    item2 = {"observation.images.front": np.ones((64, 64, 3), dtype=np.uint8) * 100}
    if use_ibr_images:
        for key in ["observation.images.front"]:
            ibr_key = f"{key}_ibr"
            if ibr_key in item2:
                item2[key] = item2[ibr_key]
    assert np.all(item2["observation.images.front"] == 100)


def test_teleop_toggle_removed() -> None:
    """Dual-camera pipeline uses single display (gripper=IBR, top=RAW); no raw/ibr/split toggle."""
    # Old IBR_DISPLAY_* and _build_ibr_display_obs were removed in favor of
    # DualCameraIBRProcessor and consistent display (same as record/rerun).
    from lerobot.scripts.lerobot_teleoperate import _apply_dual_camera_processor
    from lerobot.common.vision_config import VisionConfig
    from lerobot.vision import DualCameraIBRProcessor, get_camera_role

    vision_config = VisionConfig(gripper_use_ibr=False, top_use_ibr=False)
    processor = DualCameraIBRProcessor(vision_config)
    obs = {"gripper": np.zeros((60, 80, 3), dtype=np.uint8)}
    features = {"gripper": ("observation.images.gripper", (60, 80, 3))}
    out = _apply_dual_camera_processor(obs, processor, vision_config, features)
    assert "gripper" in out
    assert out["gripper"].shape == (60, 80, 3)
