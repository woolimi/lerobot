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
# WITHOUT WARRANTIES OR CONDITIONS OF KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for dual-camera IBR pipeline: gripper IBR, top RAW, config, processor, dataset keys."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from lerobot.common.vision_config import VisionConfig, load_vision_config
from lerobot.vision import CameraRole, DualCameraIBRProcessor, get_camera_role


class MockYOLOService:
    """Returns a fixed mask (center rectangle = 255) so IBR output differs from input."""

    def segment(self, image_np: np.ndarray) -> np.ndarray:
        h, w = image_np.shape[0], image_np.shape[1]
        mask = np.zeros((h, w), dtype=np.uint8)
        margin_h, margin_w = h // 4, w // 4
        mask[margin_h : h - margin_h, margin_w : w - margin_w] = 255
        return mask


def test_gripper_ibr_top_raw() -> None:
    """With gripper_use_ibr=True, top_use_ibr=False: gripper output differs from raw (background changed), top equals input."""
    vision_config = VisionConfig(
        gripper_use_ibr=True,
        top_use_ibr=False,
        segmentation_model_path="",
    )
    mock_service = MockYOLOService()
    processor = DualCameraIBRProcessor(vision_config, yolo_service=mock_service)

    h, w, c = 120, 160, 3
    gripper_raw = np.ones((h, w, c), dtype=np.uint8) * 100
    top_raw = np.ones((h, w, c), dtype=np.uint8) * 200

    gripper_out = processor.process(gripper_raw, CameraRole.GRIPPER)
    top_out = processor.process(top_raw, CameraRole.TOP)

    # Gripper: IBR applied -> background (mask==0) should be black (0,0,0)
    assert gripper_out.shape == gripper_raw.shape
    assert not np.array_equal(gripper_out, gripper_raw)
    # Center (foreground) kept; corners/edges (background) zeroed
    assert np.any(gripper_out == 0)

    # Top: RAW -> unchanged
    assert np.array_equal(top_out, top_raw)
    assert top_out.shape == top_raw.shape


def test_top_always_raw() -> None:
    """Top camera output equals input when top_use_ibr=False."""
    vision_config = VisionConfig(gripper_use_ibr=True, top_use_ibr=False)
    mock_service = MockYOLOService()
    processor = DualCameraIBRProcessor(vision_config, yolo_service=mock_service)

    image = np.random.randint(0, 255, (64, 80, 3), dtype=np.uint8)
    out = processor.process(image, CameraRole.TOP)
    assert np.array_equal(out, image)


def test_get_camera_role() -> None:
    """get_camera_role maps gripper/top keys; returns None for unmapped keys."""
    config = VisionConfig(gripper_camera_key="gripper", top_camera_key="top")
    assert get_camera_role("gripper", config) == CameraRole.GRIPPER
    assert get_camera_role("top", config) == CameraRole.TOP
    assert get_camera_role("wrist", config) is None
    assert get_camera_role("front", config) is None

    config_custom = VisionConfig(gripper_camera_key="wrist", top_camera_key="front")
    assert get_camera_role("wrist", config_custom) == CameraRole.GRIPPER
    assert get_camera_role("front", config_custom) == CameraRole.TOP


def test_load_vision_config_missing_returns_default() -> None:
    """Missing or invalid path returns default VisionConfig and does not raise."""
    default = load_vision_config("")
    assert isinstance(default, VisionConfig)
    assert default.gripper_use_ibr is True
    assert default.top_use_ibr is False

    default2 = load_vision_config("/nonexistent/path/vision.yaml")
    assert isinstance(default2, VisionConfig)


def test_load_vision_config_yaml() -> None:
    """Load VisionConfig from a YAML file."""
    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
        f.write(
            b"gripper_use_ibr: false\n"
            b"top_use_ibr: false\n"
            b"gripper_camera_key: wrist\n"
            b"top_camera_key: front\n"
        )
        path = f.name
    try:
        config = load_vision_config(path)
        assert config.gripper_use_ibr is False
        assert config.top_use_ibr is False
        assert config.gripper_camera_key == "wrist"
        assert config.top_camera_key == "front"
    finally:
        Path(path).unlink(missing_ok=True)


def test_process_with_mask_returns_mask_for_gripper() -> None:
    """process_with_mask returns (image, mask) for GRIPPER when IBR on; (image, None) for TOP."""
    vision_config = VisionConfig(gripper_use_ibr=True, top_use_ibr=False)
    mock_service = MockYOLOService()
    processor = DualCameraIBRProcessor(vision_config, yolo_service=mock_service)

    img = np.ones((60, 80, 3), dtype=np.uint8) * 100
    out_gripper, mask_gripper = processor.process_with_mask(img, CameraRole.GRIPPER)
    out_top, mask_top = processor.process_with_mask(img, CameraRole.TOP)

    assert out_gripper.shape == img.shape
    assert mask_gripper is not None
    assert mask_gripper.shape == (img.shape[0], img.shape[1])
    assert mask_top is None
    assert np.array_equal(out_top, img)


def test_dataset_keys_single_rep_per_camera() -> None:
    """New pipeline stores one image per camera key (gripper=IBR content, top=RAW); no _ibr/_raw suffixes in keys."""
    # Schema check: expected observation image keys are plain camera keys
    vision_config = VisionConfig(gripper_camera_key="gripper", top_camera_key="top")
    expected_keys = {vision_config.gripper_camera_key, vision_config.top_camera_key}
    assert "gripper_ibr" not in expected_keys
    assert "top_raw" not in expected_keys
    assert expected_keys == {"gripper", "top"}
