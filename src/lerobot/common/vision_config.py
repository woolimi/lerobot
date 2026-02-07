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
Single source of truth for vision/IBR pipeline parameters.
Both teleoperate and record MUST import from this module only.
"""

import json
import logging
from dataclasses import dataclass, fields
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class VisionConfig:
    """
    Dual-camera vision config: gripper IBR vs top RAW.
    All segmentation and display parameters are defined here only.
    """

    gripper_use_ibr: bool = True
    top_use_ibr: bool = False
    segmentation_model_path: str = ""
    segmentation_confidence: float = 0.35
    segmentation_iou_threshold: float = 0.5
    segmentation_frame_skip: int = 2
    device: str = "auto"
    rerun_show_processed: bool = True
    gripper_camera_key: str = "gripper"
    top_camera_key: str = "top"
    # Top view: stabilize brightness so it looks similar in any lighting (CLAHE on luminance).
    top_brightness_stabilize: bool = False
    top_brightness_clip_limit: float = 2.0  # CLAHE clip limit (higher = more contrast)
    top_brightness_tile_size: int = 8  # CLAHE grid size (8x8)
    # Top view: manual brightness/contrast/gamma (applied before CLAHE). Use when top is overexposed.
    top_brightness: float = 1.0  # luminance scale: <1 darker, >1 brighter (e.g. 0.6 to reduce overexposure)
    top_contrast: float = 1.0  # contrast: >1 more contrast, <1 less
    top_gamma: float = 1.0  # gamma: >1 darken (e.g. 1.2~1.5 for overexposed), <1 brighten
    # Gripper view: same CLAHE brightness stabilization (applied after IBR when enabled).
    gripper_brightness_stabilize: bool = False
    gripper_brightness_clip_limit: float = 2.0
    gripper_brightness_tile_size: int = 8
    # Gripper view: manual brightness/contrast/gamma (optional).
    gripper_brightness: float = 1.0
    gripper_contrast: float = 1.0
    gripper_gamma: float = 1.0

    def use_ibr_for_camera_key(self, camera_key: str) -> bool:
        if camera_key == self.gripper_camera_key:
            return self.gripper_use_ibr
        if camera_key == self.top_camera_key:
            return self.top_use_ibr
        return False


def load_vision_config(config_path: str | Path | None) -> VisionConfig:
    """
    Load VisionConfig from YAML or JSON file.
    If path is None, missing, or invalid, returns default VisionConfig() and logs a warning.
    """
    if config_path is None or config_path == "":
        return VisionConfig()

    path = Path(config_path)
    if not path.exists():
        logger.warning("Vision config path does not exist: %s. Using defaults.", path)
        return VisionConfig()

    try:
        raw = path.read_text()
    except OSError as e:
        logger.warning("Failed to read vision config %s: %s. Using defaults.", path, e)
        return VisionConfig()

    data: dict
    suffix = path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        try:
            import yaml
            data = yaml.safe_load(raw)
        except ImportError:
            logger.warning("PyYAML not installed; cannot load %s. Using defaults.", path)
            return VisionConfig()
        except Exception as e:
            logger.warning("Invalid YAML in %s: %s. Using defaults.", path, e)
            return VisionConfig()
    elif suffix == ".json":
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            logger.warning("Invalid JSON in %s: %s. Using defaults.", path, e)
            return VisionConfig()
    else:
        logger.warning("Unknown vision config extension: %s. Using defaults.", suffix)
        return VisionConfig()

    if not isinstance(data, dict):
        logger.warning("Vision config must be a dict. Using defaults.")
        return VisionConfig()

    # Only pass keys that VisionConfig accepts
    valid_keys = {f.name for f in fields(VisionConfig)}
    filtered = {k: v for k, v in data.items() if k in valid_keys}
    return VisionConfig(**filtered)
