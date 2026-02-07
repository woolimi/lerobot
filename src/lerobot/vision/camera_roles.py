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

"""Camera role enum for dual-camera IBR: gripper (IBR) vs top (RAW)."""

from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lerobot.common.vision_config import VisionConfig


class CameraRole(Enum):
    GRIPPER = "gripper"
    TOP = "top"


def get_camera_role(camera_key: str, vision_config: "VisionConfig") -> CameraRole | None:
    """
    Map a robot camera key to a CameraRole using vision_config mapping.
    Returns None for unmapped keys (e.g. other cameras).
    """
    if camera_key == vision_config.gripper_camera_key:
        return CameraRole.GRIPPER
    if camera_key == vision_config.top_camera_key:
        return CameraRole.TOP
    return None
