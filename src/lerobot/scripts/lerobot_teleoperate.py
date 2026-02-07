# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
Simple script to control a robot from teleoperation.
Dual-camera vision: gripper IBR, top RAW (same config as record). Rerun shows camera/gripper/ibr, camera/top/raw.

Example:

```shell
lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem58760431541 \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 1920, height: 1080, fps: 30}}" \
    --robot.id=black \
    --teleop.type=so101_leader \
    --teleop.port=/dev/tty.usbmodem58760431551 \
    --teleop.id=blue \
    --display_data=true
```

Example teleoperation with bimanual so100:

```shell
lerobot-teleoperate \
  --robot.type=bi_so_follower \
  --robot.left_arm_config.port=/dev/tty.usbmodem5A460822851 \
  --robot.right_arm_config.port=/dev/tty.usbmodem5A460814411 \
  --robot.id=bimanual_follower \
  --robot.left_arm_config.cameras='{
    wrist: {"type": "opencv", "index_or_path": 1, "width": 640, "height": 480, "fps": 30},
  }' --robot.right_arm_config.cameras='{
    wrist: {"type": "opencv", "index_or_path": 2, "width": 640, "height": 480, "fps": 30},
  }' \
  --teleop.type=bi_so_leader \
  --teleop.left_arm_config.port=/dev/tty.usbmodem5A460852721 \
  --teleop.right_arm_config.port=/dev/tty.usbmodem5A460819811 \
  --teleop.id=bimanual_leader \
  --display_data=true
```

"""

import logging
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from pprint import pformat

import numpy as np
import rerun as rr

from lerobot.common.vision_config import VisionConfig, load_vision_config
from lerobot.configs.default import VisionConfigPath
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.processor import (
    RobotAction,
    RobotObservation,
    RobotProcessorPipeline,
    make_default_processors,
    make_default_robot_observation_processor,
)
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_so_follower,
    earthrover_mini_plus,
    hope_jr,
    koch_follower,
    make_robot_from_config,
    omx_follower,
    reachy2,
    so_follower,
)
from lerobot.teleoperators import (  # noqa: F401
    Teleoperator,
    TeleoperatorConfig,
    bi_so_leader,
    gamepad,
    homunculus,
    keyboard,
    koch_leader,
    make_teleoperator_from_config,
    omx_leader,
    reachy2_teleoperator,
    so_leader,
)
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging, move_cursor_up
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data
from lerobot.vision import DualCameraIBRProcessor, get_camera_role


def _apply_dual_camera_processor(
    obs: RobotObservation,
    processor: DualCameraIBRProcessor,
    vision_config: VisionConfig,
    robot_observation_features: dict,
) -> RobotObservation:
    """Apply gripper IBR / top RAW per key; same as record pipeline."""
    out = dict(obs)
    for key in list(out.keys()):
        if key not in robot_observation_features or not isinstance(
            robot_observation_features.get(key), tuple
        ):
            continue
        role = get_camera_role(key, vision_config)
        if role is not None:
            out[key] = processor.process(out[key], role)
    return out


@dataclass
class TeleoperateConfig:
    # TODO: pepijn, steven: if more robots require multiple teleoperators (like lekiwi) its good to make this possibele in teleop.py and record.py with List[Teleoperator]
    teleop: TeleoperatorConfig
    robot: RobotConfig
    # Limit the maximum frames per second.
    fps: int = 60
    teleop_time_s: float | None = None
    # Display all cameras on screen
    display_data: bool = False
    # Display data on a remote Rerun server
    display_ip: str | None = None
    # Port of the remote Rerun server
    display_port: int | None = None
    # Whether to  display compressed images in Rerun
    display_compressed_images: bool = False
    # Vision: path to shared config YAML/JSON (lerobot.common.vision_config).
    vision: VisionConfigPath | None = None


def teleop_loop(
    teleop: Teleoperator,
    robot: Robot,
    fps: int,
    teleop_action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],
    robot_action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],
    robot_observation_processor: RobotProcessorPipeline[RobotObservation, RobotObservation],
    display_data: bool = False,
    duration: float | None = None,
    display_compressed_images: bool = False,
    dual_camera_processor: DualCameraIBRProcessor | None = None,
    vision_config: VisionConfig | None = None,
    camera_stream_map: dict[str, str] | None = None,
    ibr_async_state: dict | None = None,
):
    """
    This function continuously reads actions from a teleoperation device, processes them through optional
    pipelines, sends them to a robot, and optionally displays the robot's state. The loop runs at a
    specified frequency until a set duration is reached or it is manually interrupted.

    Args:
        teleop: The teleoperator device instance providing control actions.
        robot: The robot instance being controlled.
        fps: The target frequency for the control loop in frames per second.
        display_data: If True, fetches robot observations and displays them in the console and Rerun.
        display_compressed_images: If True, compresses images before sending them to Rerun for display.
        duration: The maximum duration of the teleoperation loop in seconds. If None, the loop runs indefinitely.
        teleop_action_processor: An optional pipeline to process raw actions from the teleoperator.
        robot_action_processor: An optional pipeline to process actions before they are sent to the robot.
        robot_observation_processor: An optional pipeline to process raw observations from the robot.
    """

    display_len = max(len(key) for key in robot.action_features)
    start = time.perf_counter()

    while True:
        loop_start = time.perf_counter()

        # Get robot observation
        obs = robot.get_observation()

        # Process observation (dual-camera: gripper IBR, top RAW; same as record)
        if ibr_async_state is not None:
            try:
                ibr_async_state["queue"].put_nowait(obs)
            except Exception:
                pass
            with ibr_async_state["lock"]:
                obs_processed = dict(ibr_async_state["shared"]) if ibr_async_state["shared"] else obs
        elif dual_camera_processor is not None and vision_config is not None:
            obs_processed = _apply_dual_camera_processor(
                obs, dual_camera_processor, vision_config, robot.observation_features
            )
        else:
            obs_processed = robot_observation_processor(obs)

        # Get teleop action and process with same obs as stored/displayed
        raw_action = teleop.get_action()
        teleop_action = teleop_action_processor((raw_action, obs_processed))
        robot_action_to_send = robot_action_processor((teleop_action, obs_processed))

        # Send processed action to robot
        _ = robot.send_action(robot_action_to_send)

        if display_data:
            log_rerun_data(
                observation=obs_processed,
                action=teleop_action,
                compress_images=display_compressed_images,
                camera_stream_map=camera_stream_map,
            )
            if vision_config and camera_stream_map:
                rr.log(
                    "camera/debug",
                    rr.TextDocument(
                        f"gripper: IBR {'ON' if vision_config.gripper_use_ibr else 'OFF'}, "
                        f"top: IBR {'ON' if vision_config.top_use_ibr else 'OFF'}, "
                        f"conf={vision_config.segmentation_confidence}"
                    ),
                )

            print("\n" + "-" * (display_len + 10))
            print(f"{'NAME':<{display_len}} | {'NORM':>7}")
            # Display the final robot action that was sent
            for motor, value in robot_action_to_send.items():
                print(f"{motor:<{display_len}} | {value:>7.2f}")
            move_cursor_up(len(robot_action_to_send) + 3)

        dt_s = time.perf_counter() - loop_start
        precise_sleep(max(1 / fps - dt_s, 0.0))
        loop_s = time.perf_counter() - loop_start
        print(f"Teleop loop time: {loop_s * 1e3:.2f}ms ({1 / loop_s:.0f} Hz)")
        move_cursor_up(1)

        if duration is not None and time.perf_counter() - start >= duration:
            return


def _ibr_worker(
    queue,
    shared: dict,
    lock: threading.Lock,
    dual_camera_processor: DualCameraIBRProcessor,
    vision_config: VisionConfig,
    robot_observation_features: dict,
) -> None:
    """Async worker: run dual-camera IBR (gripper IBR, top RAW) and update shared processed obs."""
    while True:
        try:
            obs = queue.get(timeout=1.0)
            if obs is None:
                break
            processed = _apply_dual_camera_processor(
                obs, dual_camera_processor, vision_config, robot_observation_features
            )
            with lock:
                shared.clear()
                shared.update(processed)
        except Exception:
            continue


@parser.wrap()
def teleoperate(cfg: TeleoperateConfig):
    import queue as queue_module

    init_logging()
    if getattr(cfg, "vision", None) is None:
        cfg.vision = VisionConfigPath()
    vision_config = load_vision_config(cfg.vision.config_path if cfg.vision else "")
    logging.info(pformat(asdict(cfg)))
    if cfg.display_data:
        init_rerun(session_name="teleoperation", ip=cfg.display_ip, port=cfg.display_port)
    display_compressed_images = (
        True
        if (cfg.display_data and cfg.display_ip is not None and cfg.display_port is not None)
        else cfg.display_compressed_images
    )

    ibr_enabled = (
        (vision_config.gripper_use_ibr or vision_config.top_use_ibr)
        and bool(vision_config.segmentation_model_path)
        and Path(vision_config.segmentation_model_path).exists()
    )
    # Always create processor so view settings (brightness/contrast/gamma, CLAHE) apply regardless of IBR.
    dual_camera_processor = DualCameraIBRProcessor(vision_config)
    camera_stream_map: dict[str, str] | None = None
    if vision_config and (vision_config.gripper_use_ibr or vision_config.top_use_ibr):
        camera_stream_map = {
            vision_config.gripper_camera_key: "camera/gripper/ibr"
            if vision_config.gripper_use_ibr
            else "camera/gripper/raw",
            vision_config.top_camera_key: "camera/top/raw"
            if not vision_config.top_use_ibr
            else "camera/top/ibr",
        }
    if ibr_enabled:
        logging.info(
            "Dual-camera IBR: gripper=%s, top=%s, model=%s",
            "IBR" if vision_config.gripper_use_ibr else "RAW",
            "IBR" if vision_config.top_use_ibr else "RAW",
            vision_config.segmentation_model_path,
        )
    elif vision_config.segmentation_model_path and (
        vision_config.gripper_use_ibr or vision_config.top_use_ibr
    ):
        logging.warning(
            "IBR requested but model not found at %s; using RAW for all cameras.",
            vision_config.segmentation_model_path,
        )

    robot_observation_processor = make_default_robot_observation_processor()
    teleop_action_processor, robot_action_processor, _ = make_default_processors()

    teleop = make_teleoperator_from_config(cfg.teleop)
    robot = make_robot_from_config(cfg.robot)
    teleop.connect()
    robot.connect()

    ibr_async_state = None
    if ibr_enabled and cfg.display_data and dual_camera_processor is not None:
        ibr_queue = queue_module.Queue(maxsize=1)
        ibr_shared: dict = {}
        ibr_lock = threading.Lock()
        ibr_async_state = {"queue": ibr_queue, "shared": ibr_shared, "lock": ibr_lock}
        worker = threading.Thread(
            target=_ibr_worker,
            args=(
                ibr_queue,
                ibr_shared,
                ibr_lock,
                dual_camera_processor,
                vision_config,
                robot.observation_features,
            ),
            daemon=True,
        )
        worker.start()

    try:
        teleop_loop(
            teleop=teleop,
            robot=robot,
            fps=cfg.fps,
            display_data=cfg.display_data,
            duration=cfg.teleop_time_s,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
            display_compressed_images=display_compressed_images,
            dual_camera_processor=dual_camera_processor,
            vision_config=vision_config,
            camera_stream_map=camera_stream_map,
            ibr_async_state=ibr_async_state,
        )
    except KeyboardInterrupt:
        pass
    finally:
        if ibr_async_state is not None:
            try:
                ibr_async_state["queue"].put(None, timeout=0.5)
            except Exception:
                pass
        if cfg.display_data:
            rr.rerun_shutdown()
        teleop.disconnect()
        robot.disconnect()


def main():
    register_third_party_plugins()
    teleoperate()


if __name__ == "__main__":
    main()
