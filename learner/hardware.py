"""Thin adapter over canvas-robot-control so the rest of the learner never
imports from it directly. Makes mocking in tests trivial.
"""

from __future__ import annotations

import sys
import time as _time
from pathlib import Path
from typing import Any

import numpy as np


def _ensure_sys_path(cfg) -> None:
    """Add sibling repo roots to sys.path once per process."""
    for attr in ("canvas_robot_control", "canvas_world_model"):
        p = getattr(cfg.paths, attr, None)
        if p and p not in sys.path:
            sys.path.insert(0, p)


class Hardware:
    """Wraps RobotInterface + WorldModelPredictor + camera stacking utilities."""

    def __init__(self, cfg, dry_run: bool = False):
        self.cfg = cfg
        self.dry_run = dry_run
        _ensure_sys_path(cfg)

        from control.robot_interface import (  # type: ignore
            RobotInterface,
            DryRunRobotInterface,
            JOINTS,
        )
        from control.world_model import WorldModelPredictor  # type: ignore
        from control.canvas_utils import (  # type: ignore
            stack_cameras_vertically,
            FRAME_SIZE,
        )

        self._JOINTS = JOINTS
        self._WorldModelPredictor = WorldModelPredictor
        self._stack_cameras_vertically = stack_cameras_vertically
        self._frame_size = FRAME_SIZE

        if dry_run:
            self.robot = DryRunRobotInterface(
                control_joint=cfg.robot.control_joint,
                step_size=cfg.robot.step_size,
                joint_min=cfg.robot.joint_min,
                joint_max=cfg.robot.joint_max,
            )
        else:
            self.robot = RobotInterface(
                port=cfg.robot.port,
                robot_id=cfg.robot.robot_id,
                control_joint=cfg.robot.control_joint,
                step_size=cfg.robot.step_size,
                joint_min=cfg.robot.joint_min,
                joint_max=cfg.robot.joint_max,
                base_camera_index=cfg.robot.base_camera,
                wrist_camera_index=cfg.robot.wrist_camera,
                camera_width=cfg.robot.camera_width,
                camera_height=cfg.robot.camera_height,
                camera_fps=cfg.robot.camera_fps,
            )

        self.control_idx = JOINTS.index(cfg.robot.control_joint)
        self.predictor: Any = None

    # ------------------------------------------------------------------ robot

    def connect(self) -> None:
        self.robot.connect()

    def disconnect(self) -> None:
        try:
            self.robot.disconnect()
        except Exception:
            pass

    def observe(self) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
        """Grab cameras + motor state and return the stacked context frame."""
        cameras, motor_state = self.robot.get_state()
        ctx = self._stack_cameras_vertically(
            cameras["base"], cameras["wrist"], self._frame_size
        )
        return cameras, motor_state, ctx

    def execute(self, action: int) -> None:
        self.robot.execute_action(action)

    def goto(
        self,
        joint: str,
        target: float,
        settle_time: float = 0.5,
        tolerance: float = 2.0,
        timeout: float = 2.0,
    ) -> None:
        """Move a single joint to an absolute position and wait for settle.

        Used by VERIFY to initialize each probe at an error-weighted
        starting state within the curriculum's active range. Other joints
        are held at their current positions (read once, written back
        unchanged in the goal dict), matching the hold-inactive-joints
        pattern in `RobotInterface.execute_action`.

        Polls `Present_Position` until within `tolerance` of target or
        `timeout` elapses, then sleeps `settle_time` to let the motor fully
        relax into position before the caller's first `observe()`.

        On `DryRunRobotInterface` we just set `_positions[joint]` directly
        (there's no simulated motion dynamics; the mock is instantaneous).
        """
        # Only clamp when moving the primary control joint — joint_min /
        # joint_max in the config describe that joint's safe range, not
        # elbow_flex / shoulder_lift / etc. Non-primary callers are
        # expected to pass safe values (curriculum ranges, home pose).
        if joint == self.cfg.robot.control_joint:
            target = float(max(
                self.cfg.robot.joint_min,
                min(self.cfg.robot.joint_max, target),
            ))
        else:
            target = float(target)

        if self.dry_run:
            positions = getattr(self.robot, "_positions", None)
            if positions is not None:
                positions[joint] = target
            return

        bus = getattr(self.robot, "bus", None)
        if bus is None:
            raise RuntimeError("robot is not connected (no bus)")

        positions = bus.sync_read("Present_Position")
        goal = {j: positions[j] for j in self._JOINTS}
        goal[joint] = target
        bus.sync_write("Goal_Position", goal)

        deadline = _time.perf_counter() + timeout
        while _time.perf_counter() < deadline:
            _time.sleep(0.02)
            try:
                current = bus.sync_read("Present_Position").get(joint)
            except Exception:
                continue
            if current is not None and abs(current - target) <= tolerance:
                break
        if settle_time > 0:
            _time.sleep(settle_time)

    def goto_home(self, home: dict, settle_time: float = 1.0) -> None:
        """Move all joints to a fixed home pose in one sync_write. Used to
        park the arm before inactive phases (training, termination) so
        gravity droop can't crash it down on release.

        `home` is a dict of joint_name -> position in the same normalized
        units as the motor bus reads (RANGE_M100_100 for SO-101).
        """
        if not home:
            return

        if self.dry_run:
            positions = getattr(self.robot, "_positions", None)
            if positions is not None:
                for j, v in home.items():
                    positions[j] = float(v)
            return

        bus = getattr(self.robot, "bus", None)
        if bus is None:
            raise RuntimeError("robot is not connected (no bus)")

        goal = {j: float(v) for j, v in home.items()}
        bus.sync_write("Goal_Position", goal)
        if settle_time > 0:
            _time.sleep(settle_time)

    # --------------------------------------------------------------- predictor

    def load_predictor(self, checkpoint_path: str) -> None:
        self.predictor = self._WorldModelPredictor(
            checkpoint_path=checkpoint_path,
            canvas_world_model_path=self.cfg.paths.canvas_world_model,
        )
        self.predictor.load()

    def reload_checkpoint(self, checkpoint_path: str) -> None:
        self.load_predictor(checkpoint_path)

    def predict(
        self,
        context_frame: np.ndarray,
        motor_state: np.ndarray,
        action: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.predictor is None:
            raise RuntimeError("Predictor not loaded — call load_predictor first")
        results = self.predictor.predict_batch(
            context_frame,
            motor_state,
            [action],
            step_size=self.cfg.robot.step_size,
            control_joint_idx=self.control_idx,
            prediction_depth=1,
        )
        return results[0]
