#!/usr/bin/env python3
"""Run the trained Go2 policy on CPU in a native MuJoCo viewer.

This is the WSLg counterpart of the CUDA/MuJoCo-Warp evaluator.  Physics and
policy inference both run locally on the CPU; only ``mujoco``, ``numpy`` and a
CPU build of ``torch`` are required.

Keys (focus the MuJoCo window first):
    W/S  increase/decrease forward velocity
    A/D  increase/decrease leftward velocity
    Q/E  increase/decrease counter-clockwise yaw rate
    X    stop
    1..6 select full-speed direction presets
    T    cycle through the five terrain types
    [/]  decrease/increase terrain difficulty (0..9)
    R    reset the robot
    P or Space  pause/resume
    Esc  close the viewer
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from pathlib import Path

# WSLg supplies an X11 display through WSL.  GLFW is required for a native
# interactive window; retain an explicit user override if one was supplied.
os.environ.setdefault("MUJOCO_GL", "glfw")
os.environ.pop("PYOPENGL_PLATFORM", None)

import mujoco
import mujoco.viewer
import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Constants copied from go2_imitation_warp_apex_terrain_fixed.ipynb

CONTROL_HZ = 50.0
CONTROL_DT = 1.0 / CONTROL_HZ
T_GAIT, H0, H_SWING = 0.40, 0.33, 0.05
CYC = int(round(T_GAIT / CONTROL_DT))
CMD_SCALE = np.asarray([0.60, 0.30, 1.00], dtype=np.float32)
SERVO_KP, SERVO_KD = 60.0, 2.0
ACT_SCALE = np.asarray([0.5, 0.8, 0.8] * 4, dtype=np.float64)

L1 = 0.213
L2 = float(np.hypot(0.002, 0.213))
DELTA = float(np.arctan2(0.002, 0.213))
D_HIP = 0.0955
FOOT_R = 0.022

LEGS = [
    ("FL", 0.1934, 0.0465, +1, 0.0),
    ("FR", 0.1934, -0.0465, -1, 0.5),
    ("RL", -0.1934, 0.0465, +1, 0.5),
    ("RR", -0.1934, -0.0465, -1, 0.0),
]
QPOS_ORDER = ["FL", "FR", "RL", "RR"]

BASE_OBS_DIM, PRIOR_OBS_DIM, ACT_DIM = 35, 36, 12
SUPPORTED_OBS_DIMS = (BASE_OBS_DIM, PRIOR_OBS_DIM)
ENCODER_POP_DIM, DECODER_POP_DIM = 64, 256
HIDDEN_SIZES = (256, 256, ACT_DIM * 2 * DECODER_POP_DIM)
MEM_DIM = sum(HIDDEN_SIZES)
NORM_CLIP_LIMIT = 50.0

TERRAIN_LEVELS = 10
TERRAIN_TYPES = 5
TERRAIN_TYPE_NAMES = (
    "smooth_slope",
    "rough_slope",
    "stairs_up",
    "stairs_down",
    "discrete",
)
TERRAIN_PATCH_SIZE = 8.0
TERRAIN_GRID_RES = 0.10
TERRAIN_GRID_N = int(round(TERRAIN_PATCH_SIZE / TERRAIN_GRID_RES)) + 1
TERRAIN_X_CENTERS = np.linspace(
    -0.5 * TERRAIN_PATCH_SIZE * (TERRAIN_LEVELS - 1),
    0.5 * TERRAIN_PATCH_SIZE * (TERRAIN_LEVELS - 1),
    TERRAIN_LEVELS,
)
TERRAIN_Y_CENTERS = np.linspace(-20.0, 20.0, TERRAIN_TYPES)
TERRAIN_Z_MIN, TERRAIN_Z_RANGE = -1.5, 3.0

LIF_BETA, LIF_THRESHOLD, SPIKE_SLOPE = 0.5, 1.0, 3.0
ENC_VTH = 0.999

SCRIPT_DIR = Path(__file__).resolve().parent


def resolve_input_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else SCRIPT_DIR / path


def leg_ik(position: np.ndarray, side: int) -> np.ndarray:
    px, py, pz = position
    d = side * D_HIP
    radius = np.hypot(py, pz)
    th1 = np.arctan2(pz, py) + np.arccos(np.clip(d / radius, -1.0, 1.0))
    qz = -np.sqrt(max(radius * radius - d * d, 1e-12))
    c3 = np.clip(
        (px * px + qz * qz - L1 * L1 - L2 * L2) / (2 * L1 * L2),
        -1.0,
        1.0,
    )
    ph3 = -np.arccos(c3)
    th2 = np.arctan2(-px, -qz) - np.arctan2(
        L2 * np.sin(ph3), L1 + L2 * np.cos(ph3)
    )
    return np.asarray([th1, th2, ph3 - DELTA], dtype=np.float64)


def standing_pose(height: float = H0) -> np.ndarray:
    pose = np.zeros(12, dtype=np.float64)
    for name, _hx, _hy, side, _offset in LEGS:
        local = np.asarray([0.0, side * D_HIP, FOOT_R - height])
        leg_index = QPOS_ORDER.index(name)
        pose[3 * leg_index : 3 * leg_index + 3] = leg_ik(local, side)
    return pose


def build_apex_terrain_bank(seed: int = 4321) -> np.ndarray:
    """Build the five-by-ten terrain bank used by the training notebook."""
    rng = np.random.default_rng(seed)
    axis = np.linspace(
        -TERRAIN_PATCH_SIZE / 2,
        TERRAIN_PATCH_SIZE / 2,
        TERRAIN_GRID_N,
    )
    xx, yy = np.meshgrid(axis, axis)
    bank = np.zeros(
        (TERRAIN_LEVELS, TERRAIN_TYPES, TERRAIN_GRID_N, TERRAIN_GRID_N),
        dtype=np.float32,
    )

    for level in range(TERRAIN_LEVELS):
        difficulty = level / max(1, TERRAIN_LEVELS - 1)
        run = np.sign(xx) * np.maximum(np.abs(xx) - 1.0, 0.0)

        slope = 0.02 + 0.20 * difficulty
        bank[level, 0] = slope * run

        amplitude = 0.008 + 0.055 * difficulty
        coarse = rng.uniform(-amplitude, amplitude, size=(11, 11)).astype(np.float32)
        rough = np.repeat(np.repeat(coarse, 8, axis=0), 8, axis=1)[
            :TERRAIN_GRID_N, :TERRAIN_GRID_N
        ]
        for _ in range(2):
            rough = (
                rough
                + np.roll(rough, 1, 0)
                + np.roll(rough, -1, 0)
                + np.roll(rough, 1, 1)
                + np.roll(rough, -1, 1)
            ) / 5.0
        bank[level, 1] = 0.65 * slope * run + rough

        step_height = 0.015 + 0.085 * difficulty
        stair = (
            np.sign(xx)
            * np.floor(np.maximum(np.abs(xx) - 1.0, 0.0) / 0.40)
            * step_height
        )
        bank[level, 2] = stair
        bank[level, 3] = -stair

        obstacle = np.zeros_like(xx, dtype=np.float32)
        obstacle_height = 0.02 + 0.13 * difficulty
        for _ in range(12 + 2 * level):
            center_x, center_y = rng.uniform(-3.5, 3.5, size=2)
            if abs(center_x) < 1.25 and abs(center_y) < 1.25:
                continue
            size_x, size_y = rng.uniform(0.20, 0.65, size=2)
            height = (
                rng.uniform(0.35, 1.0)
                * obstacle_height
                * rng.choice([-0.45, 1.0], p=[0.2, 0.8])
            )
            mask = (np.abs(xx - center_x) < size_x) & (
                np.abs(yy - center_y) < size_y
            )
            obstacle[mask] = height
        bank[level, 4] = obstacle

        spawn = (np.abs(xx) <= 1.0) & (np.abs(yy) <= 1.0)
        bank[level, :, spawn] = 0.0

    return np.clip(
        bank,
        TERRAIN_Z_MIN + 0.02,
        TERRAIN_Z_MIN + TERRAIN_Z_RANGE - 0.02,
    ).astype(np.float32)


def install_terrain_bank(model: mujoco.MjModel, bank: np.ndarray) -> bool:
    """Populate an APEX terrain XML; return False for an ordinary flat XML."""
    terrain_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_HFIELD, f"terrain_t{i}")
        for i in range(TERRAIN_TYPES)
    ]
    if all(terrain_id < 0 for terrain_id in terrain_ids):
        return False
    if any(terrain_id < 0 for terrain_id in terrain_ids):
        raise ValueError("Terrain XML must contain terrain_t0 through terrain_t4")

    for terrain_type, terrain_id in enumerate(terrain_ids):
        pieces = [
            bank[level, terrain_type, :, :-1]
            for level in range(TERRAIN_LEVELS - 1)
        ]
        pieces.append(bank[-1, terrain_type])
        strip = np.concatenate(pieces, axis=1)
        normalized = np.clip(
            (strip - TERRAIN_Z_MIN) / TERRAIN_Z_RANGE, 0.0, 1.0
        )
        address = int(model.hfield_adr[terrain_id])
        count = int(model.hfield_nrow[terrain_id] * model.hfield_ncol[terrain_id])
        if normalized.size != count:
            raise ValueError(
                f"Terrain heightfield size mismatch: {normalized.shape} != {count}"
            )
        model.hfield_data[address : address + count] = normalized.reshape(-1)
    return True


class BatchTrot:
    """The same analytic trot used to initialize training episodes."""

    def __init__(self, period: float = T_GAIT, beta: float = 0.5, height: float = H0):
        self.period, self.beta, self.height = period, beta, height
        self.hx = torch.tensor([leg[1] for leg in LEGS], dtype=torch.float64)
        self.hy = torch.tensor([leg[2] for leg in LEGS], dtype=torch.float64)
        self.side = torch.tensor([float(leg[3]) for leg in LEGS], dtype=torch.float64)
        self.offset = torch.tensor([leg[4] for leg in LEGS], dtype=torch.float64)

    @staticmethod
    def _base_pose(command: torch.Tensor, t: torch.Tensor):
        vx, vy, wz = command[..., 0], command[..., 1], command[..., 2]
        yaw = wz * t
        small = wz.abs() < 1e-9
        safe_wz = torch.where(small, torch.ones_like(wz), wz)
        x = (vx * torch.sin(yaw) + vy * (torch.cos(yaw) - 1.0)) / safe_wz
        y = (vx * (1.0 - torch.cos(yaw)) + vy * torch.sin(yaw)) / safe_wz
        return torch.where(small, vx * t, x), torch.where(small, vy * t, y), yaw

    def _foothold(self, command: torch.Tensor, stride: torch.Tensor) -> torch.Tensor:
        t_mid = (stride - self.offset + 0.5 * self.beta) * self.period
        x, y, yaw = self._base_pose(command.unsqueeze(-2), t_mid)
        cosine, sine = torch.cos(yaw), torch.sin(yaw)
        neutral_x, neutral_y = self.hx, self.hy + self.side * D_HIP
        return torch.stack(
            [
                x + cosine * neutral_x - sine * neutral_y,
                y + sine * neutral_x + cosine * neutral_y,
            ],
            dim=-1,
        )

    def _foot_world(
        self, command: torch.Tensor, t: torch.Tensor, swing_height: torch.Tensor
    ) -> torch.Tensor:
        u = t.unsqueeze(-1) / self.period + self.offset
        stride = torch.floor(u)
        phase = u - stride
        foothold0 = self._foothold(command, stride)
        foothold1 = self._foothold(command, stride + 1.0)
        swing = ((phase - self.beta) / (1.0 - self.beta)).unsqueeze(-1)
        xy_swing = foothold0 + (foothold1 - foothold0) * 0.5 * (
            1.0 - torch.cos(np.pi * swing)
        )
        z_swing = FOOT_R + swing_height.unsqueeze(-1) * torch.sin(
            np.pi * swing.squeeze(-1)
        )
        stance = phase < self.beta
        xy = torch.where(stance.unsqueeze(-1), foothold0, xy_swing)
        z = torch.where(stance, torch.full_like(z_swing, FOOT_R), z_swing)
        return torch.cat([xy, z.unsqueeze(-1)], dim=-1)

    @staticmethod
    def _leg_ik(position: torch.Tensor, side: torch.Tensor) -> torch.Tensor:
        px, py, pz = position[..., 0], position[..., 1], position[..., 2]
        d = side * D_HIP
        radius = torch.hypot(py, pz)
        th1 = torch.atan2(pz, py) + torch.acos(torch.clamp(d / radius, -1.0, 1.0))
        qz = -torch.sqrt(torch.clamp(radius * radius - d * d, min=1e-12))
        c3 = torch.clamp(
            (px * px + qz * qz - L1**2 - L2**2) / (2 * L1 * L2),
            -1.0,
            1.0,
        )
        ph3 = -torch.acos(c3)
        th2 = torch.atan2(-px, -qz) - torch.atan2(
            L2 * torch.sin(ph3), L1 + L2 * torch.cos(ph3)
        )
        return torch.stack([th1, th2, ph3 - DELTA], dim=-1)

    def qpos_at(
        self, command: torch.Tensor, t: torch.Tensor, swing_height: torch.Tensor
    ) -> torch.Tensor:
        x, y, yaw = self._base_pose(command, t)
        cosine, sine = torch.cos(yaw), torch.sin(yaw)
        hip_x = x.unsqueeze(-1) + cosine.unsqueeze(-1) * self.hx - sine.unsqueeze(-1) * self.hy
        hip_y = y.unsqueeze(-1) + sine.unsqueeze(-1) * self.hx + cosine.unsqueeze(-1) * self.hy
        foot = self._foot_world(command, t, swing_height)
        dx, dy = foot[..., 0] - hip_x, foot[..., 1] - hip_y
        dz = foot[..., 2] - self.height
        local = torch.stack(
            [
                cosine.unsqueeze(-1) * dx + sine.unsqueeze(-1) * dy,
                -sine.unsqueeze(-1) * dx + cosine.unsqueeze(-1) * dy,
                dz,
            ],
            dim=-1,
        )
        angles = self._leg_ik(local, self.side)
        qpos = torch.zeros(command.shape[0], 19, dtype=torch.float64)
        qpos[:, 0], qpos[:, 1], qpos[:, 2] = x, y, self.height
        qpos[:, 3], qpos[:, 6] = torch.cos(yaw / 2), torch.sin(yaw / 2)
        qpos[:, 7:] = angles.reshape(-1, 12)
        return qpos

    def qvel_at(
        self,
        command: torch.Tensor,
        t: torch.Tensor,
        swing_height: torch.Tensor,
        eps: float = 1e-4,
    ) -> torch.Tensor:
        yaw = command[:, 2] * t
        cosine, sine = torch.cos(yaw), torch.sin(yaw)
        qvel = torch.zeros(command.shape[0], 18, dtype=torch.float64)
        qvel[:, 0] = cosine * command[:, 0] - sine * command[:, 1]
        qvel[:, 1] = sine * command[:, 0] + cosine * command[:, 1]
        qvel[:, 5] = command[:, 2]
        qpos0 = self.qpos_at(command, t - eps, swing_height)
        qpos1 = self.qpos_at(command, t + eps, swing_height)
        qvel[:, 6:] = (qpos1[:, 7:] - qpos0[:, 7:]) / (2 * eps)
        return qvel

    def joint_cycle(
        self, command: torch.Tensor, swing_height: torch.Tensor, dt: float
    ) -> np.ndarray:
        """Return one gait cycle in MuJoCo qpos joint order."""
        times = torch.arange(CYC, dtype=torch.float64) * dt
        commands = command.reshape(1, 3).expand(CYC, -1)
        swings = swing_height.reshape(1).expand(CYC)
        return self.qpos_at(commands, times, swings)[:, 7:].numpy()


# ---------------------------------------------------------------------------
# CPU MuJoCo environment


class Go2CpuEnv:
    # qpos joint order: FL, FR, RL, RR.  Actuator/sensor order: FR, FL, RR, RL.
    PERM = np.asarray([3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8], dtype=np.int64)
    JOINT_SENSOR_NAMES = [
        "FR_hip",
        "FR_thigh",
        "FR_calf",
        "FL_hip",
        "FL_thigh",
        "FL_calf",
        "RR_hip",
        "RR_thigh",
        "RR_calf",
        "RL_hip",
        "RL_thigh",
        "RL_calf",
    ]

    def __init__(
        self,
        xml_path: Path,
        max_steps: int = 0,
        preset: str = "balanced",
        reset_noise: float = 0.0,
        seed: int = 0,
        actor_obs_dim: int = BASE_OBS_DIM,
        prior_factor: float = 0.0,
        terrain_type: str = TERRAIN_TYPE_NAMES[0],
        terrain_level: int = 0,
        terrain_seed: int = 4321,
    ):
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.terrain_bank = build_apex_terrain_bank(terrain_seed)
        self.has_terrain = install_terrain_bank(self.model, self.terrain_bank)
        self.data = mujoco.MjData(self.model)
        self.max_steps = max_steps
        self.reset_noise = reset_noise
        self.rng = np.random.default_rng(seed)
        if actor_obs_dim not in SUPPORTED_OBS_DIMS:
            raise ValueError(
                f"Unsupported actor observation dimension {actor_obs_dim}; "
                f"expected one of {SUPPORTED_OBS_DIMS}"
            )
        if not 0.0 <= prior_factor <= 1.0:
            raise ValueError("prior_factor must be between 0 and 1")
        self.actor_obs_dim = actor_obs_dim
        self.prior_factor = float(prior_factor)
        self.terrain_type = 0
        self.terrain_level = 0
        if self.has_terrain:
            self.set_terrain(terrain_type, terrain_level)

        if self.model.nu != ACT_DIM:
            raise ValueError(f"Expected {ACT_DIM} actuators, found {self.model.nu}")

        self.model.actuator_ctrllimited[:] = 0
        self.model.actuator_ctrlrange[:] = np.asarray([-1e6, 1e6])
        if preset == "balanced":
            self.model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
            self.model.opt.iterations = 8
            self.model.opt.ls_iterations = 8
        elif preset == "fast":
            self.model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
            self.model.opt.iterations = 1
            self.model.opt.ls_iterations = 4
            self.model.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
            self.model.opt.impratio = 1.0
        elif preset != "exact":
            raise ValueError(f"Unknown physics preset: {preset}")

        # Match the contact path used during MuJoCo-Warp training.
        for flag_name in ("mjDSBL_MULTICCD", "mjDSBL_NATIVECCD"):
            flag = getattr(mujoco.mjtDisableBit, flag_name, None)
            if flag is not None:
                self.model.opt.disableflags |= flag

        self.frame_skip = int(round(CONTROL_DT / self.model.opt.timestep))
        self.dt = self.frame_skip * self.model.opt.timestep
        if self.frame_skip < 1 or not np.isclose(self.dt, CONTROL_DT, atol=1e-8):
            raise ValueError(
                f"XML timestep {self.model.opt.timestep:g} cannot produce exactly "
                f"{CONTROL_HZ:g} Hz; computed dt={self.dt:g}"
            )

        self.a_pos = self._sensor_address(f"{self.JOINT_SENSOR_NAMES[0]}_pos")
        self.a_vel = self._sensor_address(f"{self.JOINT_SENSOR_NAMES[0]}_vel")
        self.a_gyro = self._sensor_address("imu_gyro")
        self.a_acc = self._sensor_address("imu_acc")

        self.command = np.zeros(3, dtype=np.float32)
        self.swing_height = 0.0
        self.trot = BatchTrot()
        self.phase_offset = 0
        self.step_count = 0
        self.q_table = np.zeros((CYC, ACT_DIM), dtype=np.float64)

        standing = standing_pose(H0)
        self.default_sensor_position = standing[self.PERM]
        self.joint_low = self.model.jnt_range[1:, 0][self.PERM].copy()
        self.joint_high = self.model.jnt_range[1:, 1][self.PERM].copy()
        self.target = self.default_sensor_position.copy()

    def _sensor_address(self, name: str) -> int:
        sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        if sensor_id < 0:
            raise ValueError(f"Required sensor not found in XML: {name}")
        return int(self.model.sensor_adr[sensor_id])

    @property
    def terrain_name(self) -> str:
        if not self.has_terrain:
            return "flat"
        return TERRAIN_TYPE_NAMES[self.terrain_type]

    def set_terrain(self, terrain_type: str | int, terrain_level: int) -> None:
        if not self.has_terrain:
            return
        if isinstance(terrain_type, str):
            try:
                terrain_index = TERRAIN_TYPE_NAMES.index(terrain_type)
            except ValueError as error:
                raise ValueError(f"Unknown terrain type: {terrain_type}") from error
        else:
            terrain_index = int(terrain_type)
        if not 0 <= terrain_index < TERRAIN_TYPES:
            raise ValueError(f"terrain type must be between 0 and {TERRAIN_TYPES - 1}")
        if not 0 <= int(terrain_level) < TERRAIN_LEVELS:
            raise ValueError(f"terrain level must be between 0 and {TERRAIN_LEVELS - 1}")
        self.terrain_type = terrain_index
        self.terrain_level = int(terrain_level)

    def _terrain_origin(self) -> np.ndarray:
        if not self.has_terrain:
            return np.zeros(2, dtype=np.float64)
        return np.asarray(
            [
                TERRAIN_X_CENTERS[self.terrain_level],
                TERRAIN_Y_CENTERS[self.terrain_type],
            ],
            dtype=np.float64,
        )

    def _ground_height(self) -> float:
        if not self.has_terrain:
            return 0.0
        local = self.data.qpos[:2] - self._terrain_origin()
        ix = int(
            np.clip(
                np.rint((local[0] + TERRAIN_PATCH_SIZE / 2) / TERRAIN_GRID_RES),
                0,
                TERRAIN_GRID_N - 1,
            )
        )
        iy = int(
            np.clip(
                np.rint((local[1] + TERRAIN_PATCH_SIZE / 2) / TERRAIN_GRID_RES),
                0,
                TERRAIN_GRID_N - 1,
            )
        )
        return float(self.terrain_bank[self.terrain_level, self.terrain_type, iy, ix])

    def set_command(self, command: np.ndarray) -> None:
        command = np.asarray(command, dtype=np.float32).reshape(3).copy()
        np.clip(command, -CMD_SCALE, CMD_SCALE, out=command)
        if np.linalg.norm(command / CMD_SCALE) < 0.05:
            command[:] = 0.0
        self.command[:] = command
        self.swing_height = H_SWING if np.any(command != 0.0) else 0.0
        command_tensor = torch.as_tensor(self.command, dtype=torch.float64)
        swing_tensor = torch.tensor(self.swing_height, dtype=torch.float64)
        self.q_table[:] = self.trot.joint_cycle(
            command_tensor, swing_tensor, self.dt
        )

    def reset(self, command: np.ndarray) -> torch.Tensor:
        self.set_command(command)
        self.step_count = 0
        self.phase_offset = int(self.rng.integers(CYC))

        command_tensor = torch.as_tensor(self.command, dtype=torch.float64).reshape(1, 3)
        time_tensor = torch.tensor([self.phase_offset * self.dt], dtype=torch.float64)
        swing_tensor = torch.tensor([self.swing_height], dtype=torch.float64)
        qpos = self.trot.qpos_at(command_tensor, time_tensor, swing_tensor)[0].numpy()
        qvel = self.trot.qvel_at(command_tensor, time_tensor, swing_tensor)[0].numpy()
        qpos[:2] = self._terrain_origin()
        qpos[2] = H0
        if self.reset_noise > 0:
            qpos[7:] += self.rng.uniform(-self.reset_noise, self.reset_noise, 12)

        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        self.data.qacc_warmstart[:] = 0.0
        self.data.ctrl[:] = 0.0
        self.target[:] = qpos[7:][self.PERM]
        mujoco.mj_forward(self.model, self.data)
        return self.observation()

    def phase(self) -> int:
        return (self.phase_offset + self.step_count) % CYC

    def observation(self) -> torch.Tensor:
        sensor = self.data.sensordata
        angle = sensor[self.a_pos : self.a_pos + 12] - self.default_sensor_position
        velocity = sensor[self.a_vel : self.a_vel + 12]
        phase = 2.0 * np.pi * self.phase() / CYC
        parts = [
            np.stack([angle, velocity], axis=-1).reshape(24),
            sensor[self.a_gyro : self.a_gyro + 3],
            sensor[self.a_acc : self.a_acc + 3],
            self.command / CMD_SCALE,
            np.asarray([np.sin(phase), np.cos(phase)]),
        ]
        if self.actor_obs_dim == PRIOR_OBS_DIM:
            parts.append(np.asarray([self.prior_factor]))
        observation = np.concatenate(parts).astype(np.float32, copy=False)
        if observation.shape != (self.actor_obs_dim,):
            raise RuntimeError(
                f"Expected observation shape {(self.actor_obs_dim,)}, "
                f"got {observation.shape}"
            )
        return torch.from_numpy(observation.copy()).unsqueeze(0)

    def step(self, action: torch.Tensor):
        action_np = action.detach().cpu().numpy().reshape(ACT_DIM)
        prior = 0.0
        if self.actor_obs_dim == PRIOR_OBS_DIM and self.prior_factor > 0.0:
            reference_sensor = self.q_table[self.phase()][self.PERM]
            prior = self.prior_factor * (
                reference_sensor - self.default_sensor_position
            )
        self.target[:] = np.clip(
            self.default_sensor_position + action_np * ACT_SCALE + prior,
            self.joint_low,
            self.joint_high,
        )

        for _ in range(self.frame_skip):
            joint_position = self.data.qpos[7:][self.PERM]
            joint_velocity = self.data.qvel[6:][self.PERM]
            self.data.ctrl[:] = (
                SERVO_KP * (self.target - joint_position) - SERVO_KD * joint_velocity
            )
            mujoco.mj_step(self.model, self.data)

        self.step_count += 1
        relative_height = self.data.qpos[2] - self._ground_height()
        fallen = bool(
            relative_height < 0.23
            or np.hypot(self.data.qpos[4], self.data.qpos[5]) > 0.25
        )
        timeout = self.max_steps > 0 and self.step_count >= self.max_steps
        return self.observation(), fallen or timeout, fallen


# ---------------------------------------------------------------------------
# Actor: names and operations match the saved notebook checkpoint


class PopSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, probability, noise, slope, threshold):
        del ctx, slope
        value = probability.unsqueeze(-1) + noise
        return value.gt(threshold).to(probability.dtype).reshape(probability.shape[0], -1)

    @staticmethod
    def backward(ctx, grad_output):
        del ctx, grad_output
        raise RuntimeError("Backward is not used by the viewer")


class SpikeFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, slope):
        del ctx, slope
        return (value > 0).to(value.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        del ctx, grad_output
        raise RuntimeError("Backward is not used by the viewer")


def lif_step(current: torch.Tensor, previous: torch.Tensor):
    reset = (previous > LIF_THRESHOLD).to(previous.dtype)
    membrane = LIF_BETA * previous + current - reset * LIF_THRESHOLD
    return SpikeFn.apply(membrane - LIF_THRESHOLD, SPIKE_SLOPE), membrane


class Encoder(nn.Module):
    def __init__(self, obs_dim: int):
        super().__init__()
        self.obs_dim = obs_dim
        self.activation = nn.Tanh()
        self.register_buffer("zeros", torch.zeros(1, obs_dim * 2), persistent=False)
        self.weight = nn.Parameter(torch.ones(1, obs_dim))
        self.bias = nn.Parameter(torch.zeros(1, obs_dim))

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        value = self.activation(observation * self.weight + self.bias)
        probability = torch.maximum(torch.cat([value, -value], dim=1), self.zeros)
        noise = torch.rand(
            probability.shape[0],
            self.obs_dim * 2,
            ENCODER_POP_DIM,
            device=probability.device,
            dtype=probability.dtype,
        )
        return PopSpike.apply(probability, noise, SPIKE_SLOPE, ENC_VTH)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.activation = nn.Tanh()
        self.weight = nn.Parameter(torch.ones(1, ACT_DIM), requires_grad=False)
        self.bias = nn.Parameter(torch.zeros(1, ACT_DIM), requires_grad=False)

    def forward(self, spikes: torch.Tensor) -> torch.Tensor:
        rates = spikes.reshape(-1, ACT_DIM * 2, DECODER_POP_DIM).sum(-1)
        rates = rates / DECODER_POP_DIM
        return self.activation(
            (rates[..., :ACT_DIM] - rates[..., ACT_DIM:]) * self.weight + self.bias
        )


class SpikeActor(nn.Module):
    def __init__(self, obs_dim: int):
        super().__init__()
        self.obs_dim = obs_dim
        self.Linear1 = nn.Linear(obs_dim * 2 * ENCODER_POP_DIM, HIDDEN_SIZES[0])
        self.Linear2 = nn.Linear(HIDDEN_SIZES[0], HIDDEN_SIZES[1])
        self.Linear3 = nn.Linear(HIDDEN_SIZES[1], HIDDEN_SIZES[2])
        self.encoder = Encoder(obs_dim)
        self.decoder = Decoder()
        self.p1 = HIDDEN_SIZES[0]
        self.p2 = HIDDEN_SIZES[0] + HIDDEN_SIZES[1]

    def forward(self, observation: torch.Tensor, membrane: torch.Tensor):
        spike1, mem1 = lif_step(
            self.Linear1(self.encoder(observation)), membrane[:, : self.p1]
        )
        spike2, mem2 = lif_step(
            self.Linear2(spike1), membrane[:, self.p1 : self.p2]
        )
        spike3, mem3 = lif_step(self.Linear3(spike2), membrane[:, self.p2 :])
        return self.decoder(spike3), torch.cat([mem1, mem2, mem3], dim=1)


def load_actor(checkpoint: Path) -> SpikeActor:
    try:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch older than weights_only support
        state = torch.load(checkpoint, map_location="cpu")
    if isinstance(state, dict) and "actor" in state:
        state = state["actor"]
    state = {
        key.removeprefix("_orig_mod."): value for key, value in dict(state).items()
    }
    for key in [name for name in state if name.startswith("lif")]:
        state.pop(key)
    encoder_weight = state.get("encoder.weight")
    if encoder_weight is None or encoder_weight.ndim != 2:
        raise RuntimeError("Checkpoint does not contain a valid encoder.weight")
    obs_dim = int(encoder_weight.shape[1])
    if obs_dim not in SUPPORTED_OBS_DIMS:
        raise RuntimeError(
            f"Checkpoint actor observation dimension is {obs_dim}; "
            f"supported dimensions are {SUPPORTED_OBS_DIMS}"
        )
    actor = SpikeActor(obs_dim)
    actor.load_state_dict(state)
    actor.eval()
    return actor


def load_normalizer(path: Path, actor_obs_dim: int):
    values = np.load(path)
    mean_values = np.asarray(values["mean"]).reshape(-1)
    variance_values = np.asarray(values["var"]).reshape(-1)
    if mean_values.size < actor_obs_dim or variance_values.size < actor_obs_dim:
        raise ValueError(
            f"Normalizer has {mean_values.size} mean and {variance_values.size} "
            f"variance values, but actor requires {actor_obs_dim}"
        )
    mean = torch.as_tensor(mean_values[:actor_obs_dim], dtype=torch.float32)
    variance = torch.as_tensor(variance_values[:actor_obs_dim], dtype=torch.float32)

    def normalize(observation: torch.Tensor) -> torch.Tensor:
        eps = torch.finfo(torch.float32).eps
        normalized = torch.clamp(
            (observation - mean) / torch.sqrt(variance + eps),
            -NORM_CLIP_LIMIT,
            NORM_CLIP_LIMIT,
        )
        # The new 36-D policy receives the raw action-prior coefficient.  The
        # training notebook restores this column after normalizing actor input.
        if actor_obs_dim == PRIOR_OBS_DIM:
            normalized = torch.cat([normalized[:, :-1], observation[:, -1:]], dim=1)
        return normalized

    return normalize


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the trained Go2 policy at 50 Hz in a WSLg MuJoCo window"
    )
    parser.add_argument("--xml", default="scene_apex_curriculum.xml")
    parser.add_argument(
        "--checkpoint", default="params_imit_warp_0/model_imit_10Kit.pt"
    )
    parser.add_argument(
        "--norm", default="params_imit_warp_0/model_imit_mean_var_10Kit.npz"
    )
    parser.add_argument(
        "--preset", choices=["exact", "balanced", "fast"], default="balanced"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="reset after this many control steps; 0 disables timeout resets",
    )
    parser.add_argument("--reset-noise", type=float, default=0.0)
    parser.add_argument(
        "--terrain-type",
        choices=TERRAIN_TYPE_NAMES,
        default=TERRAIN_TYPE_NAMES[0],
    )
    parser.add_argument(
        "--terrain-level",
        type=int,
        choices=range(TERRAIN_LEVELS),
        default=0,
        metavar="0..9",
    )
    parser.add_argument("--terrain-seed", type=int, default=4321)
    parser.add_argument(
        "--camera",
        choices=("side", "track", "free"),
        default="side",
        help="side and track cameras follow the robot",
    )
    parser.add_argument(
        "--prior-factor",
        type=float,
        default=0.0,
        help=(
            "analytic gait assistance for the 36-D policy (0=evaluation default, "
            "1=full assistance)"
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--report-interval", type=float, default=2.0)
    parser.add_argument(
        "--start",
        type=float,
        nargs=3,
        metavar=("VX", "VY", "WZ"),
        default=(0.0, 0.0, 0.0),
    )
    parser.add_argument("--vx-step", type=float, default=0.05)
    parser.add_argument("--vy-step", type=float, default=0.05)
    parser.add_argument("--wz-step", type=float, default=0.10)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        raise RuntimeError(
            "DISPLAY is not set. Run this from the local WSL2/WSLg terminal, not over "
            "the GPU server SSH session."
        )
    if args.torch_threads < 1:
        raise ValueError("--torch-threads must be at least 1")
    if not 0.0 <= args.prior_factor <= 1.0:
        raise ValueError("--prior-factor must be between 0 and 1")

    xml_path = resolve_input_path(args.xml)
    checkpoint_path = resolve_input_path(args.checkpoint)
    norm_path = resolve_input_path(args.norm)
    for label, path in (
        ("XML", xml_path),
        ("checkpoint", checkpoint_path),
        ("normalizer", norm_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} file not found: {path}")

    torch.set_num_threads(args.torch_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.manual_seed(args.seed)

    actor = load_actor(checkpoint_path)
    normalize = load_normalizer(norm_path, actor.obs_dim)
    env = Go2CpuEnv(
        xml_path,
        max_steps=args.max_steps,
        preset=args.preset,
        reset_noise=args.reset_noise,
        seed=args.seed,
        actor_obs_dim=actor.obs_dim,
        prior_factor=args.prior_factor,
        terrain_type=args.terrain_type,
        terrain_level=args.terrain_level,
        terrain_seed=args.terrain_seed,
    )

    state_lock = threading.Lock()
    state = {
        "command": np.clip(
            np.asarray(args.start, dtype=np.float32), -CMD_SCALE, CMD_SCALE
        ),
        "changed": True,
        "reset": True,
        "paused": False,
        "terrain_type": env.terrain_type,
        "terrain_level": env.terrain_level,
    }

    def print_command(prefix: str = "command") -> None:
        command = state["command"]
        print(
            f"{prefix}: vx={command[0]:+.2f} m/s  vy={command[1]:+.2f} m/s  "
            f"wz={command[2]:+.2f} rad/s",
            flush=True,
        )

    def print_terrain() -> None:
        if env.has_terrain:
            terrain_name = TERRAIN_TYPE_NAMES[state["terrain_type"]]
            print(
                f"terrain: {terrain_name}  level={state['terrain_level']}",
                flush=True,
            )
        else:
            print("terrain: flat (terrain selection is unavailable)", flush=True)

    def key_callback(keycode: int) -> None:
        try:
            key = chr(keycode).upper()
        except (ValueError, OverflowError):
            return
        with state_lock:
            command = state["command"]
            if key == "W":
                command[0] += args.vx_step
            elif key == "S":
                command[0] -= args.vx_step
            elif key == "A":
                command[1] += args.vy_step
            elif key == "D":
                command[1] -= args.vy_step
            elif key == "Q":
                command[2] += args.wz_step
            elif key == "E":
                command[2] -= args.wz_step
            elif key == "X":
                command[:] = 0.0
            elif key == "1":
                command[:] = (CMD_SCALE[0], 0.0, 0.0)
            elif key == "2":
                command[:] = (-CMD_SCALE[0], 0.0, 0.0)
            elif key == "3":
                command[:] = (0.0, CMD_SCALE[1], 0.0)
            elif key == "4":
                command[:] = (0.0, -CMD_SCALE[1], 0.0)
            elif key == "5":
                command[:] = (0.0, 0.0, CMD_SCALE[2])
            elif key == "6":
                command[:] = (0.0, 0.0, -CMD_SCALE[2])
            elif key == "R":
                state["reset"] = True
                return
            elif key == "T":
                if not env.has_terrain:
                    print_terrain()
                    return
                state["terrain_type"] = (
                    state["terrain_type"] + 1
                ) % TERRAIN_TYPES
                state["reset"] = True
                print_terrain()
                return
            elif key == "[":
                if not env.has_terrain:
                    print_terrain()
                    return
                state["terrain_level"] = max(0, state["terrain_level"] - 1)
                state["reset"] = True
                print_terrain()
                return
            elif key == "]":
                if not env.has_terrain:
                    print_terrain()
                    return
                state["terrain_level"] = min(
                    TERRAIN_LEVELS - 1, state["terrain_level"] + 1
                )
                state["reset"] = True
                print_terrain()
                return
            elif key in ("P", " "):
                state["paused"] = not state["paused"]
                print("paused" if state["paused"] else "running", flush=True)
                return
            else:
                return
            np.clip(command, -CMD_SCALE, CMD_SCALE, out=command)
            state["changed"] = True
            print_command()

    print(
        "Keys: W/S=vx  A/D=vy  Q/E=yaw  X=stop  "
        "1..6=direction presets  T=terrain  [ / ]=level  "
        "R=reset  Space/P=pause  Esc=quit",
        flush=True,
    )
    print_terrain()
    print(
        f"Limits: vx=+-{CMD_SCALE[0]:.2f}, vy=+-{CMD_SCALE[1]:.2f}, "
        f"wz=+-{CMD_SCALE[2]:.2f}; control={CONTROL_HZ:.0f} Hz; device=CPU; "
        f"actor_obs={actor.obs_dim}; prior={args.prior_factor:.2f}",
        flush=True,
    )

    command = state["command"].copy()
    observation = env.reset(command)
    membrane = torch.randn(1, MEM_DIM, dtype=torch.float32)

    with mujoco.viewer.launch_passive(
        env.model,
        env.data,
        key_callback=key_callback,
        show_left_ui=True,
        show_right_ui=True,
    ) as viewer:
        camera_id = (
            -1
            if args.camera == "free"
            else mujoco.mj_name2id(
                env.model, mujoco.mjtObj.mjOBJ_CAMERA, args.camera
            )
        )
        if camera_id >= 0 and args.camera != "free":
            with viewer.lock():
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                viewer.cam.fixedcamid = camera_id

        next_tick = time.perf_counter()
        report_start = next_tick
        report_steps = 0

        while viewer.is_running():
            with state_lock:
                command = state["command"].copy()
                changed = state["changed"]
                reset = state["reset"]
                paused = state["paused"]
                state["changed"] = False
                state["reset"] = False

            if reset:
                with viewer.lock():
                    env.set_terrain(
                        state["terrain_type"], state["terrain_level"]
                    )
                    observation = env.reset(command)
                membrane = torch.randn(1, MEM_DIM, dtype=torch.float32)
                print_command("reset")
            elif changed:
                env.set_command(command)
                observation = env.observation()

            if paused:
                viewer.sync()
                next_tick = time.perf_counter()
                report_start = next_tick
                report_steps = 0
                time.sleep(0.01)
                continue

            with torch.no_grad():
                action, membrane = actor(normalize(observation), membrane)
                action = action.clamp(-1.0, 1.0)

            with viewer.lock():
                observation, done, fallen = env.step(action)
            if done:
                if fallen:
                    print("fallen -> automatic reset", flush=True)
                with viewer.lock():
                    observation = env.reset(command)
                membrane = torch.randn(1, MEM_DIM, dtype=torch.float32)

            viewer.sync()
            report_steps += 1
            now = time.perf_counter()
            if args.report_interval > 0 and now - report_start >= args.report_interval:
                actual_hz = report_steps / (now - report_start)
                print(f"control: {actual_hz:5.1f} Hz", flush=True)
                report_start, report_steps = now, 0

            next_tick += env.dt
            delay = next_tick - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            elif delay < -env.dt:
                # Do not create a catch-up burst after a slow render or window drag.
                next_tick = time.perf_counter()


if __name__ == "__main__":
    main()
