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
# Constants copied from go2_imitation_warp_release(1).ipynb

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

OBS_DIM, ACT_DIM = 35, 12
ENCODER_POP_DIM, DECODER_POP_DIM = 64, 256
HIDDEN_SIZES = (256, 256, ACT_DIM * 2 * DECODER_POP_DIM)
MEM_DIM = sum(HIDDEN_SIZES)
NORM_CLIP_LIMIT = 50.0

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
    ):
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        self.max_steps = max_steps
        self.reset_noise = reset_noise
        self.rng = np.random.default_rng(seed)

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

    def set_command(self, command: np.ndarray) -> None:
        command = np.asarray(command, dtype=np.float32).reshape(3).copy()
        np.clip(command, -CMD_SCALE, CMD_SCALE, out=command)
        if np.linalg.norm(command / CMD_SCALE) < 0.05:
            command[:] = 0.0
        self.command[:] = command
        self.swing_height = H_SWING if np.any(command != 0.0) else 0.0

    def reset(self, command: np.ndarray) -> torch.Tensor:
        self.set_command(command)
        self.step_count = 0
        self.phase_offset = int(self.rng.integers(CYC))

        command_tensor = torch.as_tensor(self.command, dtype=torch.float64).reshape(1, 3)
        time_tensor = torch.tensor([self.phase_offset * self.dt], dtype=torch.float64)
        swing_tensor = torch.tensor([self.swing_height], dtype=torch.float64)
        qpos = self.trot.qpos_at(command_tensor, time_tensor, swing_tensor)[0].numpy()
        qvel = self.trot.qvel_at(command_tensor, time_tensor, swing_tensor)[0].numpy()
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
        observation = np.concatenate(
            [
                np.stack([angle, velocity], axis=-1).reshape(24),
                sensor[self.a_gyro : self.a_gyro + 3],
                sensor[self.a_acc : self.a_acc + 3],
                self.command / CMD_SCALE,
                np.asarray([np.sin(phase), np.cos(phase)]),
            ]
        ).astype(np.float32, copy=False)
        if observation.shape != (OBS_DIM,):
            raise RuntimeError(f"Expected observation shape {(OBS_DIM,)}, got {observation.shape}")
        return torch.from_numpy(observation.copy()).unsqueeze(0)

    def step(self, action: torch.Tensor):
        action_np = action.detach().cpu().numpy().reshape(ACT_DIM)
        self.target[:] = np.clip(
            self.default_sensor_position + action_np * ACT_SCALE,
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
        fallen = bool(
            self.data.qpos[2] < 0.23
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
    def __init__(self):
        super().__init__()
        self.activation = nn.Tanh()
        self.register_buffer("zeros", torch.zeros(1, OBS_DIM * 2), persistent=False)
        self.weight = nn.Parameter(torch.ones(1, OBS_DIM))
        self.bias = nn.Parameter(torch.zeros(1, OBS_DIM))

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        value = self.activation(observation * self.weight + self.bias)
        probability = torch.maximum(torch.cat([value, -value], dim=1), self.zeros)
        noise = torch.rand(
            probability.shape[0],
            OBS_DIM * 2,
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
    def __init__(self):
        super().__init__()
        self.Linear1 = nn.Linear(OBS_DIM * 2 * ENCODER_POP_DIM, HIDDEN_SIZES[0])
        self.Linear2 = nn.Linear(HIDDEN_SIZES[0], HIDDEN_SIZES[1])
        self.Linear3 = nn.Linear(HIDDEN_SIZES[1], HIDDEN_SIZES[2])
        self.encoder = Encoder()
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
    actor = SpikeActor()
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
    actor.load_state_dict(state)
    actor.eval()
    return actor


def load_normalizer(path: Path):
    values = np.load(path)
    mean = torch.as_tensor(values["mean"], dtype=torch.float32)
    variance = torch.as_tensor(values["var"], dtype=torch.float32)

    def normalize(observation: torch.Tensor) -> torch.Tensor:
        eps = torch.finfo(torch.float32).eps
        return torch.clamp(
            (observation - mean) / torch.sqrt(variance + eps),
            -NORM_CLIP_LIMIT,
            NORM_CLIP_LIMIT,
        )

    return normalize


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the trained Go2 policy at 50 Hz in a WSLg MuJoCo window"
    )
    parser.add_argument("--xml", default="scene_flat.xml")
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
    normalize = load_normalizer(norm_path)
    env = Go2CpuEnv(
        xml_path,
        max_steps=args.max_steps,
        preset=args.preset,
        reset_noise=args.reset_noise,
        seed=args.seed,
    )

    state_lock = threading.Lock()
    state = {
        "command": np.clip(
            np.asarray(args.start, dtype=np.float32), -CMD_SCALE, CMD_SCALE
        ),
        "changed": True,
        "reset": True,
        "paused": False,
    }

    def print_command(prefix: str = "command") -> None:
        command = state["command"]
        print(
            f"{prefix}: vx={command[0]:+.2f} m/s  vy={command[1]:+.2f} m/s  "
            f"wz={command[2]:+.2f} rad/s",
            flush=True,
        )

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
        "1..6=direction presets  R=reset  Space/P=pause  Esc=quit",
        flush=True,
    )
    print(
        f"Limits: vx=+-{CMD_SCALE[0]:.2f}, vy=+-{CMD_SCALE[1]:.2f}, "
        f"wz=+-{CMD_SCALE[2]:.2f}; control={CONTROL_HZ:.0f} Hz; device=CPU",
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
        camera_id = mujoco.mj_name2id(
            env.model, mujoco.mjtObj.mjOBJ_CAMERA, "track"
        )
        if camera_id >= 0:
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
