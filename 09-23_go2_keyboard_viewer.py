#!/usr/bin/env python3
"""WSL/WSLgでGo2の学習済みSNNをキーボード操作する単体ビューア。

平地・最大2 cm・5 cm・10 cmの地形を実行中に切り替えられる。
物理計算はCPU、SNN推論はCUDAがあればGPUを使用する。
"""


import os
# 学習ノートの EGL ではなく、WSLg の対話ウィンドウを使う。
# この設定は mujoco を import する前に必要。
os.environ["MUJOCO_GL"] = "glfw"
os.environ.setdefault("GALLIUM_DRIVER", "d3d12")
os.environ.pop("PYOPENGL_PLATFORM", None)

import re
import time
from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image
import torch
from torch import nn
import glfw
import mujoco

# ---------- ユーザー設定 ----------
WORK_DIR = Path(__file__).resolve().parent    # scene_flat.xml があるフォルダ
FLAT_SCENE = WORK_DIR / "scene_flat.xml"

# None の場合は WORK_DIR/params_* から自動検索する。
PARAMS_DIR = "params_09-23_fixed"
BASE_ACTOR_CHECKPOINT = None                  # 例: WORK_DIR / "params_09-22/go2_imitation_10Kit.pt"
BASE_NORM_CHECKPOINT = None                   # 例: WORK_DIR / "params_09-22/go2_imitation_norm_10Kit.npz"
RESIDUAL_CHECKPOINT = None                    # 例: WORK_DIR / "params_09-22/go2_residual_lidar_stateful_10000it.pt"

BASE_MODEL_NAME = "go2_imitation"
BASE_NORM_NAME = "go2_imitation_norm"
RESIDUAL_MODEL_NAME = "go2_residual_lidar"
RESIDUAL_FALLBACK_MODEL_NAME = "go2_residual"  # 再学習前の旧checkpointも利用可能

START_TERRAIN = "easy"                       # flat / easy / medium / hard
USE_RESIDUAL_AT_START = True
TORCH_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CONTROL_SEED = 0

# ---------- 軽量表示設定 ----------
# 描画だけを15 Hzへ下げる。制御50 Hz・内部物理500 Hzは変更しない。
WINDOW_WIDTH, WINDOW_HEIGHT = 320, 180
RENDER_HZ = 15.0
VSYNC = False
TORCH_THREADS = 1
REPORT_INTERVAL = 5.0

# バッチ1のSNN推論では多数のCPUスレッドを使うと起動・同期コストが増えやすい。
torch.set_num_threads(TORCH_THREADS)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

print("作業フォルダ:", WORK_DIR.resolve())
print("推論デバイス:", TORCH_DEVICE)
print("DISPLAY:", os.environ.get("DISPLAY"), "/ WAYLAND_DISPLAY:", os.environ.get("WAYLAND_DISPLAY"))

if not FLAT_SCENE.is_file():
    raise FileNotFoundError(f"scene_flat.xml がありません: {FLAT_SCENE.resolve()}")
if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
    raise RuntimeError("WSLg の DISPLAY/WAYLAND_DISPLAY が見つかりません。WSL上のGUI対応端末から実行してください。")


@dataclass(frozen=True)
class ViewerTerrainConfig:
    half_size: float = 8.0
    max_step_height: float = 0.02
    base_depth: float = 0.10
    resolution: int = 129
    block_size: float = 0.40
    height_levels: int = 11
    flat_radius: float = 0.75
    transition_radius: float = 1.25
    seed: int = 7
    friction: tuple = (1.0, 0.005, 0.0001)


def make_heightmap(config):
    rng = np.random.default_rng(config.seed)
    metres_per_pixel = 2 * config.half_size / (config.resolution - 1)
    pixels_per_block = max(1, int(round(config.block_size / metres_per_pixel)))
    coarse_size = int(np.ceil(config.resolution / pixels_per_block))
    coarse = rng.integers(0, config.height_levels, size=(coarse_size, coarse_size))
    coarse = coarse.astype(np.float64) / (config.height_levels - 1)
    height = np.repeat(np.repeat(coarse, pixels_per_block, axis=0),
                       pixels_per_block, axis=1)[:config.resolution, :config.resolution]

    axis = np.linspace(-config.half_size, config.half_size, config.resolution)
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    radius = np.hypot(xx, yy)
    allowed = np.clip((radius - config.flat_radius) /
                      (config.transition_radius - config.flat_radius), 0.0, 1.0)
    allowed = allowed * allowed * (3.0 - 2.0 * allowed)
    height = np.minimum(height, allowed)
    return np.clip(np.rint(height * 255.0), 0, 255).astype(np.uint8)


def create_viewer_terrain(flat_xml, output_xml, config):
    flat_path = Path(flat_xml).resolve()
    output_path = Path(output_xml).resolve()
    if output_path.parent != flat_path.parent:
        raise ValueError("地形XMLはscene_flat.xmlと同じフォルダへ生成してください")

    image_path = output_path.with_suffix(".png")
    Image.fromarray(make_heightmap(config)).save(image_path)

    tree = ET.parse(flat_path)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    asset = root.find("asset")
    if worldbody is None:
        raise ValueError("scene_flat.xml に <worldbody> がありません")
    if asset is None:
        asset = ET.Element("asset")
        root.insert(list(root).index(worldbody), asset)

    for parent in root.iter():
        for element in list(parent):
            if element.tag == "hfield" and element.get("name") == "research_terrain_hf":
                parent.remove(element)
            elif element.tag == "geom" and (
                element.get("type") == "plane" or element.get("name") == "research_terrain"
            ):
                parent.remove(element)

    ET.SubElement(
        asset, "hfield", name="research_terrain_hf",
        file=image_path.resolve().as_posix(),
        size=f"{config.half_size:g} {config.half_size:g} {config.max_step_height:g} {config.base_depth:g}",
    )
    ET.SubElement(
        worldbody, "geom", name="research_terrain", type="hfield",
        hfield="research_terrain_hf", pos="0 0 0",
        friction=" ".join(f"{v:g}" for v in config.friction),
        rgba="0.32 0.38 0.28 1", contype="1", conaffinity="1",
    )
    ET.indent(tree, space="  ")
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    return output_path


SCENES = {"flat": FLAT_SCENE.resolve()}
for terrain_name, height in (("easy", 0.02), ("medium", 0.05), ("hard", 0.10)):
    SCENES[terrain_name] = create_viewer_terrain(
        FLAT_SCENE,
        WORK_DIR / f"scene_steps_{terrain_name}.xml",
        ViewerTerrainConfig(max_step_height=height),
    )

for name, path in SCENES.items():
    print(f"{name:6s}: {path}")


LIF_BETA, LIF_THRESHOLD, SPIKE_SLOPE = 0.5, 1.0, 3.0
ENC_VTH = 0.999


class PopSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, probability, noise, slope, threshold):
        voltage = probability.unsqueeze(-1) + noise
        return voltage.gt(threshold).to(probability.dtype).reshape(probability.shape[0], -1)


class SpikeFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, slope):
        return (value > 0).to(value.dtype)


def lif_step(current, previous, beta=LIF_BETA, threshold=LIF_THRESHOLD):
    reset = (previous > threshold).to(previous.dtype)
    membrane = beta * previous + current - reset * threshold
    return SpikeFn.apply(membrane - threshold, SPIKE_SLOPE), membrane


class Encoder(nn.Module):
    def __init__(self, obs_dim, pop_dim):
        super().__init__()
        self.obs_dim = obs_dim
        self.pop_dim = pop_dim
        self.weight = nn.Parameter(torch.ones(1, obs_dim))
        self.bias = nn.Parameter(torch.zeros(1, obs_dim))

    def forward(self, obs):
        obs = torch.tanh(obs * self.weight + self.bias)
        probability = torch.maximum(
            torch.cat([obs, -obs], dim=1),
            torch.zeros(1, self.obs_dim * 2, device=obs.device, dtype=obs.dtype),
        )
        noise = torch.rand(
            probability.shape[0], self.obs_dim * 2, self.pop_dim,
            device=obs.device, dtype=obs.dtype,
        )
        return PopSpike.apply(probability, noise, SPIKE_SLOPE, ENC_VTH)


class Decoder(nn.Module):
    def __init__(self, act_dim, pop_dim):
        super().__init__()
        self.act_dim = act_dim
        self.pop_dim = pop_dim
        self.weight = nn.Parameter(torch.ones(1, act_dim), requires_grad=False)
        self.bias = nn.Parameter(torch.zeros(1, act_dim), requires_grad=False)

    def forward(self, spikes):
        rates = spikes.reshape(-1, self.act_dim * 2, self.pop_dim).mean(-1)
        return torch.tanh(
            (rates[:, :self.act_dim] - rates[:, self.act_dim:]) * self.weight + self.bias
        )


class ViewerSpikeActor(nn.Module):
    def __init__(self, obs_dim, hidden1, hidden2, output_spikes, act_dim, enc_pop, dec_pop):
        super().__init__()
        self.Linear1 = nn.Linear(obs_dim * 2 * enc_pop, hidden1)
        self.Linear2 = nn.Linear(hidden1, hidden2)
        self.Linear3 = nn.Linear(hidden2, output_spikes)
        self.encoder = Encoder(obs_dim, enc_pop)
        self.decoder = Decoder(act_dim, dec_pop)
        self.p1 = hidden1
        self.p2 = hidden1 + hidden2
        self.mem_dim = hidden1 + hidden2 + output_spikes
        self.obs_dim = obs_dim
        self.act_dim = act_dim

    def forward(self, obs, membrane):
        spike1, mem1 = lif_step(self.Linear1(self.encoder(obs)), membrane[:, :self.p1])
        spike2, mem2 = lif_step(self.Linear2(spike1), membrane[:, self.p1:self.p2])
        spike3, mem3 = lif_step(self.Linear3(spike2), membrane[:, self.p2:])
        return self.decoder(spike3), torch.cat([mem1, mem2, mem3], dim=1)


def torch_load(path):
    try:
        return torch.load(path, map_location=TORCH_DEVICE, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=TORCH_DEVICE)


def clean_actor_state(state):
    cleaned = {}
    for key, value in state.items():
        key = key.removeprefix("_orig_mod.")
        if key.startswith("lif"):
            continue
        cleaned[key] = value
    return cleaned


def actor_from_state(state):
    state = clean_actor_state(state)
    obs_dim = int(state["encoder.weight"].shape[1])
    act_dim = int(state["decoder.weight"].shape[1])
    hidden1, encoded = state["Linear1.weight"].shape
    hidden2 = int(state["Linear2.weight"].shape[0])
    output_spikes = int(state["Linear3.weight"].shape[0])
    enc_pop = encoded // (obs_dim * 2)
    dec_pop = output_spikes // (act_dim * 2)
    actor = ViewerSpikeActor(
        obs_dim, hidden1, hidden2, output_spikes, act_dim, enc_pop, dec_pop
    ).to(TORCH_DEVICE)
    actor.load_state_dict(state, strict=True)
    actor.eval()
    for parameter in actor.parameters():
        parameter.requires_grad_(False)
    return actor


def numeric_suffix(path):
    values = re.findall(r"(\d+)(?:Kit|it)", path.name)
    return int(values[-1]) if values else -1


def params_directories():
    if PARAMS_DIR is not None:
        return [Path(PARAMS_DIR)]
    return sorted(WORK_DIR.glob("params_*"), key=lambda p: p.stat().st_mtime, reverse=True)


def newest(pattern):
    # 更新日時が新しい params_* を優先し、そのフォルダ内で最大ステップを選ぶ。
    for folder in params_directories():
        candidates = list(folder.glob(pattern))
        if candidates:
            return max(candidates, key=lambda p: (numeric_suffix(p), p.stat().st_mtime))
    return None


base_actor_path = Path(BASE_ACTOR_CHECKPOINT) if BASE_ACTOR_CHECKPOINT else newest(f"{BASE_MODEL_NAME}_*Kit.pt")
if base_actor_path is None or not base_actor_path.is_file():
    raise FileNotFoundError("平地actorが見つかりません。BASE_ACTOR_CHECKPOINT または PARAMS_DIR を設定してください。")

if BASE_NORM_CHECKPOINT:
    base_norm_path = Path(BASE_NORM_CHECKPOINT)
else:
    same_step = re.search(r"_(\d+Kit)\.pt$", base_actor_path.name)
    expected = base_actor_path.with_name(
        f"{BASE_NORM_NAME}_{same_step.group(1)}.npz" if same_step else ""
    )
    base_norm_path = expected if expected.is_file() else newest(f"{BASE_NORM_NAME}_*Kit.npz")

residual_path = (
    Path(RESIDUAL_CHECKPOINT)
    if RESIDUAL_CHECKPOINT
    else newest(f"{RESIDUAL_MODEL_NAME}_*_accepted.pt")
)
if residual_path is None and RESIDUAL_CHECKPOINT is None:
    residual_path = newest(f"{RESIDUAL_MODEL_NAME}_*_*it.pt")
if residual_path is None and RESIDUAL_CHECKPOINT is None:
    residual_path = newest(f"{RESIDUAL_FALLBACK_MODEL_NAME}_*_*it.pt")

base_actor = actor_from_state(torch_load(base_actor_path))
residual_checkpoint = None
residual_actor = None
if residual_path is not None and residual_path.is_file():
    residual_checkpoint = torch_load(residual_path)
    residual_actor = actor_from_state(residual_checkpoint["residual_actor"])
    if residual_actor.obs_dim != base_actor.obs_dim + base_actor.act_dim:
        raise ValueError(
            f"残差actorの入力次元が不一致です: {residual_actor.obs_dim} != "
            f"{base_actor.obs_dim} + {base_actor.act_dim}"
        )
    if residual_actor.act_dim != base_actor.act_dim:
        raise ValueError("平地actorと残差actorの行動次元が一致しません")

if residual_checkpoint is not None:
    obs_mean = residual_checkpoint["obs_mean"].to(TORCH_DEVICE).float()
    obs_var = residual_checkpoint["obs_var"].to(TORCH_DEVICE).float()
    residual_scale = float(residual_checkpoint.get("config", {}).get("residual_scale", 0.25))
    residual_limit = float(residual_checkpoint.get("config", {}).get("residual_limit", 0.50))
    residual_stateful = bool(residual_checkpoint.get("config", {}).get("stateful", True))
else:
    if base_norm_path is None or not base_norm_path.is_file():
        raise FileNotFoundError("観測正規化ファイルが見つかりません。BASE_NORM_CHECKPOINT を設定してください。")
    normalizer = np.load(base_norm_path)
    obs_mean = torch.as_tensor(normalizer["mean"], dtype=torch.float32, device=TORCH_DEVICE)
    obs_var = torch.as_tensor(normalizer["var"], dtype=torch.float32, device=TORCH_DEVICE)
    residual_scale = 0.25
    residual_limit = 0.50
    residual_stateful = True

print("平地actor:", base_actor_path)
print("正規化統計:", base_norm_path if base_norm_path else "残差checkpoint内")
print("残差actor:", residual_path if residual_actor is not None else "なし（平地actorのみ）")
print(
    f"obs={base_actor.obs_dim}, action={base_actor.act_dim}, "
    f"residual_scale={residual_scale}, residual_limit={residual_limit}"
)


H0 = 0.33
D_HIP = 0.0955
L1 = 0.213
L2 = float(np.hypot(0.002, 0.213))
DELTA = float(np.arctan2(0.002, 0.213))
FOOT_R = 0.022
LEGS = [
    ("FL", +1), ("FR", -1), ("RL", +1), ("RR", -1),
]
PERM = np.array([3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8])
CMD_SCALE = np.array([0.60, 0.30, 1.00], dtype=np.float32)
ACT_SCALE = np.asarray([0.5, 0.8, 0.8] * 4, dtype=np.float64)
T_GAIT = 0.40
CYC = int(round(T_GAIT / 0.02))
SERVO_KP, SERVO_KD = 60.0, 2.0


def leg_ik_viewer(position, side):
    px, py, pz = position
    distance = side * D_HIP
    radius = np.hypot(py, pz)
    hip = np.arctan2(pz, py) + np.arccos(np.clip(distance / radius, -1.0, 1.0))
    sagittal_z = -np.sqrt(max(radius * radius - distance * distance, 1e-12))
    cosine = np.clip(
        (px * px + sagittal_z * sagittal_z - L1 * L1 - L2 * L2) / (2 * L1 * L2),
        -1.0, 1.0,
    )
    calf_plane = -np.arccos(cosine)
    thigh = np.arctan2(-px, -sagittal_z) - np.arctan2(
        L2 * np.sin(calf_plane), L1 + L2 * np.cos(calf_plane)
    )
    return np.array([hip, thigh, calf_plane - DELTA])


def standing_pose_viewer():
    pose = np.zeros(12)
    for index, (_, side) in enumerate(LEGS):
        pose[3 * index:3 * index + 3] = leg_ik_viewer(
            np.array([0.0, side * D_HIP, FOOT_R - H0]), side
        )
    return pose


class KeyboardPolicy:
    def __init__(
        self, prior, mean, variance, residual=None, scale=0.25,
        residual_limit=0.50, stateful=True
    ):
        self.prior = prior
        self.residual = residual
        self.mean = mean
        self.variance = variance
        self.scale = scale
        self.residual_limit = residual_limit
        self.stateful = stateful
        self.use_residual = bool(USE_RESIDUAL_AT_START and residual is not None)
        self.rng = torch.Generator(device=TORCH_DEVICE).manual_seed(CONTROL_SEED)
        self.reset()

    def reset(self):
        # 学習時の評価と同様にprior膜電位は正規乱数、残差膜電位はゼロで始める。
        self.prior_mem = torch.randn(
            1, self.prior.mem_dim, generator=self.rng, device=TORCH_DEVICE
        )
        self.residual_mem = (
            torch.zeros(1, self.residual.mem_dim, device=TORCH_DEVICE)
            if self.residual is not None else None
        )

    def toggle_residual(self):
        if self.residual is None:
            print("残差checkpointが読み込まれていないため、平地actorのみです。")
            return
        self.use_residual = not self.use_residual
        self.reset()
        print("制御:", "prior + residual" if self.use_residual else "prior only")

    @torch.inference_mode()
    def act(self, observation):
        obs = torch.as_tensor(observation, dtype=torch.float32, device=TORCH_DEVICE).unsqueeze(0)
        norm = torch.clamp(
            (obs - self.mean) / torch.sqrt(self.variance + torch.finfo(torch.float32).eps),
            -50.0, 50.0,
        )
        prior_action, self.prior_mem = self.prior(norm, self.prior_mem)
        residual_rms = 0.0
        if self.use_residual:
            residual_input = torch.cat([norm, prior_action], dim=-1)
            residual_action, next_mem = self.residual(residual_input, self.residual_mem)
            residual_action = torch.clamp(
                residual_action, -self.residual_limit, self.residual_limit
            )
            if self.stateful:
                self.residual_mem = next_mem
            else:
                self.residual_mem.zero_()
            action = torch.clamp(prior_action + self.scale * residual_action, -1.0, 1.0)
            residual_rms = float((self.scale * residual_action).square().mean().sqrt())
        else:
            action = prior_action
        return action[0].cpu().numpy(), residual_rms


class Go2KeyboardViewer:
    TERRAIN_KEYS = {ord("1"): "flat", ord("2"): "easy", ord("3"): "medium", ord("4"): "hard"}

    def __init__(self, scenes, policy, start_terrain="easy"):
        if start_terrain not in scenes:
            raise ValueError(f"未知の地形: {start_terrain}")
        self.scenes = scenes
        self.policy = policy
        self.terrain = start_terrain
        self.command = np.zeros(3, dtype=np.float32)
        self.quit_requested = False
        self.switch_to = None
        self.reset_requested = False
        self.paused = False
        self.phase_step = 0
        self.last_residual_rms = 0.0
        self.window = None

    def print_help(self):
        print("""
操作キー
  W / S : 前進 / 後退を増やす
  A / D : 左 / 右移動を増やす
  Q / E : 左 / 右旋回を増やす
  X     : 速度指令をゼロにする
  R     : ロボットを開始位置へ戻す
  B     : prior only / prior + residual を切り替える
  P     : 一時停止 / 再開
  1     : flat   2: easy(2cm)   3: medium(5cm)   4: hard(10cm)
  H     : このヘルプを表示
  Esc   : 終了
速度指令はキーを押すたびに段階的に変わります。
""")

    def status(self):
        mode = "prior+residual" if self.policy.use_residual else "prior only"
        print(
            f"地形={self.terrain:6s}  cmd=[{self.command[0]:+.2f}, {self.command[1]:+.2f}, "
            f"{self.command[2]:+.2f}]  制御={mode}  residual_rms={self.last_residual_rms:.4f}"
        )
        self.update_title()

    def update_title(self):
        if self.window is None:
            return
        mode = "prior+residual" if self.policy.use_residual else "prior only"
        state = "PAUSE" if self.paused else "RUN"
        glfw.set_window_title(
            self.window,
            f"Go2 {state} | {self.terrain} | "
            f"vx {self.command[0]:+.2f} vy {self.command[1]:+.2f} "
            f"wz {self.command[2]:+.2f} | {mode}",
        )

    def key_callback(self, keycode):
        if keycode in self.TERRAIN_KEYS:
            requested = self.TERRAIN_KEYS[keycode]
            if requested != self.terrain:
                self.switch_to = requested
                self.command[:] = 0.0
                print("地形切替:", requested)
            return
        if keycode in (27, 256):  # Escape (ASCII / GLFW)
            self.quit_requested = True
            return
        try:
            key = chr(keycode).upper()
        except (ValueError, OverflowError):
            return

        if key == "W": self.command[0] += 0.10
        elif key == "S": self.command[0] -= 0.10
        elif key == "A": self.command[1] += 0.05
        elif key == "D": self.command[1] -= 0.05
        elif key == "Q": self.command[2] += 0.20
        elif key == "E": self.command[2] -= 0.20
        elif key == "X": self.command[:] = 0.0
        elif key == "R": self.reset_requested = True
        elif key == "B": self.policy.toggle_residual()
        elif key == "P":
            self.paused = not self.paused
            print("一時停止" if self.paused else "再開")
        elif key == "H":
            self.print_help()
        else:
            return

        self.command[:] = np.clip(self.command, -CMD_SCALE, CMD_SCALE)
        self.status()

    @staticmethod
    def sensor_address(model, name):
        sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        if sensor_id < 0:
            raise KeyError(f"センサがありません: {name}")
        return int(model.sensor_adr[sensor_id])

    def prepare_model(self, scene_path):
        model = mujoco.MjModel.from_xml_path(str(scene_path))
        data = mujoco.MjData(model)
        model.actuator_ctrllimited[:] = 0
        model.actuator_ctrlrange[:] = np.array([-1e6, 1e6])
        model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
        model.opt.iterations = 8
        # ls_iterations はMuJoCoのバージョンによって存在しないことがある。
        if hasattr(model.opt, "ls_iterations"):
            model.opt.ls_iterations = 8
        # 学習用のMuJoCo Warp環境ではCCDフラグを無効化しているが、
        # このCPUビューアでは不要。古いMuJoCoに存在しない
        # mjDSBL_MULTICCD も参照しない。
        return model, data

    def reset_robot(self, model, data):
        mujoco.mj_resetData(model, data)
        data.qpos[0:3] = [0.0, 0.0, H0]
        data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        data.qpos[7:19] = standing_pose_viewer()
        data.qvel[:] = 0.0
        self.phase_step = 0
        self.policy.reset()
        mujoco.mj_forward(model, data)

    def observation(self, model, data, default_sensor, addresses):
        joint_angle = data.sensordata[addresses["pos"]:addresses["pos"] + 12] - default_sensor
        joint_velocity = data.sensordata[addresses["vel"]:addresses["vel"] + 12]
        phase = 2.0 * np.pi * (self.phase_step % CYC) / CYC
        return np.concatenate([
            np.stack([joint_angle, joint_velocity], axis=-1).reshape(24),
            data.sensordata[addresses["gyro"]:addresses["gyro"] + 3],
            data.sensordata[addresses["acc"]:addresses["acc"] + 3],
            self.command / CMD_SCALE,
            np.array([np.sin(phase), np.cos(phase)]),
        ]).astype(np.float32)

    @staticmethod
    def disable_expensive_rendering(scene):
        """歩行確認に不要な描画効果を切り、WSLgの負荷を下げる。"""
        for name in (
            "mjRND_SHADOW",
            "mjRND_REFLECTION",
            "mjRND_FOG",
            "mjRND_HAZE",
            "mjRND_SKYBOX",
        ):
            flag = getattr(mujoco.mjtRndFlag, name, None)
            if flag is not None:
                scene.flags[int(flag)] = 0

    def render_frame(self, model, data, option, camera, scene, context):
        width, height = glfw.get_framebuffer_size(self.window)
        if width <= 0 or height <= 0:
            return
        mujoco.mjv_updateScene(
            model,
            data,
            option,
            None,
            camera,
            mujoco.mjtCatBit.mjCAT_ALL,
            scene,
        )
        viewport = mujoco.MjrRect(0, 0, width, height)
        mujoco.mjr_render(viewport, scene, context)
        glfw.swap_buffers(self.window)

    def run_scene(self):
        model, data = self.prepare_model(self.scenes[self.terrain])
        default_qpos = standing_pose_viewer()
        default_sensor = default_qpos[PERM]
        joint_low = model.jnt_range[1:, 0][PERM]
        joint_high = model.jnt_range[1:, 1][PERM]
        addresses = {
            "pos": self.sensor_address(model, "FR_hip_pos"),
            "vel": self.sensor_address(model, "FR_hip_vel"),
            "gyro": self.sensor_address(model, "imu_gyro"),
            "acc": self.sensor_address(model, "imu_acc"),
        }
        frame_skip = max(1, int(round(0.02 / model.opt.timestep)))
        control_dt = frame_skip * model.opt.timestep
        self.reset_robot(model, data)
        self.reset_requested = False
        self.switch_to = None

        if not glfw.init():
            raise RuntimeError("GLFW initialization failed")
        glfw.window_hint(glfw.SAMPLES, 0)          # MSAAを無効化
        glfw.window_hint(glfw.RESIZABLE, glfw.FALSE)
        window = glfw.create_window(
            WINDOW_WIDTH, WINDOW_HEIGHT, "Go2 lightweight viewer", None, None
        )
        if window is None:
            glfw.terminate()
            raise RuntimeError("GLFW window could not be created")

        self.window = window
        glfw.make_context_current(window)
        glfw.swap_interval(1 if VSYNC else 0)

        camera = mujoco.MjvCamera()
        option = mujoco.MjvOption()
        scene = mujoco.MjvScene(model, maxgeom=512)
        context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_100)
        self.disable_expensive_rendering(scene)

        camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "track")
        if camera_id >= 0:
            camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
            camera.fixedcamid = camera_id
        else:
            camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            camera.azimuth = 90.0
            camera.elevation = -15.0
            camera.distance = 2.0

        def glfw_key_callback(_window, key, _scancode, action, _mods):
            if action in (glfw.PRESS, glfw.REPEAT):
                self.key_callback(key)

        glfw.set_key_callback(window, glfw_key_callback)
        self.status()
        print(
            f"軽量表示: {WINDOW_WIDTH}x{WINDOW_HEIGHT} / {RENDER_HZ:g} fps / "
            f"制御 {1/control_dt:.0f} Hz / torch threads={TORCH_THREADS}"
        )

        render_period = 1.0 / RENDER_HZ
        now = time.perf_counter()
        next_control = now
        next_render = now
        report_start = now
        control_steps = render_frames = late_steps = 0

        try:
            while (
                not glfw.window_should_close(window)
                and not self.quit_requested
                and self.switch_to is None
            ):
                glfw.poll_events()
                now = time.perf_counter()

                if self.reset_requested:
                    self.reset_robot(model, data)
                    self.reset_requested = False
                    print("ロボットをリセットしました")

                if not self.paused and now >= next_control:
                    if now - next_control > control_dt * 0.5:
                        late_steps += 1
                    obs = self.observation(model, data, default_sensor, addresses)
                    action, self.last_residual_rms = self.policy.act(obs)
                    target = np.clip(default_sensor + action * ACT_SCALE, joint_low, joint_high)
                    for _ in range(frame_skip):
                        position = data.qpos[7:19][PERM]
                        velocity = data.qvel[6:18][PERM]
                        data.ctrl[:] = SERVO_KP * (target - position) - SERVO_KD * velocity
                        mujoco.mj_step(model, data)
                    self.phase_step += 1
                    control_steps += 1
                    next_control += control_dt
                    if next_control < time.perf_counter() - control_dt:
                        next_control = time.perf_counter()
                elif self.paused:
                    next_control = now + control_dt

                now = time.perf_counter()
                if now >= next_render:
                    self.render_frame(model, data, option, camera, scene, context)
                    render_frames += 1
                    next_render += render_period
                    if next_render < time.perf_counter() - render_period:
                        next_render = time.perf_counter()

                now = time.perf_counter()
                if REPORT_INTERVAL > 0 and now - report_start >= REPORT_INTERVAL:
                    elapsed = now - report_start
                    print(
                        f"実測: control={control_steps/elapsed:4.1f} Hz  "
                        f"render={render_frames/elapsed:4.1f} fps  遅延={late_steps}"
                    )
                    report_start = now
                    control_steps = render_frames = late_steps = 0

                deadline = next_render if self.paused else min(next_control, next_render)
                delay = deadline - time.perf_counter()
                if delay > 0:
                    time.sleep(min(delay, 0.002))
        finally:
            context.free()
            glfw.destroy_window(window)
            glfw.terminate()
            self.window = None

        if not self.quit_requested and self.switch_to is None:
            self.quit_requested = True

    def run(self):
        self.print_help()
        while not self.quit_requested:
            self.run_scene()
            if self.switch_to is not None:
                self.terrain = self.switch_to
                self.switch_to = None
                self.policy.reset()
        print("ビューアを終了しました")


def main():
    keyboard_policy = KeyboardPolicy(
        base_actor,
        obs_mean,
        obs_var,
        residual=residual_actor,
        scale=residual_scale,
        residual_limit=residual_limit,
        stateful=residual_stateful,
    )
    viewer_app = Go2KeyboardViewer(SCENES, keyboard_policy, start_terrain=START_TERRAIN)
    viewer_app.run()


if __name__ == "__main__":
    main()
