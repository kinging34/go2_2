#!/usr/bin/env python3
"""WSL/WSLg用 Go2 SNN キーボードビューア (旧33次元版).

学習済みSNN Actorを単体で推論し、W/S/A/D/Q/Eで速度指令、1〜4で地形を切り替える。
Action Prior / Critic / PPOは推論時には使用しない。

必要ファイル:
  - scene_flat.xml と、そのXMLから参照されるmesh/texture等
  - Actor checkpoint (*.pt)
  - 観測正規化 (*.npz)

主な操作:
  W/S: 前進/後退, A/D: 左/右, Q/E: 左/右旋回, X: 停止
  1: flat, 2: easy(2cm), 3: medium(5cm), 4: hard(10cm)
  R: リセット, P: 一時停止, H: ヘルプ, Esc: 終了
"""

import os
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

# ============================================================
# ユーザー設定
# ============================================================
WORK_DIR = Path(__file__).resolve().parent
FLAT_SCENE = WORK_DIR / "scene_flat.xml"

# 手動指定する場合はPathを入れる。Noneならparams_*から入力次元を見て自動探索。
PARAMS_DIR = "params_09-23_APEX"
ACTOR_CHECKPOINT = None
NORM_CHECKPOINT = None

EXPECTED_OBS_DIM = 33
START_TERRAIN = "easy"  # flat / easy / medium / hard
TORCH_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 描画は軽量化。制御周期は学習と同じ20msを目標にする。
WINDOW_WIDTH, WINDOW_HEIGHT = 640, 360
RENDER_HZ = 30.0
VSYNC = False
TORCH_THREADS = 1
REPORT_INTERVAL = 5.0

# 学習時と同じ制御定数
H0 = 0.33
CMD_LIMIT = np.array([0.60, 0.30, 1.00], dtype=np.float32)
ACT_SCALE = np.asarray([0.5, 0.8, 0.8] * 4, dtype=np.float64)
SERVO_KP, SERVO_KD = 60.0, 2.0
PERM = np.array([3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8])

# キー1回あたりの指令変化量
DELTA_VX = 0.10
DELTA_VY = 0.05
DELTA_WZ = 0.20

# SNN定数（学習Notebookと一致）
LIF_BETA = 0.5
LIF_THRESHOLD = 1.0
SPIKE_SLOPE = 3.0
ENC_VTH = 0.999

# 45D APEX Actorの観測scale
APEX_CMD_OBS_SCALE = np.array([2.0, 2.0, 0.25], dtype=np.float32)

# Go2脚形状（standing pose生成用）
D_HIP = 0.0955
L1 = 0.213
L2 = float(np.hypot(0.002, 0.213))
DELTA = float(np.arctan2(0.002, 0.213))
FOOT_R = 0.022
LEGS = [("FL", +1), ("FR", -1), ("RL", +1), ("RR", -1)]


torch.set_num_threads(TORCH_THREADS)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

print("作業フォルダ:", WORK_DIR.resolve())
print("モデル種別:", "旧33次元版")
print("推論デバイス:", TORCH_DEVICE)
print("DISPLAY:", os.environ.get("DISPLAY"), "/ WAYLAND_DISPLAY:", os.environ.get("WAYLAND_DISPLAY"))

if not FLAT_SCENE.is_file():
    raise FileNotFoundError(f"scene_flat.xml がありません: {FLAT_SCENE.resolve()}")
if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
    raise RuntimeError("WSLg の DISPLAY/WAYLAND_DISPLAY が見つかりません。WSL上のGUI対応端末から実行してください。")


# ============================================================
# 簡易テスト地形: flat / 2cm / 5cm / 10cm
# ============================================================
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

    # 既存のplane/hfieldを除去してビューア地形へ置き換える。
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


# ============================================================
# SNN Actor（PPO版Notebookと同じ deterministic population encoder）
# ============================================================
def lif_step(current, previous, beta=LIF_BETA, threshold=LIF_THRESHOLD):
    reset = (previous > threshold).to(previous.dtype)
    membrane = beta * previous + current - reset * threshold
    spike = (membrane - threshold > 0).to(membrane.dtype)
    return spike, membrane


class Encoder(nn.Module):
    def __init__(self, obs_dim, pop_dim):
        super().__init__()
        self.obs_dim = obs_dim
        self.pop_dim = pop_dim
        self.weight = nn.Parameter(torch.ones(1, obs_dim))
        self.bias = nn.Parameter(torch.zeros(1, obs_dim))
        self.register_buffer("zeros", torch.zeros(1, obs_dim * 2))
        u = (torch.arange(pop_dim, dtype=torch.float32) + 0.5) / pop_dim
        self.register_buffer("fixed_u", u.view(1, 1, pop_dim))

    def forward(self, obs):
        obs = torch.tanh(obs * self.weight + self.bias)
        probability = torch.maximum(torch.cat([obs, -obs], dim=1), self.zeros)
        u = self.fixed_u.expand(probability.shape[0], probability.shape[1], self.pop_dim)
        voltage = probability.unsqueeze(-1) + u
        return voltage.gt(ENC_VTH).to(probability.dtype).reshape(probability.shape[0], -1)


class Decoder(nn.Module):
    def __init__(self, act_dim, pop_dim):
        super().__init__()
        self.act_dim = act_dim
        self.pop_dim = pop_dim
        self.register_buffer("weight", torch.ones(1, act_dim))
        self.register_buffer("bias", torch.zeros(1, act_dim))

    def forward(self, spikes):
        rates = spikes.reshape(-1, self.act_dim * 2, self.pop_dim).mean(-1)
        return torch.tanh((rates[:, :self.act_dim] - rates[:, self.act_dim:]) * self.weight + self.bias)


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
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _strip_prefix(key):
    for prefix in ("_orig_mod.", "module."):
        if key.startswith(prefix):
            key = key[len(prefix):]
    return key


def extract_actor_state(checkpoint):
    """actor単体ptとfull checkpointの両方を許容する。"""
    if not isinstance(checkpoint, dict):
        raise TypeError("checkpointが辞書形式ではありません")

    if "policy" in checkpoint and isinstance(checkpoint["policy"], dict):
        raw = checkpoint["policy"]
        selected = {}
        for k, v in raw.items():
            k = _strip_prefix(k)
            if k.startswith("san."):
                selected[k[len("san."):]] = v
        if not selected:
            raise KeyError("full checkpointの policy 内に san.* が見つかりません")
        return selected

    for nested_key in ("actor", "san", "state_dict"):
        if nested_key in checkpoint and isinstance(checkpoint[nested_key], dict):
            checkpoint = checkpoint[nested_key]
            break

    result = {}
    for k, v in checkpoint.items():
        if not torch.is_tensor(v):
            continue
        k = _strip_prefix(k)
        if k.startswith("san."):
            k = k[len("san."):]
        if k.startswith("lif"):
            continue
        result[k] = v
    return result


def infer_obs_dim_from_state(state):
    if "encoder.weight" not in state:
        raise KeyError("encoder.weight がcheckpointにありません")
    return int(state["encoder.weight"].shape[1])


def actor_from_state(state):
    obs_dim = infer_obs_dim_from_state(state)
    act_dim = int(state["decoder.weight"].shape[1])
    hidden1, encoded = state["Linear1.weight"].shape
    hidden2 = int(state["Linear2.weight"].shape[0])
    output_spikes = int(state["Linear3.weight"].shape[0])
    enc_pop = encoded // (obs_dim * 2)
    dec_pop = output_spikes // (act_dim * 2)

    actor = ViewerSpikeActor(obs_dim, hidden1, hidden2, output_spikes,
                             act_dim, enc_pop, dec_pop)
    missing, unexpected = actor.load_state_dict(state, strict=False)
    # 古い/別形式checkpointでfixed_uが無い場合を除き、重要parameterの欠損は拒否。
    critical_missing = [k for k in missing if k not in ("encoder.zeros", "encoder.fixed_u")]
    if critical_missing or unexpected:
        raise RuntimeError(f"checkpointの構造が想定外です。missing={critical_missing}, unexpected={unexpected}")
    actor.to(TORCH_DEVICE).eval()
    for parameter in actor.parameters():
        parameter.requires_grad_(False)
    return actor


def numeric_suffix(path):
    values = re.findall(r"(\d+)(?:Kit|it)", path.name)
    return int(values[-1]) if values else -1


def candidate_param_dirs():
    if PARAMS_DIR is not None:
        p = Path(PARAMS_DIR)
        if not p.is_absolute():
            p = WORK_DIR / p
        return [p]
    dirs = [p for p in WORK_DIR.glob("params_*") if p.is_dir()]
    # 33Dは旧params_*（*_APEX以外）を優先する。
    return sorted(dirs, key=lambda p: (0 if p.name.endswith("_APEX") else 1, p.stat().st_mtime), reverse=True)


def find_actor_checkpoint():
    if ACTOR_CHECKPOINT is not None:
        p = Path(ACTOR_CHECKPOINT)
        return p if p.is_absolute() else WORK_DIR / p

    checked = []
    for folder in candidate_param_dirs():
        # actor単体を優先。fullも手動指定なら読める。
        candidates = sorted(folder.glob("*_actor_*Kit.pt"),
                            key=lambda p: (numeric_suffix(p), p.stat().st_mtime), reverse=True)
        for p in candidates:
            try:
                state = extract_actor_state(torch_load(p))
                dim = infer_obs_dim_from_state(state)
                checked.append((p, dim))
                if dim == EXPECTED_OBS_DIM:
                    return p
            except Exception as exc:
                print(f"checkpoint候補をスキップ: {p.name}: {exc}")
    details = "\n".join(f"  {p} -> obs_dim={d}" for p, d in checked[:20])
    raise FileNotFoundError(
        f"{EXPECTED_OBS_DIM}次元Actorが見つかりません。ACTOR_CHECKPOINTを指定してください。"
        + ("\n確認した候補:\n" + details if details else "")
    )


def find_norm_checkpoint(actor_path):
    if NORM_CHECKPOINT is not None:
        p = Path(NORM_CHECKPOINT)
        return p if p.is_absolute() else WORK_DIR / p

    # foo_actor_500Kit.pt -> foo_norm_500Kit.npz
    expected = actor_path.with_name(actor_path.name.replace("_actor_", "_norm_").replace(".pt", ".npz"))
    if expected.is_file():
        return expected

    same_step = re.search(r"_(\d+Kit)\.pt$", actor_path.name)
    step_tag = same_step.group(1) if same_step else None
    candidates = list(actor_path.parent.glob("*_norm_*.npz"))
    if step_tag:
        same = [p for p in candidates if p.name.endswith(f"_{step_tag}.npz")]
        if same:
            candidates = same
    for p in sorted(candidates, key=lambda p: (numeric_suffix(p), p.stat().st_mtime), reverse=True):
        try:
            z = np.load(p)
            if int(np.asarray(z["mean"]).size) == EXPECTED_OBS_DIM:
                return p
        except Exception:
            pass
    raise FileNotFoundError("対応するnorm npzが見つかりません。NORM_CHECKPOINTを指定してください。")


actor_path = find_actor_checkpoint()
actor_state = extract_actor_state(torch_load(actor_path))
actual_obs_dim = infer_obs_dim_from_state(actor_state)
if actual_obs_dim != EXPECTED_OBS_DIM:
    raise ValueError(
        f"このビューアは{EXPECTED_OBS_DIM}次元用ですが、checkpointは{actual_obs_dim}次元です: {actor_path}"
    )
actor = actor_from_state(actor_state)

norm_path = find_norm_checkpoint(actor_path)
norm_npz = np.load(norm_path)
obs_mean = torch.as_tensor(norm_npz["mean"], dtype=torch.float32, device=TORCH_DEVICE).reshape(-1)
obs_var = torch.as_tensor(norm_npz["var"], dtype=torch.float32, device=TORCH_DEVICE).reshape(-1)
if obs_mean.numel() != EXPECTED_OBS_DIM or obs_var.numel() != EXPECTED_OBS_DIM:
    raise ValueError(
        f"norm次元が不一致です: mean={obs_mean.numel()}, var={obs_var.numel()}, expected={EXPECTED_OBS_DIM}"
    )

print("Actor:", actor_path)
print("Norm :", norm_path)
print(f"obs={actor.obs_dim}, action={actor.act_dim}, mem={actor.mem_dim}")


# ============================================================
# Go2 utility
# ============================================================
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


def projected_gravity_from_quat(quat_wxyz):
    """学習45D版と同じ: world重力[0,0,-1]をbody frameへ射影。"""
    w, x, y, z = [float(v) for v in quat_wxyz]
    R = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
    ], dtype=np.float32)
    return R.T @ np.array([0.0, 0.0, -1.0], dtype=np.float32)


class KeyboardPolicy:
    def __init__(self, actor, mean, variance):
        self.actor = actor
        self.mean = mean
        self.variance = variance
        self.reset()

    def reset(self):
        # 学習評価時 new_mem() と同じゼロ初期化。
        self.mem = torch.zeros(1, self.actor.mem_dim, dtype=torch.float32, device=TORCH_DEVICE)

    @torch.inference_mode()
    def act(self, observation):
        obs = torch.as_tensor(observation, dtype=torch.float32, device=TORCH_DEVICE).unsqueeze(0)
        norm = torch.clamp(
            (obs - self.mean) / torch.sqrt(self.variance + 1e-6),
            -50.0, 50.0,
        )
        action, self.mem = self.actor(norm, self.mem)
        return torch.clamp(action[0], -1.0, 1.0).cpu().numpy()


class Go2KeyboardViewer:
    TERRAIN_KEYS = {ord("1"): "flat", ord("2"): "easy", ord("3"): "medium", ord("4"): "hard"}

    def __init__(self, scenes, policy, start_terrain="easy"):
        if start_terrain not in scenes:
            raise ValueError(f"未知の地形: {start_terrain}")
        self.scenes = scenes
        self.policy = policy
        self.terrain = start_terrain
        self.command = np.zeros(3, dtype=np.float32)
        self.last_action = np.zeros(12, dtype=np.float32)
        self.quit_requested = False
        self.switch_to = None
        self.reset_requested = False
        self.paused = False
        self.window = None

    def print_help(self):
        print(r"""
操作キー
  W / S : 前進速度 vx を + / -
  A / D : 左右速度 vy を + / -
  Q / E : 左右旋回 wz を + / -
  X     : 速度指令をゼロ
  R     : ロボットを開始位置へ戻す
  P     : 一時停止 / 再開
  1     : flat
  2     : easy   (最大 2 cm)
  3     : medium (最大 5 cm)
  4     : hard   (最大10 cm)
  H     : ヘルプ
  Esc   : 終了

現在の指令上限:
  vx = ±0.60 m/s, vy = ±0.30 m/s, wz = ±1.00 rad/s
""")

    def status(self):
        print(
            f"地形={self.terrain:6s}  "
            f"cmd=[vx {self.command[0]:+.2f}, vy {self.command[1]:+.2f}, wz {self.command[2]:+.2f}]"
        )
        self.update_title()

    def update_title(self):
        if self.window is None:
            return
        state = "PAUSE" if self.paused else "RUN"
        glfw.set_window_title(
            self.window,
            f"Go2 {EXPECTED_OBS_DIM}D {state} | {self.terrain} | "
            f"vx {self.command[0]:+.2f} vy {self.command[1]:+.2f} wz {self.command[2]:+.2f}",
        )

    def key_callback(self, keycode):
        if keycode in self.TERRAIN_KEYS:
            requested = self.TERRAIN_KEYS[keycode]
            if requested != self.terrain:
                self.switch_to = requested
                self.command[:] = 0.0
                print("地形切替:", requested)
            return
        if keycode in (27, 256):
            self.quit_requested = True
            return
        try:
            key = chr(keycode).upper()
        except (ValueError, OverflowError):
            return

        if key == "W": self.command[0] += DELTA_VX
        elif key == "S": self.command[0] -= DELTA_VX
        elif key == "A": self.command[1] += DELTA_VY
        elif key == "D": self.command[1] -= DELTA_VY
        elif key == "Q": self.command[2] += DELTA_WZ
        elif key == "E": self.command[2] -= DELTA_WZ
        elif key == "X": self.command[:] = 0.0
        elif key == "R": self.reset_requested = True
        elif key == "P":
            self.paused = not self.paused
            print("一時停止" if self.paused else "再開")
        elif key == "H":
            self.print_help()
        else:
            return

        self.command[:] = np.clip(self.command, -CMD_LIMIT, CMD_LIMIT)
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
        if hasattr(model.opt, "ls_iterations"):
            model.opt.ls_iterations = 8
        return model, data

    def reset_robot(self, model, data):
        mujoco.mj_resetData(model, data)
        data.qpos[0:3] = [0.0, 0.0, H0]
        data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        data.qpos[7:19] = standing_pose_viewer()
        data.qvel[:] = 0.0
        self.last_action[:] = 0.0
        self.policy.reset()
        mujoco.mj_forward(model, data)

    def observation(self, model, data, default_sensor, addresses):
        joint_angle = data.sensordata[addresses["pos"]:addresses["pos"] + 12] - default_sensor
        joint_velocity = data.sensordata[addresses["vel"]:addresses["vel"] + 12]
        gyro = data.sensordata[addresses["gyro"]:addresses["gyro"] + 3]
        acc = data.sensordata[addresses["acc"]:addresses["acc"] + 3]
        # 33D旧APEX版: [q0,dq0,q1,dq1,...] + gyro + raw acc + command/CMD_LIMIT
        joint_pair = np.stack([joint_angle, joint_velocity], axis=-1).reshape(24)
        obs = np.concatenate([
            joint_pair,
            gyro,
            acc,
            self.command / CMD_LIMIT,
        ])
        obs = np.asarray(obs, dtype=np.float32)
        if obs.size != EXPECTED_OBS_DIM:
            raise RuntimeError(f"観測次元が{obs.size}です。期待値={EXPECTED_OBS_DIM}")
        return obs

    @staticmethod
    def disable_expensive_rendering(scene):
        for name in ("mjRND_SHADOW", "mjRND_REFLECTION", "mjRND_FOG", "mjRND_HAZE", "mjRND_SKYBOX"):
            flag = getattr(mujoco.mjtRndFlag, name, None)
            if flag is not None:
                scene.flags[int(flag)] = 0

    def render_frame(self, model, data, option, camera, scene, context):
        width, height = glfw.get_framebuffer_size(self.window)
        if width <= 0 or height <= 0:
            return
        mujoco.mjv_updateScene(model, data, option, None, camera,
                               mujoco.mjtCatBit.mjCAT_ALL, scene)
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
        glfw.window_hint(glfw.SAMPLES, 0)
        glfw.window_hint(glfw.RESIZABLE, glfw.FALSE)
        window = glfw.create_window(WINDOW_WIDTH, WINDOW_HEIGHT,
                                    f"Go2 {EXPECTED_OBS_DIM}D keyboard viewer", None, None)
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
        print(f"表示={WINDOW_WIDTH}x{WINDOW_HEIGHT} / {RENDER_HZ:g} fps / 制御={1/control_dt:.0f} Hz")

        render_period = 1.0 / RENDER_HZ
        now = time.perf_counter()
        next_control = now
        next_render = now
        report_start = now
        control_steps = render_frames = late_steps = 0

        try:
            while (not glfw.window_should_close(window)
                   and not self.quit_requested
                   and self.switch_to is None):
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
                    action = self.policy.act(obs)
                    target = np.clip(default_sensor + action * ACT_SCALE, joint_low, joint_high)

                    for _ in range(frame_skip):
                        position = data.qpos[7:19][PERM]
                        velocity = data.qvel[6:18][PERM]
                        data.ctrl[:] = SERVO_KP * (target - position) - SERVO_KD * velocity
                        mujoco.mj_step(model, data)

                    # 45D版では次stepのprevious actionになる。33D版では観測に使わない。
                    self.last_action[:] = action
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
                self.last_action[:] = 0.0
                self.policy.reset()
        print("ビューアを終了しました")


def main():
    policy = KeyboardPolicy(actor, obs_mean, obs_var)
    app = Go2KeyboardViewer(SCENES, policy, start_terrain=START_TERRAIN)
    app.run()


if __name__ == "__main__":
    main()
