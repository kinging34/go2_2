#!/usr/bin/env python3
"""WSL/WSLg用 Go2 SNN APEX v5 キーボードビューア (47次元・膜電位モーターヘッド版).

学習済みSNN Actorを単体で推論し、W/S/A/D/Q/Eで速度指令、1〜4で地形を切り替える。
Action Prior / Critic / PPOは推論時には使用しない。

v5対応点:
  - 45D APEX観測 + 歩容phase sin/cos = 47D
  - SNN第1/第2層の膜電位をLayerNormして連続モーターヘッドへ入力
  - 最終spike-rate decoderを小さい残差として加算
  - 学習時と同じ APEX PD (Kp=20, Kd=0.5), action scale=0.25, torque limit

注意:
  v5のcheckpointからLayerNorm/continuous headを形状とキー名で自動検出します。
  構造を特定できない場合は、推測して動かさず診断を表示して停止します。

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
import torch.nn.functional as F
import glfw
import mujoco

# ============================================================
# ユーザー設定
# ============================================================
WORK_DIR = Path(__file__).resolve().parent
FLAT_SCENE = WORK_DIR / "scene_flat.xml"

# 手動指定する場合はPathを入れる。Noneならparams_*から入力次元を見て自動探索。
PARAMS_DIR = "params_09-24_APEX_6"
ACTOR_CHECKPOINT = None
NORM_CHECKPOINT = None

EXPECTED_OBS_DIM = 47
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
CMD_LIMIT = np.array([0.60, 0.60, 1.00], dtype=np.float32)
ACT_SCALE = np.full(12, 0.25, dtype=np.float64)
SERVO_KP, SERVO_KD = 20.0, 0.5
ACTION_CLIP = 100.0
FALLBACK_TORQUE_LIMIT = np.asarray([23.7, 23.7, 45.43] * 4, dtype=np.float64)

# v5でActorへ追加した歩容phase。学習Notebookの T_GAIT=0.40 s と20 ms制御に合わせる。
T_GAIT = 0.40
GAIT_CONTROL_DT = 0.02
GAIT_CYCLE_STEPS = int(round(T_GAIT / GAIT_CONTROL_DT))

# v5の「最終spike-rate出力を小さい残差として残す」の既定値。
# checkpoint/full checkpoint内に対応するscalar/configがあればそちらを優先する。
DEFAULT_SPIKE_RESIDUAL_SCALE = 0.10
DEFAULT_ACTION_MEAN_CLIP = 4.0
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

# 47D v5 Actorのうち先頭45Dに使うAPEX観測scale
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
print("モデル種別:", "APEX v5 47次元・膜電位モーターヘッド版")
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
# SNN Actor v5
# - deterministic population encoder
# - LIF 3層
# - mem1/mem2 -> LayerNorm -> continuous motor head
# - spike-rate decoder is a small residual
# ============================================================
def lif_step(current, previous, beta=LIF_BETA, threshold=LIF_THRESHOLD):
    reset = (previous > threshold).to(previous.dtype)
    membrane = beta * previous + current - reset * threshold
    spike = (membrane - threshold > 0).to(membrane.dtype)
    return spike, membrane


def torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _strip_prefix(key):
    # torch.compile / DDP由来のprefixを複数段許容する。
    changed = True
    while changed:
        changed = False
        for prefix in ("_orig_mod.", "module."):
            if key.startswith(prefix):
                key = key[len(prefix):]
                changed = True
    return key


def extract_actor_state_and_config(checkpoint):
    """actor単体pt/full checkpointの両方からActor stateとconfigを取り出す。"""
    if not isinstance(checkpoint, dict):
        raise TypeError("checkpointが辞書形式ではありません")

    config = checkpoint.get("config", {}) if isinstance(checkpoint.get("config", {}), dict) else {}

    if "policy" in checkpoint and isinstance(checkpoint["policy"], dict):
        raw = checkpoint["policy"]
        selected = {}
        for k, v in raw.items():
            if not torch.is_tensor(v):
                continue
            k = _strip_prefix(k)
            if k.startswith("san."):
                selected[k[len("san."):]] = v
        if not selected:
            raise KeyError("full checkpointの policy 内に san.* が見つかりません")
        return selected, config

    actor_obj = checkpoint
    for nested_key in ("actor", "san", "state_dict"):
        if nested_key in actor_obj and isinstance(actor_obj[nested_key], dict):
            actor_obj = actor_obj[nested_key]
            break

    result = {}
    for k, v in actor_obj.items():
        if not torch.is_tensor(v):
            continue
        k = _strip_prefix(k)
        if k.startswith("san."):
            k = k[len("san."):]
        if k.startswith("lif"):
            # LIFがparameterを持たない版との互換。Tensorがあれば使わない。
            continue
        result[k] = v
    return result, config


def infer_obs_dim_from_state(state):
    if "encoder.weight" not in state:
        raise KeyError("encoder.weight がcheckpointにありません")
    w = state["encoder.weight"]
    if w.ndim != 2 or w.shape[0] != 1:
        raise ValueError(f"encoder.weight shapeが想定外です: {tuple(w.shape)}")
    return int(w.shape[1])


def _state_shapes(state):
    return "\n".join(f"  {k:42s} {tuple(v.shape)}" for k, v in sorted(state.items()))


def _find_layernorm_prefixes(state, hidden1, hidden2):
    """v5のmembrane LayerNormをstate_dict形状から特定する。"""
    items = []
    for k, w in state.items():
        if not k.endswith(".weight") or w.ndim != 1:
            continue
        prefix = k[:-len(".weight")]
        bkey = prefix + ".bias"
        if bkey not in state or state[bkey].shape != w.shape:
            continue
        low = prefix.lower()
        if "encoder" in low or "decoder" in low:
            continue
        # LayerNormらしい名前を優先。Linear bias等はweightが2Dなのでここには来ない。
        score = 0
        if "norm" in low or ".ln" in low or low.startswith("ln"):
            score += 20
        if "mem" in low:
            score += 10
        if "motor" in low or "head" in low or "readout" in low:
            score += 4
        items.append((prefix, int(w.numel()), score))

    # 連結後1個のLayerNormを使う実装にも対応。
    concat = sorted([x for x in items if x[1] == hidden1 + hidden2], key=lambda x: x[2], reverse=True)
    if concat:
        return {"mode": "concat", "prefix": concat[0][0]}

    same1 = [x for x in items if x[1] == hidden1]
    same2 = [x for x in items if x[1] == hidden2]
    if hidden1 == hidden2:
        candidates = sorted(same1, key=lambda x: (x[2], x[0]), reverse=True)
        if len(candidates) < 2:
            raise RuntimeError("hidden1==hidden2ですが、membrane用LayerNormを2個特定できません")

        def numeric_score(name, want):
            low = name.lower()
            score = 0
            if f"norm{want}" in low or f"norm_{want}" in low or f"ln{want}" in low or f"ln_{want}" in low:
                score += 100
            if f"mem{want}" in low or f"mem_{want}" in low:
                score += 80
            return score

        p1 = max(candidates, key=lambda x: numeric_score(x[0], 1) + x[2])[0]
        remaining = [x for x in candidates if x[0] != p1]
        p2 = max(remaining, key=lambda x: numeric_score(x[0], 2) + x[2])[0]
        return {"mode": "separate", "prefix1": p1, "prefix2": p2}

    if not same1 or not same2:
        raise RuntimeError(
            f"membrane LayerNormを特定できません (hidden1={hidden1}, hidden2={hidden2})"
        )
    p1 = max(same1, key=lambda x: x[2])[0]
    p2 = max(same2, key=lambda x: x[2])[0]
    return {"mode": "separate", "prefix1": p1, "prefix2": p2}


def _find_motor_head_prefix(state, act_dim, hidden1, hidden2):
    candidates = []
    for k, w in state.items():
        if not k.endswith(".weight") or w.ndim != 2 or int(w.shape[0]) != act_dim:
            continue
        prefix = k[:-len(".weight")]
        low = prefix.lower()
        if prefix in ("Linear1", "Linear2", "Linear3") or "decoder" in low:
            continue
        in_dim = int(w.shape[1])
        if in_dim not in (hidden1, hidden2, hidden1 + hidden2):
            continue
        score = 0
        if "motor" in low: score += 30
        if "head" in low: score += 25
        if "readout" in low: score += 20
        if "continuous" in low: score += 15
        if "action" in low: score += 10
        if in_dim == hidden1 + hidden2: score += 8
        if "critic" in low or "value" in low: score -= 100
        candidates.append((score, prefix, in_dim))
    if not candidates:
        raise RuntimeError("12出力のcontinuous motor headを特定できません")
    candidates.sort(reverse=True)
    _, prefix, in_dim = candidates[0]
    return prefix, in_dim


def _decoder_kind(state):
    if "decoder.log_gain" in state:
        return "log_gain"
    if "decoder.weight" in state:
        return "weight"
    # 最終spike residualを完全に廃止しているcheckpointならnoneも許容。
    return "none"


def _scalar_from_config_or_state(config, state, default):
    config_keys = (
        "spike_residual_scale", "snn_spike_residual_scale",
        "motor_spike_residual_scale", "decoder_residual_scale",
    )
    for k in config_keys:
        if k in config:
            try:
                return float(config[k]), f"config[{k}]"
            except Exception:
                pass
    for k, v in state.items():
        low = k.lower()
        if ("resid" in low and "scale" in low) or ("spike" in low and "scale" in low):
            if torch.is_tensor(v) and v.numel() == 1:
                return float(v.reshape(-1)[0]), k
    return float(default), "viewer default"


def _action_clip_from_config(config):
    for k in ("snn_action_mean_clip", "action_mean_clip", "actor_mean_clip"):
        if k in config:
            try:
                return float(config[k]), f"config[{k}]"
            except Exception:
                pass
    return float(DEFAULT_ACTION_MEAN_CLIP), "viewer default"


class ViewerV5SpikeActor(nn.Module):
    """checkpointのTensorをそのまま使うv5推論Actor。

    module名を固定して再実装するとNotebook側の命名変更だけでロード不能になるため、
    state_dictの形状からLayerNormとmotor headを特定してfunctionalに実行する。
    """
    def __init__(self, state, config):
        super().__init__()
        self.obs_dim = infer_obs_dim_from_state(state)
        self.act_dim = 12

        required = (
            "Linear1.weight", "Linear1.bias", "Linear2.weight", "Linear2.bias",
            "Linear3.weight", "Linear3.bias", "encoder.weight", "encoder.bias",
        )
        missing = [k for k in required if k not in state]
        if missing:
            raise KeyError(f"v5 Actorの必須キーがありません: {missing}\nstate_dict:\n{_state_shapes(state)}")

        h1, encoded = state["Linear1.weight"].shape
        h2 = int(state["Linear2.weight"].shape[0])
        out_spikes = int(state["Linear3.weight"].shape[0])
        if encoded % (self.obs_dim * 2) != 0:
            raise ValueError("Linear1入力次元からencoder population数を復元できません")
        self.enc_pop = int(encoded // (self.obs_dim * 2))
        if out_spikes % (self.act_dim * 2) != 0:
            raise ValueError("Linear3出力次元からdecoder population数を復元できません")
        self.dec_pop = int(out_spikes // (self.act_dim * 2))
        self.h1, self.h2, self.out_spikes = int(h1), int(h2), out_spikes
        self.p1 = self.h1
        self.p2 = self.h1 + self.h2
        self.mem_dim = self.h1 + self.h2 + self.out_spikes

        norm_info = _find_layernorm_prefixes(state, self.h1, self.h2)
        head_prefix, head_in = _find_motor_head_prefix(state, self.act_dim, self.h1, self.h2)
        self.norm_info = norm_info
        self.head_prefix = head_prefix
        self.head_in = head_in
        self.decoder_kind = _decoder_kind(state)
        self.residual_scale, self.residual_scale_source = _scalar_from_config_or_state(
            config, state, DEFAULT_SPIKE_RESIDUAL_SCALE
        )
        self.action_mean_clip, self.action_clip_source = _action_clip_from_config(config)

        # 使用Tensorをbufferとして保持する。
        for name, key in (
            ("W1", "Linear1.weight"), ("b1", "Linear1.bias"),
            ("W2", "Linear2.weight"), ("b2", "Linear2.bias"),
            ("W3", "Linear3.weight"), ("b3", "Linear3.bias"),
            ("enc_weight", "encoder.weight"), ("enc_bias", "encoder.bias"),
            ("head_weight", head_prefix + ".weight"),
        ):
            self.register_buffer(name, state[key].detach().float().clone())
        h_bias_key = head_prefix + ".bias"
        hb = state[h_bias_key] if h_bias_key in state else torch.zeros(self.act_dim)
        self.register_buffer("head_bias", hb.detach().float().clone())

        if norm_info["mode"] == "concat":
            p = norm_info["prefix"]
            self.register_buffer("norm_weight", state[p + ".weight"].detach().float().clone())
            self.register_buffer("norm_bias", state[p + ".bias"].detach().float().clone())
        else:
            p1, p2 = norm_info["prefix1"], norm_info["prefix2"]
            self.register_buffer("norm1_weight", state[p1 + ".weight"].detach().float().clone())
            self.register_buffer("norm1_bias", state[p1 + ".bias"].detach().float().clone())
            self.register_buffer("norm2_weight", state[p2 + ".weight"].detach().float().clone())
            self.register_buffer("norm2_bias", state[p2 + ".bias"].detach().float().clone())

        if self.decoder_kind == "log_gain":
            self.register_buffer("dec_log_gain", state["decoder.log_gain"].detach().float().clone())
            db = state.get("decoder.bias", torch.zeros(1, self.act_dim))
            self.register_buffer("dec_bias", db.detach().float().clone())
        elif self.decoder_kind == "weight":
            self.register_buffer("dec_weight", state["decoder.weight"].detach().float().clone())
            db = state.get("decoder.bias", torch.zeros(1, self.act_dim))
            self.register_buffer("dec_bias", db.detach().float().clone())

        # deterministic population threshold
        u = (torch.arange(self.enc_pop, dtype=torch.float32) + 0.5) / self.enc_pop
        self.register_buffer("fixed_u", u.view(1, 1, self.enc_pop))
        self.register_buffer("enc_zeros", torch.zeros(1, self.obs_dim * 2))

    def _encode(self, obs):
        x = torch.tanh(obs * self.enc_weight + self.enc_bias)
        p = torch.maximum(torch.cat([x, -x], dim=1), self.enc_zeros)
        u = self.fixed_u.expand(p.shape[0], p.shape[1], self.enc_pop)
        voltage = p.unsqueeze(-1) + u
        return voltage.gt(ENC_VTH).to(p.dtype).reshape(p.shape[0], -1)

    def _spike_decoder(self, spk3):
        if self.decoder_kind == "none":
            return torch.zeros(spk3.shape[0], self.act_dim, device=spk3.device, dtype=spk3.dtype)
        s = spk3.reshape(-1, self.act_dim * 2, self.dec_pop).mean(-1)
        signed = s[:, :self.act_dim] - s[:, self.act_dim:]
        if self.decoder_kind == "log_gain":
            gain = torch.exp(self.dec_log_gain).clamp(0.5, 4.0)
            return torch.clamp(signed * gain + self.dec_bias, -4.0, 4.0)
        # 旧decoder互換: tanh((positive-negative)*weight + bias)
        return torch.tanh(signed * self.dec_weight + self.dec_bias)

    def _motor_features(self, mem1, mem2):
        if self.norm_info["mode"] == "concat":
            cat = torch.cat([mem1, mem2], dim=1)
            normed = F.layer_norm(cat, (cat.shape[1],), self.norm_weight, self.norm_bias, 1e-5)
            if self.head_in != normed.shape[1]:
                raise RuntimeError("motor head入力次元とconcat LayerNorm出力次元が一致しません")
            return normed

        n1 = F.layer_norm(mem1, (self.h1,), self.norm1_weight, self.norm1_bias, 1e-5)
        n2 = F.layer_norm(mem2, (self.h2,), self.norm2_weight, self.norm2_bias, 1e-5)
        if self.head_in == self.h1 + self.h2:
            return torch.cat([n1, n2], dim=1)
        if self.head_in == self.h1:
            return n1
        if self.head_in == self.h2:
            return n2
        raise RuntimeError(f"未対応motor head入力次元: {self.head_in}")

    def forward(self, obs, membrane):
        spk1, mem1 = lif_step(F.linear(self._encode(obs), self.W1, self.b1), membrane[:, :self.p1])
        spk2, mem2 = lif_step(F.linear(spk1, self.W2, self.b2), membrane[:, self.p1:self.p2])
        spk3, mem3 = lif_step(F.linear(spk2, self.W3, self.b3), membrane[:, self.p2:])

        motor = F.linear(self._motor_features(mem1, mem2), self.head_weight, self.head_bias)
        spike_residual = self._spike_decoder(spk3)
        action = motor + self.residual_scale * spike_residual
        action = torch.clamp(action, -self.action_mean_clip, self.action_mean_clip)
        mem_out = torch.cat([mem1, mem2, mem3], dim=1)
        return action, mem_out

    def describe(self):
        if self.norm_info["mode"] == "concat":
            norm_text = self.norm_info["prefix"]
        else:
            norm_text = f"{self.norm_info['prefix1']} + {self.norm_info['prefix2']}"
        print(f"v5 Actor: obs={self.obs_dim}, action={self.act_dim}, mem={self.mem_dim}")
        print(f"  hidden=({self.h1}, {self.h2}, {self.out_spikes}), enc_pop={self.enc_pop}, dec_pop={self.dec_pop}")
        print(f"  membrane norm: {norm_text}")
        print(f"  motor head   : {self.head_prefix} (in={self.head_in} -> 12)")
        print(f"  spike decoder: {self.decoder_kind}, residual_scale={self.residual_scale:g} ({self.residual_scale_source})")
        print(f"  mean clip    : ±{self.action_mean_clip:g} ({self.action_clip_source})")


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
    return sorted(dirs, key=lambda p: p.stat().st_mtime, reverse=True)


def find_actor_checkpoint():
    if ACTOR_CHECKPOINT is not None:
        p = Path(ACTOR_CHECKPOINT)
        p = p if p.is_absolute() else WORK_DIR / p
        if not p.is_file():
            raise FileNotFoundError(f"ACTOR_CHECKPOINTがありません: {p}")
        return p

    checked = []
    for folder in candidate_param_dirs():
        patterns = ("*_actor_*Kit.pt", "*_actor_*.pt", "*_full_*Kit.pt", "*_full_*.pt")
        candidates = []
        for pat in patterns:
            candidates.extend(folder.glob(pat))
        candidates = sorted(set(candidates), key=lambda p: (numeric_suffix(p), p.stat().st_mtime), reverse=True)
        for p in candidates:
            try:
                state, _ = extract_actor_state_and_config(torch_load(p))
                dim = infer_obs_dim_from_state(state)
                checked.append((p, dim))
                if dim == EXPECTED_OBS_DIM:
                    # v5構造まで事前検査する。45D等を誤ロードしない。
                    ViewerV5SpikeActor(state, {}).describe()
                    return p
            except Exception as exc:
                print(f"checkpoint候補をスキップ: {p.name}: {exc}")
    details = "\n".join(f"  {p} -> obs_dim={d}" for p, d in checked[:30])
    raise FileNotFoundError(
        f"{EXPECTED_OBS_DIM}次元v5 Actorが見つかりません。ACTOR_CHECKPOINTを指定してください。"
        + ("\n確認した候補:\n" + details if details else "")
    )


def find_norm_checkpoint(actor_path):
    if NORM_CHECKPOINT is not None:
        p = Path(NORM_CHECKPOINT)
        p = p if p.is_absolute() else WORK_DIR / p
        if not p.is_file():
            raise FileNotFoundError(f"NORM_CHECKPOINTがありません: {p}")
        return p

    expected = actor_path.with_name(actor_path.name.replace("_actor_", "_norm_").replace("_full_", "_norm_").replace(".pt", ".npz"))
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
            if int(np.asarray(z["mean"]).size) == EXPECTED_OBS_DIM and int(np.asarray(z["var"]).size) == EXPECTED_OBS_DIM:
                return p
        except Exception:
            pass
    raise FileNotFoundError(
        f"対応する{EXPECTED_OBS_DIM}次元norm npzが見つかりません。NORM_CHECKPOINTを指定してください。"
    )


actor_path = find_actor_checkpoint()
checkpoint_obj = torch_load(actor_path)
actor_state, actor_config = extract_actor_state_and_config(checkpoint_obj)
actual_obs_dim = infer_obs_dim_from_state(actor_state)
if actual_obs_dim != EXPECTED_OBS_DIM:
    raise ValueError(f"このビューアは{EXPECTED_OBS_DIM}次元v5用ですが、checkpointは{actual_obs_dim}次元です: {actor_path}")

try:
    actor = ViewerV5SpikeActor(actor_state, actor_config).to(TORCH_DEVICE).eval()
except Exception as exc:
    raise RuntimeError(
        "v5 Actor構造をcheckpointから復元できませんでした。\n"
        f"原因: {exc}\n\nstate_dict keys/shapes:\n{_state_shapes(actor_state)}"
    ) from exc
for parameter in actor.parameters():
    parameter.requires_grad_(False)

norm_path = find_norm_checkpoint(actor_path)
norm_npz = np.load(norm_path)
obs_mean = torch.as_tensor(norm_npz["mean"], dtype=torch.float32, device=TORCH_DEVICE).reshape(-1)
obs_var = torch.as_tensor(norm_npz["var"], dtype=torch.float32, device=TORCH_DEVICE).reshape(-1)
if obs_mean.numel() != EXPECTED_OBS_DIM or obs_var.numel() != EXPECTED_OBS_DIM:
    raise ValueError(
        f"norm次元が不一致です: mean={obs_mean.numel()}, var={obs_var.numel()}, expected={EXPECTED_OBS_DIM}"
    )
if torch.any(obs_var < 0):
    raise ValueError("norm varに負値があります")

print("Actor:", actor_path)
print("Norm :", norm_path)
actor.describe()


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
        return torch.clamp(action[0], -ACTION_CLIP, ACTION_CLIP).cpu().numpy()


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
        self.phase_step = 0
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
  vx = ±0.60 m/s, vy = ±0.60 m/s, wz = ±1.00 rad/s
""")

    def status(self):
        print(
            f"地形={self.terrain:6s}  "
            f"cmd=[vx {self.command[0]:+.2f}, vy {self.command[1]:+.2f}, wz {self.command[2]:+.2f}]  "
            f"phase={self.phase_step:02d}/{GAIT_CYCLE_STEPS}"
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
        if model.nu != 12:
            raise RuntimeError(f"Go2の12 actuatorを想定していますが nu={model.nu} です")

        # 学習Notebookと同じtorque limit選択:
        # actuator_forcerange -> ctrlrange -> Go2 URDF fallback の順。
        force_range = np.asarray(model.actuator_forcerange, dtype=np.float64)
        force_limited = np.asarray(model.actuator_forcelimited).reshape(-1) > 0
        force_lim = np.max(np.abs(force_range), axis=1)
        ctrl_range = np.asarray(model.actuator_ctrlrange, dtype=np.float64)
        ctrl_limited = np.asarray(model.actuator_ctrllimited).reshape(-1) > 0
        ctrl_lim = np.max(np.abs(ctrl_range), axis=1)
        valid_force = force_limited & np.isfinite(force_lim) & (force_lim > 1.0) & (force_lim < 1e4)
        valid_ctrl = ctrl_limited & np.isfinite(ctrl_lim) & (ctrl_lim > 1.0) & (ctrl_lim < 1e4)
        torque_limits = np.where(valid_force, force_lim, np.where(valid_ctrl, ctrl_lim, FALLBACK_TORQUE_LIMIT))
        model.actuator_ctrllimited[:] = 1
        model.actuator_ctrlrange[:, 0] = -torque_limits
        model.actuator_ctrlrange[:, 1] = torque_limits

        model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
        model.opt.iterations = 8
        if hasattr(model.opt, "ls_iterations"):
            model.opt.ls_iterations = 8
        print("torque limit [Nm]:", np.array2string(torque_limits, precision=2))
        return model, data, torque_limits

    def reset_robot(self, model, data):
        mujoco.mj_resetData(model, data)
        data.qpos[0:3] = [0.0, 0.0, H0]
        data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        data.qpos[7:19] = standing_pose_viewer()
        data.qvel[:] = 0.0
        self.last_action[:] = 0.0
        self.phase_step = 0
        self.policy.reset()
        mujoco.mj_forward(model, data)

    def observation(self, model, data, default_sensor, addresses):
        joint_angle = data.sensordata[addresses["pos"]:addresses["pos"] + 12] - default_sensor
        joint_velocity = data.sensordata[addresses["vel"]:addresses["vel"] + 12]
        gyro = data.sensordata[addresses["gyro"]:addresses["gyro"] + 3]
        # v5 47D: APEX標準45D + gait phase sin/cos。
        # phaseはAction Prior/模倣軌道の周期 T_GAIT=0.40s に合わせ、制御stepごとに進める。
        projected_gravity = projected_gravity_from_quat(data.qpos[3:7])
        phase = 2.0 * np.pi * (self.phase_step % GAIT_CYCLE_STEPS) / GAIT_CYCLE_STEPS
        phase_obs = np.array([np.sin(phase), np.cos(phase)], dtype=np.float32)
        obs = np.concatenate([
            gyro * 0.25,
            projected_gravity,
            self.command * APEX_CMD_OBS_SCALE,
            joint_angle,
            joint_velocity * 0.05,
            self.last_action,
            phase_obs,
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
        model, data, torque_limits = self.prepare_model(self.scenes[self.terrain])
        default_qpos = standing_pose_viewer()
        default_sensor = default_qpos[PERM]
        joint_low = model.jnt_range[1:, 0][PERM]
        joint_high = model.jnt_range[1:, 1][PERM]
        addresses = {
            "pos": self.sensor_address(model, "FR_hip_pos"),
            "vel": self.sensor_address(model, "FR_hip_vel"),
            "gyro": self.sensor_address(model, "imu_gyro"),

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
                                    f"Go2 APEX v5 {EXPECTED_OBS_DIM}D keyboard viewer", None, None)
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
                    action = np.clip(action, -ACTION_CLIP, ACTION_CLIP)
                    target = np.clip(default_sensor + action * ACT_SCALE, joint_low, joint_high)

                    for _ in range(frame_skip):
                        position = data.qpos[7:19][PERM]
                        velocity = data.qvel[6:18][PERM]
                        torque = SERVO_KP * (target - position) - SERVO_KD * velocity
                        data.ctrl[:] = np.clip(torque, -torque_limits, torque_limits)
                        mujoco.mj_step(model, data)

                    # 次stepのprevious actionとphase。学習環境と同じく制御後にphaseを1つ進める。
                    self.last_action[:] = action
                    self.phase_step = (self.phase_step + 1) % GAIT_CYCLE_STEPS
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
