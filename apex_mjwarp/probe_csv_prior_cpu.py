"""CPU MuJoCo check of official canter CSV and additive APEX joint prior."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from go2_mjcf import build_go2_mjcf
from apex_paths import find_apex_root


JOINTS = tuple(f"{leg}_{part}_joint" for leg in ("FL", "FR", "RL", "RR")
               for part in ("hip", "thigh", "calf"))
DEFAULT = np.array([0.1, 0.8, -1.5, -0.1, 0.8, -1.5,
                    0.1, 1.0, -1.5, -0.1, 1.0, -1.5])


def run(apex_root: Path, steps: int = 200, scene: Path | None = None,
        enable_prior: bool = True) -> dict:
    urdf = apex_root / "resources/robots/go2/urdf/go2.urdf"
    motion_path = apex_root / "imitation_data/animal_mocap/go2_retarget_canter_2ms.csv"
    model = (mujoco.MjModel.from_xml_path(str(scene)) if scene is not None
             else mujoco.MjModel.from_xml_string(build_go2_mjcf(urdf)))
    data = mujoco.MjData(model)
    with motion_path.open(newline="", encoding="utf-8") as handle:
        csv_rows = list(csv.reader(handle))
    motion = np.asarray(csv_rows[1:], dtype=np.float64)
    q_idx = np.array([model.jnt_qposadr[model.joint(name).id] for name in JOINTS])
    v_idx = np.array([model.jnt_dofadr[model.joint(name).id] for name in JOINTS])
    a_idx = np.array([model.actuator(name if scene is None else name.removesuffix("_joint")).id
                      for name in JOINTS])
    urdf_root = ET.parse(urdf).getroot()
    effort = {joint.get("name"): float(joint.find("limit").get("effort"))
              for joint in urdf_root.findall("joint") if joint.get("type") == "revolute"}
    limit = np.array([effort[name] for name in JOINTS])
    data.qpos[2] = 0.35
    data.qpos[3] = 1.0
    data.qpos[q_idx] = DEFAULT
    mujoco.mj_forward(model, data)
    vx, heights = [], []
    for control_step in range(steps):
        ref = motion[min(control_step, len(motion) - 1), 6:18]
        prior = (0.99 ** (control_step / 100)) if enable_prior else 0.0
        for _ in range(4):
            q = data.qpos[q_idx]
            dq = data.qvel[v_idx]
            torque = 20.0 * (DEFAULT - q) - 0.5 * dq + prior * 20.0 * (ref - q)
            data.ctrl[a_idx] = np.clip(torque, -limit, limit)
            mujoco.mj_step(model, data)
        vx.append(float(data.qvel[0]))
        heights.append(float(data.qpos[2]))
        if data.qpos[2] < 0.18:
            break
    result = {"steps": len(vx), "mean_vx": float(np.mean(vx)),
              "peak_vx": float(np.max(vx)), "final_x": float(data.qpos[0]),
              "min_height": float(np.min(heights))}
    print(result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apex-root", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--scene", type=Path, default=None)
    parser.add_argument("--without-prior", action="store_true")
    args = parser.parse_args()
    run(find_apex_root(args.apex_root), args.steps,
        args.scene.resolve() if args.scene is not None else None,
        enable_prior=not args.without_prior)
