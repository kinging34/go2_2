"""Evaluate a trained SNN APEX Actor with Action Prior disabled."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys

import mujoco
import torch

from env_snn import ApexGo2Warp
from apex_paths import find_apex_root
from snn_actor import build_actor_critic_class


def load_policy(apex_root: Path, checkpoint: Path, snn_steps: int, device: str):
    os.chdir(apex_root)
    sys.path.insert(0, str(apex_root / "rsl_rl"))
    from rsl_rl.modules import MultiCriticActorCritic

    cls = build_actor_critic_class(MultiCriticActorCritic)
    policy = cls(45, 77, 12, num_critics=2,
                 actor_hidden_dims=[512, 256, 128],
                 critic_hidden_dims=[512, 256, 128],
                 activation="elu", init_noise_std=1.0,
                 snn_hidden_sizes=(256, 256), encoder_pop_dim=64,
                 decoder_pop_dim=256, snn_steps=snn_steps).to(device)
    saved = torch.load(checkpoint, map_location=device)
    policy.load_state_dict(saved["model_state_dict"])
    policy.eval()
    return policy, saved.get("iter")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apex-root", default=None, type=Path,
                        help="Existing apex_mjwarp/APEX checkout; found automatically when omitted")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("apex_snn_prior_off_eval.csv"))
    parser.add_argument("--video", type=Path, default=None)
    parser.add_argument("--vx", type=float, default=1.0,
                        help="Official canter training commands use positive forward speeds")
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--wz", type=float, default=0.0)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--snn-steps", type=int, default=None)
    parser.add_argument("--motion", default=None)
    args = parser.parse_args()
    apex_root = find_apex_root(args.apex_root)
    checkpoint = args.checkpoint.resolve()
    metadata_path = checkpoint.parent / "apex_snn_run.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
    snn_steps = args.snn_steps or metadata.get("snn_steps", 4)
    motion = args.motion or metadata.get("motion", "imitation_data/animal_mocap/go2_retarget_canter_2ms.csv")
    if args.snn_steps is not None and metadata and args.snn_steps != metadata.get("snn_steps"):
        parser.error("--snn-steps differs from the checkpoint's run metadata")
    output = args.output.resolve()
    video = args.video.resolve() if args.video else None
    output.parent.mkdir(parents=True, exist_ok=True)
    if video:
        video.parent.mkdir(parents=True, exist_ok=True)
    if args.steps <= 0:
        parser.error("--steps must be positive")

    policy, iteration = load_policy(apex_root, checkpoint, snn_steps, "cuda:0")
    env = ApexGo2Warp(apex_root, num_envs=1, motion=motion,
                      add_noise=False, enable_prior=False,
                      domain_randomization=False, resample_commands=False,
                      push_robots=False, auto_reset=False)
    env.reset()
    env.commands[0, :3] = torch.tensor([args.vx, args.vy, args.wz], device=env.device)
    env._observations()

    image_writer = None
    renderer = None
    render_data = None
    if video:
        import imageio.v2 as imageio

        image_writer = imageio.get_writer(video, fps=round(1.0 / env.dt))
        renderer = mujoco.Renderer(env.cpu_model, width=640, height=480)
        render_data = mujoco.MjData(env.cpu_model)

    rows = []
    try:
        for step_no in range(args.steps):
            with torch.no_grad():
                action = policy.act_inference(env.get_observations())
                _, _, rewards, dones, _ = env.step(action)
            row = {
                "step": step_no,
                "time_s": (step_no + 1) * env.dt,
                "command_vx": args.vx,
                "command_vy": args.vy,
                "command_wz": args.wz,
                "vx_body": float(env.base_lin_vel[0, 0]),
                "vy_body": float(env.base_lin_vel[0, 1]),
                "wz_body": float(env.base_ang_vel[0, 2]),
                "mean_action_abs": float(action[0].abs().mean()),
                "mean_torque_abs": float(env.torques[0].abs().mean()),
                "reward_style": float(rewards[0, 0]),
                "reward_task": float(rewards[0, 1]),
                "done": bool(dones[0]),
            }
            rows.append(row)
            if image_writer:
                render_data.qpos[:] = env.qpos[0].detach().cpu().numpy()
                render_data.qvel[:] = env.qvel[0].detach().cpu().numpy()
                mujoco.mj_forward(env.cpu_model, render_data)
                renderer.update_scene(render_data)
                image_writer.append_data(renderer.render())
            if bool(dones[0]):
                break
    finally:
        if image_writer:
            image_writer.close()
        if renderer:
            renderer.close()

    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    n = len(rows)
    avg = lambda key: sum(row[key] for row in rows) / n
    print(f"Prior OFF: checkpoint iter={iteration}, steps={n}/{args.steps}, "
          f"vx={avg('vx_body'):+.3f}, vy={avg('vy_body'):+.3f} m/s, "
          f"wz={avg('wz_body'):+.3f} rad/s, "
          f"|action|={avg('mean_action_abs'):.3f}, "
          f"|torque|={avg('mean_torque_abs'):.2f} Nm")
    print(f"Trace: {output}")
    if video:
        print(f"Video: {video}")


if __name__ == "__main__":
    main()
