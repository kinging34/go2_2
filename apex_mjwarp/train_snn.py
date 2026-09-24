"""Train a Go2 SNN Actor using APEX's unchanged Multi-Critic PPO runner."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

import torch

from env_snn import ApexGo2Warp
from apex_paths import find_apex_root
from snn_actor import build_actor_critic_class


def load_official_runner(apex_root: Path):
    sys.path.insert(0, str(apex_root / "rsl_rl"))
    from rsl_rl.modules import MultiCriticActorCritic

    runner_path = apex_root / "rsl_rl/rsl_rl/runners/on_policy_runner.py"
    spec = importlib.util.spec_from_file_location("apex_snn_on_policy_runner", runner_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load APEX runner: {runner_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.SNNMultiCriticActorCritic = build_actor_critic_class(MultiCriticActorCritic)
    return module.OnPolicyRunner


def make_runner(apex_root: Path, num_envs: int, log_dir: Path,
                motion: str, snn_steps: int):
    apex_root = apex_root.resolve()
    if not (apex_root / "legged_gym/envs/param_config.yaml").is_file():
        raise FileNotFoundError(f"Not an APEX repository: {apex_root}")
    if not (apex_root / motion).is_file():
        raise FileNotFoundError(f"Motion CSV not found: {apex_root / motion}")
    log_dir = log_dir.resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("WANDB_MODE", "offline")
    torch.manual_seed(39)

    # The APEX storage and runner read this relative configuration file.
    os.chdir(apex_root)
    OnPolicyRunner = load_official_runner(apex_root)
    env = ApexGo2Warp(apex_root, num_envs=num_envs, motion=motion)
    cfg = {
        "policy": {
            "num_critics": 2,
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "activation": "elu",
            "init_noise_std": 1.0,
            "snn_hidden_sizes": [256, 256],
            "encoder_pop_dim": 64,
            "decoder_pop_dim": 256,
            "snn_steps": snn_steps,
        },
        "algorithm": {
            "value_loss_coef": 1.0,
            "use_clipped_value_loss": True,
            "clip_param": 0.2,
            "entropy_coef": 0.01,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "learning_rate": 1e-3,
            "schedule": "adaptive",
            "gamma": 0.99,
            "lam": 0.95,
            "desired_kl": 0.01,
            "max_grad_norm": 1.0,
        },
        "runner": {
            "policy_class_name": "SNNMultiCriticActorCritic",
            "algorithm_class_name": "MultiCriticPPO",
            "num_steps_per_env": 24,
            "save_interval": 200,
            "experiment_name": "apex_go2_snn_mjwarp",
            "run_name": "paper_snn_actor_64x256",
            "critic_num": 2,
        },
    }
    (log_dir / "apex_snn_run.json").write_text(json.dumps({
        "apex_root": str(apex_root),
        "motion": motion,
        "num_envs": num_envs,
        "snn_hidden_sizes": [256, 256],
        "encoder_pop_dim": 64,
        "decoder_pop_dim": 256,
        "snn_steps": snn_steps,
        "actor_obs": 45,
        "critic_obs": 77,
        "algorithm": "official MultiCriticPPO",
    }, indent=2), encoding="utf-8")
    runner = OnPolicyRunner(env, cfg, log_dir=str(log_dir), device=str(env.device))
    return env, runner


def check_actor_path(env, runner):
    actor = runner.alg.actor_critic.actor
    obs = env.get_observations()[:4]
    mean = actor(obs)
    head_grad, first_grad = torch.autograd.grad(
        mean.square().mean(),
        (actor.motor_readout.weight, actor.Linear1.weight),
        allow_unused=False,
    )
    mean_abs = float(mean.detach().abs().mean())
    head_grad_abs = float(head_grad.detach().abs().mean())
    first_grad_abs = float(first_grad.detach().abs().mean())
    print(f"SNN Actor preflight: |mean|={mean_abs:.4f}, "
          f"head_grad={head_grad_abs:.2e}, first_grad={first_grad_abs:.2e}, "
          f"spike_rates={actor.last_spike_rates.tolist()}")
    if not torch.isfinite(mean).all() or mean_abs < 0.03:
        raise RuntimeError("SNN motor mean is zero or non-finite before PPO")
    if head_grad_abs < 1e-10 or first_grad_abs < 1e-12:
        raise RuntimeError("PPO gradient does not reach the SNN motor readout and first layer")


def check_prior_motion(apex_root: Path, motion: str) -> bool:
    """Measure prior-only behavior without assuming a policy-free gait must survive."""
    probe = ApexGo2Warp(apex_root, num_envs=1, motion=motion,
                        add_noise=False, enable_prior=True,
                        domain_randomization=False, resample_commands=False,
                        push_robots=False, auto_reset=False)
    probe.commands[0, :3] = torch.tensor([1.0, 0.0, 0.0], device=probe.device)
    probe._observations()
    zeros = torch.zeros(1, 12, device=probe.device)
    speeds = []
    survived = 0
    with torch.no_grad():
        for _ in range(200):
            _, _, _, done, _ = probe.step(zeros)
            speeds.append(float(probe.base_lin_vel[0, 0]))
            survived += 1
            if bool(done[0]):
                break
    mean_speed = sum(speeds) / len(speeds)
    print(f"Prior-only preflight: survived={survived}/200, "
          f"mean vx={mean_speed:+.3f} m/s, peak vx={max(speeds):+.3f} m/s")
    success = survived >= 100 and max(speeds) >= 0.10
    if not success:
        print("Prior-only probe did not sustain locomotion. APEX's prior also "
              "needs policy actions; inspect the Warp contact/PD behavior before a long run.")
    del probe
    return success


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apex-root", type=Path, default=None,
                        help="Existing apex_mjwarp/APEX checkout; found automatically when omitted")
    parser.add_argument("--num-envs", type=int, default=1024,
                        help="4096 reproduces the official environment count; 1024 uses less GPU memory")
    parser.add_argument("--iterations", type=int, default=1200)
    parser.add_argument("--log-dir", type=Path, default=Path("runs/apex_go2_snn_paper"))
    parser.add_argument("--motion", default="imitation_data/animal_mocap/go2_retarget_canter_2ms.csv")
    parser.add_argument("--snn-steps", type=int, default=4)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--smoke", action="store_true", help="Run one PPO update with 64 worlds")
    parser.add_argument("--require-prior-motion", action="store_true",
                        help="Abort if the independent prior-only probe fails")
    args = parser.parse_args()
    if args.num_envs <= 0 or args.iterations <= 0:
        parser.error("--num-envs and --iterations must be positive")
    if args.smoke:
        args.num_envs, args.iterations = 64, 1

    apex_root = find_apex_root(args.apex_root)
    print(f"Using existing APEX checkout: {apex_root}")
    prior_ok = check_prior_motion(apex_root, args.motion)
    if args.require_prior_motion and not prior_ok:
        raise RuntimeError("Prior-only motion requirement failed")
    env, runner = make_runner(apex_root, args.num_envs, args.log_dir,
                              args.motion, args.snn_steps)
    if args.resume is not None:
        runner.load(str(args.resume.resolve()))
        env.torque_ref_decay_factor = runner.current_learning_iteration * runner.num_steps_per_env
        env.common_step_counter = env.torque_ref_decay_factor
        print(f"Resumed at iteration {runner.current_learning_iteration}; "
              f"DecAP step={env.torque_ref_decay_factor}")
    check_actor_path(env, runner)
    print(f"APEX Multi-Critic PPO: worlds={args.num_envs}, "
          f"iterations={args.iterations}, rollout=24, "
          f"SNN={runner.alg.actor_critic.actor.hidden_sizes}, "
          f"encoder=64, decoder=256, ticks={args.snn_steps}")
    runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=False)


if __name__ == "__main__":
    main()
