"""Use APEX's unmodified Multi-Critic PPO runner with the Warp environment."""
from __future__ import annotations

import os
import importlib.util
from pathlib import Path
import sys
import torch

from env import ApexGo2Warp


def make_runner(apex_root: str | Path, num_envs: int = 256, log_dir: str | Path = "logs/apex_warp"):
    apex_root = Path(apex_root).resolve()
    sys.path.insert(0, str(apex_root / "rsl_rl"))
    torch.manual_seed(39)
    os.environ.setdefault("WANDB_MODE", "offline")
    # The original runner records these two source paths as W&B artifacts.
    os.chdir(apex_root)
    # Import the original runner file directly. The package __init__ eagerly
    # imports the unused AMP runner, which in turn imports Isaac Gym utilities.
    runner_file = apex_root / "rsl_rl/rsl_rl/runners/on_policy_runner.py"
    spec = importlib.util.spec_from_file_location("apex_on_policy_runner", runner_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    OnPolicyRunner = module.OnPolicyRunner
    log_dir = Path(log_dir).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    env = ApexGo2Warp(apex_root, num_envs=num_envs)
    cfg = {
        "policy": {"init_noise_std": 1.0, "actor_hidden_dims": [512, 256, 128],
                   "critic_hidden_dims": [512, 256, 128], "activation": "elu", "num_critics": 2},
        "algorithm": {"value_loss_coef": 1.0, "use_clipped_value_loss": True,
                      "clip_param": .2, "entropy_coef": .01, "num_learning_epochs": 5,
                      "num_mini_batches": 4, "learning_rate": .001, "schedule": "adaptive",
                      "gamma": .99, "lam": .95, "desired_kl": .01, "max_grad_norm": 1.0},
        "runner": {"policy_class_name": "MultiCriticActorCritic",
                   "algorithm_class_name": "MultiCriticPPO", "num_steps_per_env": 24,
                   "save_interval": 200, "experiment_name": "apex_IROS_pronk_warp",
                   "run_name": "", "critic_num": 2},
    }
    return env, OnPolicyRunner(env, cfg, log_dir=str(log_dir), device=str(env.device))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--apex-root", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--log-dir", type=Path, default=Path("logs/apex_warp"))
    args = parser.parse_args()
    env, runner = make_runner(args.apex_root, args.num_envs, args.log_dir)
    runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=False)
