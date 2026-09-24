"""Go2 APEX flat-terrain VecEnv backed by MuJoCo Warp.

Keep the original APEX rsl_rl package as the learner. This module replaces
only the Isaac Gym environment for the repository's default Go2 configuration.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import torch
import mujoco
import mujoco_warp as mjw
import warp as wp
import xml.etree.ElementTree as ET

from go2_mjcf import build_go2_mjcf

LEG_ORDER = ("FL", "FR", "RL", "RR")
JOINTS = tuple(f"{leg}_{part}_joint" for leg in LEG_ORDER for part in ("hip", "thigh", "calf"))
FEET = tuple(f"{leg}_foot" for leg in LEG_ORDER)


def quat_rotate_inverse_xyzw(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    xyz, w = q[..., :3], q[..., 3:4]
    return v * (2 * w.square() - 1) - 2 * w * torch.cross(xyz, v, dim=-1) + 2 * xyz * (xyz * v).sum(-1, keepdim=True)


def yaw_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    yaw = torch.atan2(2 * (q[:, 3] * q[:, 2] + q[:, 0] * q[:, 1]),
                      1 - 2 * (q[:, 1].square() + q[:, 2].square()))
    c, s = torch.cos(yaw)[:, None], torch.sin(yaw)[:, None]
    return torch.stack((c * v[..., 0] + s * v[..., 1],
                        -s * v[..., 0] + c * v[..., 1], v[..., 2]), dim=-1)


class ApexGo2Warp:
    num_obs = 45
    num_privileged_obs = 77
    num_actions = 12
    dt = 0.02
    decimation = 4

    def __init__(self, apex_root: str | Path, num_envs: int = 256, device: str = "cuda:0",
                 motion: str = "imitation_data/animal_mocap/go2_retarget_canter_2ms.csv"):
        if not torch.cuda.is_available():
            raise RuntimeError("MuJoCo Warp training requires an NVIDIA CUDA GPU")
        self.device = torch.device(device)
        wp.init()
        wp.set_device(str(self.device))
        self.apex_root = Path(apex_root).resolve()
        urdf = self.apex_root / "resources/robots/go2/urdf/go2.urdf"
        self.cpu_model = mujoco.MjModel.from_xml_string(build_go2_mjcf(urdf))
        self.model = mjw.put_model(self.cpu_model)
        self.data = mjw.make_data(self.cpu_model, nworld=num_envs)
        self.qpos = wp.to_torch(self.data.qpos)
        self.qvel = wp.to_torch(self.data.qvel)
        self.ctrl = wp.to_torch(self.data.ctrl)
        self.num_envs = num_envs
        self.joint_qpos = [int(self.cpu_model.jnt_qposadr[self.cpu_model.joint(n).id]) for n in JOINTS]
        self.joint_qvel = [int(self.cpu_model.jnt_dofadr[self.cpu_model.joint(n).id]) for n in JOINTS]
        self.motor_ids = [self.cpu_model.actuator(n).id for n in JOINTS]
        self.foot_ids = [self.cpu_model.body(n).id for n in FEET]
        self.foot_root_ids = [int(self.cpu_model.body_rootid[i]) for i in self.foot_ids]
        self.penalty_ids = [i for i in range(1, self.cpu_model.nbody)
                            if any(s in self.cpu_model.body(i).name for s in ("base", "hip", "thigh", "calf", "trunk"))]
        self.termination_ids = [i for i in range(1, self.cpu_model.nbody)
                                if any(s in self.cpu_model.body(i).name for s in ("base", "hip"))]
        urdf_root = ET.parse(urdf).getroot()
        effort = {j.get("name"): float(j.find("limit").get("effort"))
                  for j in urdf_root.findall("joint") if j.get("type") == "revolute"}
        self.torque_limits = torch.tensor([effort[n] for n in JOINTS], device=self.device)
        default = {"hip": (0.1, -0.1, 0.1, -0.1),
                   "thigh": (0.8, 0.8, 1.0, 1.0), "calf": (-1.5,) * 4}
        self.default_dof_pos = torch.tensor([default[part][i] for i in range(4)
                                             for part in ("hip", "thigh", "calf")], device=self.device)
        self.p_gains = torch.full((12,), 20.0, device=self.device)
        self.d_gains = torch.full((12,), 0.5, device=self.device)
        self.actions = torch.zeros(num_envs, 12, device=self.device)
        self.last_actions = torch.zeros_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.actions)
        self.torques = torch.zeros_like(self.actions)
        self.commands = torch.zeros(num_envs, 4, device=self.device)
        self.commands_scale = torch.tensor([2.0, 2.0, 0.25], device=self.device)
        self.motor_offsets = torch.zeros_like(self.actions)
        self.Kp_factors = torch.ones(num_envs, 1, device=self.device)
        self.Kd_factors = torch.ones(num_envs, 1, device=self.device)
        self.motor_strengths = torch.ones(num_envs, 1, device=self.device)
        self.decap_factor = torch.ones(num_envs, 1, device=self.device)
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.imitation_index = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.reset_buf = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.time_out_buf = torch.zeros_like(self.reset_buf)
        self.last_foot_velocities = torch.zeros(num_envs, 4, 3, device=self.device)
        self.last_contacts = torch.zeros(num_envs, 4, dtype=torch.bool, device=self.device)
        self.global_training_iteration = 0
        self.torque_ref_decay_factor = 0
        self.common_step_counter = 0
        self.motion = torch.tensor(pd.read_csv(self.apex_root / motion).to_numpy(dtype=np.float32), device=self.device)
        if self.motion.shape[1] < 40:
            raise ValueError("Go2 APEX imitation CSV needs at least 40 columns")
        self.max_episode_length = self.motion.shape[0] - 1
        self.max_episode_length_s = self.max_episode_length * self.dt
        self.obs_buf = torch.zeros(num_envs, 45, device=self.device)
        self.privileged_obs_buf = torch.zeros(num_envs, 77, device=self.device)
        self.rew_buf = torch.zeros(num_envs, 2, device=self.device)
        self.extras = {}
        self.reset()
        # The standing target is calculated once from the default pose.
        self.default_foot_pos_body_frame = yaw_rotate_inverse(self.base_quat,
             self.foot_positions - self.base_pos[:, None, :]).reshape(num_envs, 12).clone()

    @property
    def dof_pos(self):
        return self.qpos[:, self.joint_qpos]

    @property
    def dof_vel(self):
        return self.qvel[:, self.joint_qvel]

    def _refresh(self):
        self.base_pos = self.qpos[:, :3]
        self.base_quat = torch.cat((self.qpos[:, 4:7], self.qpos[:, 3:4]), dim=1)
        self.base_lin_vel = quat_rotate_inverse_xyzw(self.base_quat, self.qvel[:, :3])
        # MuJoCo free-joint qvel stores angular velocity in the body frame.
        self.base_ang_vel = self.qvel[:, 3:6]
        self.projected_gravity = quat_rotate_inverse_xyzw(self.base_quat,
            torch.tensor([0., 0., -1.], device=self.device).expand(self.num_envs, 3))
        self.foot_positions = wp.to_torch(self.data.xpos)[:, self.foot_ids, :]
        # cvel uses the root subtree COM as its reference point and stores
        # (angular, linear) world velocities; shift to each foot body origin.
        cvel = wp.to_torch(self.data.cvel)[:, self.foot_ids, :]
        subtree_com = wp.to_torch(self.data.subtree_com)[:, self.foot_root_ids, :]
        self.foot_velocities = cvel[..., 3:6] + torch.cross(
            cvel[..., :3], self.foot_positions - subtree_com, dim=-1)
        self.contact_forces = wp.to_torch(self.data.cfrc_ext)[..., 3:6]

    def _sample_commands(self, ids):
        if ids.numel() == 0:
            return
        ref = self.motion[self.imitation_index[ids].clamp(0, self.max_episode_length)]
        self.commands[ids, 0] = (ref[:, 18] + 2 * torch.rand(len(ids), device=self.device) - 1).clamp(0, 3)
        self.commands[ids, 1] = (ref[:, 19] + .2 * torch.rand(len(ids), device=self.device) - .1).clamp(-.1, .1)
        self.commands[ids, 2] = 3 * torch.rand(len(ids), device=self.device) - 1.5
        moving = torch.linalg.norm(self.commands[ids, :2], dim=1) > .1
        self.commands[ids, :2] *= moving[:, None]

    def reset_idx(self, ids):
        ids = torch.as_tensor(ids, device=self.device, dtype=torch.long).flatten()
        if ids.numel() == 0:
            return
        n = len(ids)
        self.qpos[ids] = 0
        self.qpos[ids, :3] = torch.tensor([0., 0., .35], device=self.device)
        self.qpos[ids, 3] = 1
        self.qpos[ids[:, None], torch.tensor(self.joint_qpos, device=self.device)] = (
            self.default_dof_pos * (.5 + torch.rand(n, 12, device=self.device)))
        self.qvel[ids] = 0
        self.qvel[ids, :6] = torch.rand(n, 6, device=self.device) - .5
        self.ctrl[ids] = 0
        self.motor_offsets[ids] = -.035 + .07 * torch.rand(n, 12, device=self.device)
        self.Kp_factors[ids] = .9 + .2 * torch.rand(n, 1, device=self.device)
        self.Kd_factors[ids] = .9 + .2 * torch.rand(n, 1, device=self.device)
        self.last_actions[ids] = 0
        self.last_dof_vel[ids] = 0
        self.last_foot_velocities[ids] = 0
        self.last_contacts[ids] = False
        self.episode_length_buf[ids] = 0
        self.imitation_index[ids] = 0
        self._sample_commands(ids)
        mjw.forward(self.model, self.data)
        self._refresh()

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self.reset_idx(env_ids)
        self._observations()
        return self.obs_buf, self.privileged_obs_buf

    def _observations(self):
        ref = self.motion[self.imitation_index.clamp(0, self.max_episode_length)]
        phase = (self.imitation_index.float() / self.motion.shape[0])[:, None]
        dof_offset = self.dof_pos - self.default_dof_pos
        common = (self.projected_gravity, self.commands[:, :3] * self.commands_scale,
                  dof_offset, self.dof_vel * .05, self.actions)
        self.obs_buf = torch.cat((self.base_ang_vel * .25, *common), dim=1)
        self.privileged_obs_buf = torch.cat((self.base_lin_vel * 2, self.base_ang_vel * .25,
            *common, phase, ref[:, 6:18], ref[:, 22:34], ref[:, 36:40]), dim=1)
        if self.obs_buf.shape[1] != 45 or self.privileged_obs_buf.shape[1] != 77:
            raise AssertionError("APEX observation shape drift")
        # Mirror the original 45-dimension noise layout, including its indexing.
        noise = torch.zeros(45, device=self.device)
        noise[:3] = .1 * 2
        noise[3:6] = .2 * .25
        noise[6:9] = .1
        noise[12:24] = .02
        noise[24:36] = 1.5 * .05
        self.obs_buf = (self.obs_buf + (2 * torch.rand_like(self.obs_buf) - 1) * noise).clamp(-100, 100)
        self.privileged_obs_buf = self.privileged_obs_buf.clamp(-100, 100)

    def _rewards(self):
        ref = self.motion[self.imitation_index.clamp(0, self.max_episode_length)]
        standing = torch.linalg.norm(self.commands[:, :2], dim=1) < .1
        joint_target = torch.where(standing[:, None], self.default_dof_pos, ref[:, 6:18])
        angle = torch.exp(-((self.dof_pos - joint_target).square().mean(dim=1)) / .01)
        foot_target = torch.where(standing[:, None], self.default_foot_pos_body_frame, ref[:, 22:34])
        foot_body = yaw_rotate_inverse(self.base_quat,
            self.foot_positions - self.base_pos[:, None, :]).reshape(self.num_envs, 12)
        foot = torch.exp(-(foot_target - foot_body).square().sum(dim=1) / .01)
        quat = torch.exp(-(ref[:, 36:40] - self.base_quat).square().sum(dim=1) / .5)
        group1 = 3.5 * angle + 2.5 * foot + .5 * quat

        lin = torch.exp(-(self.commands[:, :2] - self.base_lin_vel[:, :2]).square().sum(dim=1) / .25)
        ang = torch.exp(-(self.commands[:, 2] - self.base_ang_vel[:, 2]).square() / .25)
        collision = (torch.linalg.norm(self.contact_forces[:, self.penalty_ids], dim=-1) > .1).float().sum(dim=1)
        foot_contact = self.contact_forces[:, self.foot_ids, 2] > 1
        contact_filt = foot_contact | self.last_contacts
        slip = (contact_filt * self.foot_velocities[:, :, :2].square().sum(dim=2)).sum(dim=1)
        impact = -torch.minimum((self.foot_velocities[:, :, 2] - self.last_foot_velocities[:, :, 2]).square(),
                                torch.tensor(2., device=self.device)).sum(dim=1)
        height = (self.base_pos[:, 2] - ref[:, 21]).square()
        group2 = (2 * lin + 1.5 * ang - .00001 * self.torques.square().sum(dim=1)
                  - 2.5e-7 * ((self.last_dof_vel - self.dof_vel) / self.dt).square().sum(dim=1)
                  - collision - .01 * (self.last_actions - self.actions).square().sum(dim=1)
                  - .04 * slip + .0025 * impact - 10 * height
                  - .05 * self.base_ang_vel[:, :2].square().sum(dim=1))
        self.rew_buf = self.dt * torch.stack((group1, group2), dim=1)
        self.last_contacts = foot_contact
        self.last_foot_velocities = self.foot_velocities.clone()

    def step(self, actions):
        self.actions = actions.to(self.device).clamp(-100, 100)
        for _ in range(self.decimation):
            ref = self.motion[self.imitation_index.clamp(0, self.max_episode_length), 6:18]
            factor = .99 ** (self.torque_ref_decay_factor / 100)
            self.decap_factor.fill_(factor)
            target = self.default_dof_pos + .25 * self.actions
            self.torques = (self.p_gains * self.Kp_factors * (target - self.dof_pos + self.motor_offsets)
                - self.d_gains * self.Kd_factors * self.dof_vel
                + factor * self.p_gains * (ref - self.dof_pos))
            self.torques = (self.torques * self.motor_strengths).clamp(-self.torque_limits, self.torque_limits)
            self.ctrl[:, self.motor_ids] = self.torques
            mjw.step(self.model, self.data)
        # step() integrates qpos after forward dynamics. Recompute derived body
        # poses and contact outputs for the newly integrated state.
        mjw.forward(self.model, self.data)
        self._refresh()
        self.episode_length_buf += 1
        self.common_step_counter += 1
        if self.common_step_counter % 250 == 0:
            self._sample_commands(torch.arange(self.num_envs, device=self.device))
        if self.common_step_counter % 200 == 0:
            self.qvel[:, :2] = 1.2 * torch.rand(self.num_envs, 2, device=self.device) - .6
            self.qvel[:, 3:6] = 1.6 * torch.rand(self.num_envs, 3, device=self.device) - .8
            mjw.forward(self.model, self.data)
            self._refresh()
        self.time_out_buf = self.episode_length_buf > self.max_episode_length
        terminate = (torch.linalg.norm(self.contact_forces[:, self.termination_ids], dim=-1) > 1).any(dim=1)
        self.reset_buf = terminate | self.time_out_buf
        self._rewards()
        dones, timeouts = self.reset_buf.clone(), self.time_out_buf.clone()
        extras = {"time_outs": timeouts, "decap_factor": self.decap_factor[0].item()}
        self.reset_idx(dones.nonzero(as_tuple=False).flatten())
        self._observations()
        self.last_actions = self.actions.clone()
        self.last_dof_vel = self.dof_vel.clone()
        self.imitation_index = (self.imitation_index + 1).clamp(max=self.max_episode_length)
        self.imitation_index[dones] = 0
        self.torque_ref_decay_factor += 1
        self.extras = extras
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, dones, extras

    def get_observations(self):
        return self.obs_buf

    def get_privileged_observations(self):
        return self.privileged_obs_buf
