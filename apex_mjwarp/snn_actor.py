"""Deterministic, stateless SNN Actor for APEX's unchanged Multi-Critic PPO.

The neuron states are reset for each policy decision and unrolled for a fixed
number of internal SNN ticks. This makes the action likelihood a function of
the stored observation alone, as required by APEX's feed-forward PPO storage.
"""

from __future__ import annotations

import torch
from torch import nn


def surrogate_spike(x: torch.Tensor, slope: float = 5.0) -> torch.Tensor:
    hard = (x > 0).to(x.dtype)
    soft = torch.sigmoid(slope * x)
    return hard + soft - soft.detach()


class SpikeActor(nn.Module):
    """Restored original SNN widths: 64 input and 256 output neurons/population."""

    def __init__(self, obs_dim: int, act_dim: int,
                 hidden_sizes: tuple[int, int] = (256, 256),
                 encoder_pop_dim: int = 64, decoder_pop_dim: int = 256,
                 snn_steps: int = 4):
        super().__init__()
        if min(*hidden_sizes, encoder_pop_dim, decoder_pop_dim, snn_steps) <= 0:
            raise ValueError("SNN sizes and tick count must be positive")
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_sizes = hidden_sizes
        self.encoder_pop_dim = encoder_pop_dim
        self.decoder_pop_dim = decoder_pop_dim
        self.snn_steps = snn_steps

        input_pop = 2 * obs_dim * encoder_pop_dim
        output_pop = 2 * act_dim * decoder_pop_dim
        self.encoder_gain = nn.Parameter(torch.ones(obs_dim))
        self.encoder_bias = nn.Parameter(torch.zeros(obs_dim))
        thresholds = (torch.arange(encoder_pop_dim, dtype=torch.float32) + 0.5) / encoder_pop_dim
        self.register_buffer("encoder_thresholds", thresholds.reshape(1, 1, -1))

        self.Linear1 = nn.Linear(input_pop, hidden_sizes[0])
        self.Linear2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.Linear3 = nn.Linear(hidden_sizes[1], output_pop)
        for layer in (self.Linear1, self.Linear2, self.Linear3):
            nn.init.kaiming_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

        # A continuous membrane readout prevents a silent final spike layer
        # from forcing all deterministic motor commands to exactly zero.
        self.readout_norm = nn.LayerNorm(hidden_sizes[1])
        self.motor_readout = nn.Linear(hidden_sizes[1], act_dim)
        nn.init.orthogonal_(self.motor_readout.weight, gain=0.40)
        nn.init.zeros_(self.motor_readout.bias)
        self.spike_gain = nn.Parameter(torch.ones(act_dim))
        self.register_buffer("last_spike_rates", torch.zeros(3), persistent=False)

    def _encode(self, obs: torch.Tensor) -> torch.Tensor:
        x = torch.tanh(obs * self.encoder_gain + self.encoder_bias)
        probability = torch.cat((x.clamp_min(0), (-x).clamp_min(0)), dim=-1)
        return surrogate_spike(probability.unsqueeze(-1) - self.encoder_thresholds).flatten(1)

    @staticmethod
    def _lif(current: torch.Tensor, membrane: torch.Tensor,
             previous_spike: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        membrane = 0.5 * membrane + current - previous_spike
        return surrogate_spike(membrane - 1.0), membrane

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.shape[-1] != self.obs_dim:
            raise ValueError(f"Expected {self.obs_dim} Actor observations, got {obs.shape[-1]}")
        encoded = self._encode(obs)
        current1 = self.Linear1(encoded)
        batch = obs.shape[0]
        mem1 = obs.new_zeros(batch, self.hidden_sizes[0])
        mem2 = obs.new_zeros(batch, self.hidden_sizes[1])
        mem3 = obs.new_zeros(batch, 2 * self.act_dim * self.decoder_pop_dim)
        spk1 = torch.zeros_like(mem1)
        spk2 = torch.zeros_like(mem2)
        spk3 = torch.zeros_like(mem3)
        spike_counts = torch.zeros_like(mem3)
        rate_counts = obs.new_zeros(3)

        for _ in range(self.snn_steps):
            spk1, mem1 = self._lif(current1, mem1, spk1)
            spk2, mem2 = self._lif(self.Linear2(spk1), mem2, spk2)
            spk3, mem3 = self._lif(self.Linear3(spk2), mem3, spk3)
            spike_counts = spike_counts + spk3
            rate_counts = rate_counts + torch.stack((spk1.mean(), spk2.mean(), spk3.mean()))

        population_rate = (spike_counts / self.snn_steps).reshape(batch, 2 * self.act_dim,
                                                                   self.decoder_pop_dim).mean(-1)
        signed_rate = population_rate[:, :self.act_dim] - population_rate[:, self.act_dim:]
        membrane_action = 2.0 * torch.tanh(self.motor_readout(self.readout_norm(mem2)))
        mean = membrane_action + 0.25 * self.spike_gain * signed_rate
        self.last_spike_rates = (rate_counts / self.snn_steps).detach()
        return mean


def build_actor_critic_class(official_class):
    """Replace only the Actor module; retain APEX's critics and distribution API."""

    class SNNMultiCriticActorCritic(official_class):
        is_recurrent = False

        def __init__(self, num_actor_obs, num_critic_obs, num_actions,
                     snn_hidden_sizes=(256, 256), encoder_pop_dim=64,
                     decoder_pop_dim=256, snn_steps=4, **kwargs):
            super().__init__(num_actor_obs, num_critic_obs, num_actions, **kwargs)
            self.actor = SpikeActor(num_actor_obs, num_actions,
                                    hidden_sizes=tuple(snn_hidden_sizes),
                                    encoder_pop_dim=encoder_pop_dim,
                                    decoder_pop_dim=decoder_pop_dim,
                                    snn_steps=snn_steps)

    SNNMultiCriticActorCritic.__name__ = "SNNMultiCriticActorCritic"
    return SNNMultiCriticActorCritic
