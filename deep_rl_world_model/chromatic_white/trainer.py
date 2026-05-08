"""PPO trainer."""

import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from chromatic_white.env import ChromaticWhiteEnv
from chromatic_white.agent import ActorCriticGRU
from chromatic_white.world_model import WorldModel
from chromatic_white.intrinsic import IntrinsicRewardModule
from chromatic_white.rules import edge_key, ALL_EDGES, EDGE_LABELS


@dataclass
class PPOConfig:
    rollout_length: int = 256
    n_envs: int = 1
    learning_rate: float = 3e-4
    n_epochs: int = 4
    minibatch_size: int = 64
    clip_range: float = 0.2
    value_clip_range: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.05
    entropy_coef_final: float | None = None
    max_grad_norm: float = 0.5
    gamma_ext: float = 0.99
    gamma_int: float = 0.95
    gae_lambda: float = 0.95
    beta: float = 1.0
    total_env_steps: int = 300_000
    log_interval: int = 1
    seed: int = 0
    device: str = "cpu"


class RolloutBuffer:
    def __init__(self, T, B, obs_dim, hidden_dim, device):
        self.T, self.B = T, B
        self.device = device
        self.obs = torch.zeros(T, B, obs_dim, device=device)
        self.actions = torch.zeros(T, B, dtype=torch.long, device=device)
        self.log_probs = torch.zeros(T, B, device=device)
        self.rewards_ext = torch.zeros(T, B, device=device)
        self.rewards_int = torch.zeros(T, B, device=device)
        self.dones = torch.zeros(T, B, dtype=torch.bool, device=device)
        self.values_ext = torch.zeros(T + 1, B, device=device)
        self.values_int = torch.zeros(T + 1, B, device=device)
        self.h_init = torch.zeros(B, hidden_dim, device=device)
        self.rollout_edge_visits = {}

    def compute_gae(self, gamma, lam, reward_stream):
        if reward_stream == "ext":
            rewards, values = self.rewards_ext, self.values_ext
        else:
            rewards, values = self.rewards_int, self.values_int
        advantages = torch.zeros_like(rewards)
        last_gae = torch.zeros(self.B, device=self.device)
        for t in reversed(range(self.T)):
            not_done = (~self.dones[t]).float()
            delta = rewards[t] + gamma * values[t + 1] * not_done - values[t]
            last_gae = delta + gamma * lam * not_done * last_gae
            advantages[t] = last_gae
        return advantages


class PPOTrainer:
    def __init__(self, env, agent, intrinsic_module, world_model, cfg, eval_callback=None):
        self.env = env
        self.agent = agent
        self.intrinsic_module = intrinsic_module
        self.world_model = world_model if world_model is not None else WorldModel()
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.agent.to(self.device)
        self.optimizer = torch.optim.Adam(agent.parameters(), lr=cfg.learning_rate)
        self.eval_callback = eval_callback
        self.global_step = 0
        self.episode_count = 0
        self.history = []
        self.episode_log = []
        self._current_obs = None
        self._current_h = self.agent.initial_hidden(1, self.device)
        self._ep_return_ext = 0.0
        self._ep_len = 0
        self._ep_overloads = 0
        self._recent_returns = []
        self._recent_lengths = []
        self._recent_solves = []
        self._recent_overloads = []

    def collect_rollout(self):
        cfg = self.cfg
        buf = RolloutBuffer(cfg.rollout_length, cfg.n_envs,
                            self.env.OBS_DIM, self.agent.cfg.hidden_dim, self.device)
        if self._current_obs is None:
            self._current_obs = self.env.reset()
            self._current_h = self.agent.initial_hidden(1, self.device)
        buf.h_init = self._current_h.clone()
        rollout_edge_visits = {e: 0 for e in ALL_EDGES}

        for t in range(cfg.rollout_length):
            obs_t = torch.from_numpy(self._current_obs).float().unsqueeze(0).to(self.device)
            action, log_prob, _, v_ext, v_int, h_next = self.agent.get_action(obs_t, self._current_h)
            a = int(action.item())
            next_obs, ext_r, done, info = self.env.step(a)
            int_r = self.intrinsic_module.compute(
                self._current_obs, next_obs, info["transition"], info["was_oob"])
            if not info["was_oob"]:
                self.world_model.update(*info["transition"])
                c_A, c_B, _ = info["transition"]
                rollout_edge_visits[edge_key(c_A, c_B)] += 1

            buf.obs[t, 0] = obs_t.squeeze(0)
            buf.actions[t, 0] = action.squeeze(0)
            buf.log_probs[t, 0] = log_prob.squeeze(0)
            buf.rewards_ext[t, 0] = float(ext_r)
            buf.rewards_int[t, 0] = float(int_r)
            buf.dones[t, 0] = bool(done)
            buf.values_ext[t, 0] = v_ext.squeeze(0)
            buf.values_int[t, 0] = v_int.squeeze(0)

            self._ep_return_ext += ext_r
            self._ep_len += 1
            self.global_step += 1
            if info.get("was_overload"):
                self._ep_overloads += 1

            if done:
                self.episode_count += 1
                self.episode_log.append({
                    "episode": self.episode_count,
                    "global_step": self.global_step,
                    "length": self._ep_len,
                    "return_ext": float(self._ep_return_ext),
                    "solved": bool(info["solved"]),
                    "overloads": int(self._ep_overloads),
                })
                self._recent_returns.append(self._ep_return_ext)
                self._recent_lengths.append(self._ep_len)
                self._recent_solves.append(1 if info["solved"] else 0)
                self._recent_overloads.append(self._ep_overloads)
                for lst in [self._recent_returns, self._recent_lengths,
                            self._recent_solves, self._recent_overloads]:
                    if len(lst) > 100:
                        lst.pop(0)
                self._ep_return_ext = 0.0
                self._ep_len = 0
                self._ep_overloads = 0
                self._current_obs = self.env.reset()
                self._current_h = self.agent.initial_hidden(1, self.device)
            else:
                self._current_obs = next_obs
                self._current_h = h_next

        with torch.no_grad():
            obs_t = torch.from_numpy(self._current_obs).float().unsqueeze(0).to(self.device)
            _, v_ext_boot, v_int_boot, _ = self.agent.forward(obs_t, self._current_h)
            buf.values_ext[-1, 0] = v_ext_boot.squeeze(0)
            buf.values_int[-1, 0] = v_int_boot.squeeze(0)

        buf.rollout_edge_visits = rollout_edge_visits
        return buf

    def ppo_update(self, buf):
        cfg = self.cfg
        adv_ext = buf.compute_gae(cfg.gamma_ext, cfg.gae_lambda, "ext")
        adv_int = buf.compute_gae(cfg.gamma_int, cfg.gae_lambda, "int")
        ret_ext = adv_ext + buf.values_ext[:-1]
        ret_int = adv_int + buf.values_int[:-1]

        def normalize(x):
            return (x - x.mean()) / (x.std() + 1e-8)
        adv = normalize(adv_ext) + cfg.beta * normalize(adv_int)

        T, B = cfg.rollout_length, cfg.n_envs
        b_actions = buf.actions.reshape(T * B)
        b_old_log_probs = buf.log_probs.reshape(T * B)
        b_advantages = adv.reshape(T * B)
        b_ret_ext = ret_ext.reshape(T * B)
        b_ret_int = ret_int.reshape(T * B)
        b_old_v_ext = buf.values_ext[:-1].reshape(T * B)
        b_old_v_int = buf.values_int[:-1].reshape(T * B)

        stats = {"pg_loss": 0.0, "v_ext_loss": 0.0, "v_int_loss": 0.0,
                 "entropy": 0.0, "approx_kl": 0.0, "clip_frac": 0.0, "n_updates": 0.0}

        for epoch in range(cfg.n_epochs):
            new_log_probs, entropy, v_ext, v_int = self.agent.evaluate_actions(
                buf.obs, buf.h_init, buf.actions, buf.dones)
            new_log_probs = new_log_probs.reshape(T * B)
            entropy = entropy.reshape(T * B)
            v_ext = v_ext.reshape(T * B)
            v_int = v_int.reshape(T * B)
            ratio = (new_log_probs - b_old_log_probs).exp()
            unclipped = ratio * b_advantages
            clipped = torch.clamp(ratio, 1 - cfg.clip_range, 1 + cfg.clip_range) * b_advantages
            pg_loss = -torch.min(unclipped, clipped).mean()

            def clipped_value_loss(v_new, v_old, ret):
                v_clipped = v_old + (v_new - v_old).clamp(-cfg.value_clip_range, cfg.value_clip_range)
                loss_unclipped = (v_new - ret).pow(2)
                loss_clipped = (v_clipped - ret).pow(2)
                return 0.5 * torch.max(loss_unclipped, loss_clipped).mean()

            v_ext_loss = clipped_value_loss(v_ext, b_old_v_ext, b_ret_ext)
            v_int_loss = clipped_value_loss(v_int, b_old_v_int, b_ret_int)
            value_loss = v_ext_loss + v_int_loss
            ent_loss = -entropy.mean()
            if cfg.entropy_coef_final is not None and cfg.total_env_steps > 0:
                progress = min(self.global_step / cfg.total_env_steps, 1.0)
                current_entropy_coef = (cfg.entropy_coef
                                         + (cfg.entropy_coef_final - cfg.entropy_coef) * progress)
            else:
                current_entropy_coef = cfg.entropy_coef
            loss = pg_loss + cfg.value_loss_coef * value_loss + current_entropy_coef * ent_loss

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.agent.parameters(), cfg.max_grad_norm)
            self.optimizer.step()

            with torch.no_grad():
                approx_kl = (b_old_log_probs - new_log_probs).mean().item()
                clip_frac = ((ratio - 1.0).abs() > cfg.clip_range).float().mean().item()

            stats["pg_loss"] += pg_loss.item()
            stats["v_ext_loss"] += v_ext_loss.item()
            stats["v_int_loss"] += v_int_loss.item()
            stats["entropy"] += entropy.mean().item()
            stats["approx_kl"] += approx_kl
            stats["clip_frac"] += clip_frac
            stats["n_updates"] += 1

        if hasattr(self.intrinsic_module, "train_step"):
            batch = {"obs": buf.obs.reshape(T * B, -1).detach()}
            extra = self.intrinsic_module.train_step(batch)
            for k, v in extra.items():
                stats[k] = v

        for k in ["pg_loss", "v_ext_loss", "v_int_loss", "entropy", "approx_kl", "clip_frac"]:
            stats[k] /= max(stats["n_updates"], 1)
        stats["current_entropy_coef"] = float(current_entropy_coef)
        return stats

    def train(self):
        cfg = self.cfg
        t_start = time.time()
        update_idx = 0
        while self.global_step < cfg.total_env_steps:
            buf = self.collect_rollout()
            stats = self.ppo_update(buf)
            update_idx += 1
            if update_idx % cfg.log_interval == 0:
                edge_visit_log = {f"visit_{EDGE_LABELS[e]}": c
                                  for e, c in buf.rollout_edge_visits.items()}
                log = {
                    "global_step": self.global_step,
                    "episodes": self.episode_count,
                    "mean_return_ext": float(np.mean(self._recent_returns or [0.0])),
                    "mean_ep_len": float(np.mean(self._recent_lengths or [0.0])),
                    "success_rate": float(np.mean(self._recent_solves or [0.0])),
                    "mean_overloads_per_ep": float(np.mean(self._recent_overloads or [0.0])),
                    "mean_int_reward_per_step": float(buf.rewards_int.mean().item()),
                    "elapsed_s": time.time() - t_start,
                    **stats,
                    **self.world_model.snapshot(),
                    **self.world_model.edge_snapshot(),
                    **edge_visit_log,
                }
                self.history.append(log)
                if self.eval_callback is not None:
                    self.eval_callback(log)
        return self.history
