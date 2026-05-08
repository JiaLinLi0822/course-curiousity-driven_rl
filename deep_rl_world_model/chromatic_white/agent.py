"""Actor-critic with GRU."""

from dataclasses import dataclass
import torch
import torch.nn as nn


@dataclass
class ActorCriticConfig:
    obs_dim: int = 333
    num_actions: int = 4
    hidden_dim: int = 256
    mlp_hidden: int = 256


class ActorCriticGRU(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.input_fc = nn.Sequential(nn.Linear(cfg.obs_dim, cfg.mlp_hidden), nn.Tanh())
        self.gru = nn.GRUCell(cfg.mlp_hidden, cfg.hidden_dim)
        self.policy_head = nn.Linear(cfg.hidden_dim, cfg.num_actions)
        self.value_ext_head = nn.Linear(cfg.hidden_dim, 1)
        self.value_int_head = nn.Linear(cfg.hidden_dim, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.constant_(m.bias, 0.0)
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        for name, p in self.gru.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(p)
            elif "bias" in name:
                nn.init.constant_(p, 0.0)

    def forward(self, obs, h):
        x = self.input_fc(obs)
        h_next = self.gru(x, h)
        logits = self.policy_head(h_next)
        v_ext = self.value_ext_head(h_next).squeeze(-1)
        v_int = self.value_int_head(h_next).squeeze(-1)
        return logits, v_ext, v_int, h_next

    @torch.no_grad()
    def get_action(self, obs, h, deterministic=False):
        logits, v_ext, v_int, h_next = self.forward(obs, h)
        dist = torch.distributions.Categorical(logits=logits)
        action = torch.argmax(logits, dim=-1) if deterministic else dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, v_ext, v_int, h_next

    def evaluate_actions(self, obs_seq, h_init, actions, dones):
        T, B, _ = obs_seq.shape
        h = h_init
        log_probs, entropies, v_exts, v_ints = [], [], [], []
        for t in range(T):
            logits, v_ext, v_int, h_next = self.forward(obs_seq[t], h)
            dist = torch.distributions.Categorical(logits=logits)
            log_probs.append(dist.log_prob(actions[t]))
            entropies.append(dist.entropy())
            v_exts.append(v_ext)
            v_ints.append(v_int)
            mask = (~dones[t]).float().unsqueeze(-1)
            h = h_next * mask
        return (torch.stack(log_probs), torch.stack(entropies),
                torch.stack(v_exts), torch.stack(v_ints))

    def initial_hidden(self, batch_size, device):
        return torch.zeros(batch_size, self.cfg.hidden_dim, device=device)
