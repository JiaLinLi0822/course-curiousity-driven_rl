"""Held-out evaluator."""

import numpy as np
import torch

from chromatic_white.env import ChromaticWhiteEnv
from chromatic_white.env import NUM_ACTIONS


@torch.no_grad()
def evaluate(agent, env_kwargs, n_episodes=50, deterministic=False,
             device="cpu", base_seed=100_000):
    dev = torch.device(device)
    agent.eval()
    solves = 0
    steps_to_solve = []
    episode_lengths = []
    overloads = []

    for ep in range(n_episodes):
        env = ChromaticWhiteEnv(**env_kwargs, seed=base_seed + ep)
        obs = env.reset()
        h = agent.initial_hidden(1, dev)
        ep_overloads = 0
        solved_this_ep = False
        while not env.done:
            obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(dev)
            action, _, _, _, _, h_next = agent.get_action(obs_t, h, deterministic=deterministic)
            obs, _, _, info = env.step(int(action.item()))
            if info.get("was_overload"):
                ep_overloads += 1
            if info.get("solved"):
                solved_this_ep = True
            h = h_next
        episode_lengths.append(env.steps)
        overloads.append(ep_overloads)
        if solved_this_ep:
            solves += 1
            steps_to_solve.append(env.steps)

    agent.train()
    return {
        "eval_success_rate": solves / n_episodes,
        "eval_mean_steps_to_solve": float(np.mean(steps_to_solve)) if steps_to_solve else float("nan"),
        "eval_mean_ep_len": float(np.mean(episode_lengths)),
        "eval_mean_overloads_per_ep": float(np.mean(overloads)),
        "eval_n_episodes": n_episodes,
    }


def evaluate_random(env_kwargs, n_episodes=50, base_seed=100_000):
    solves = 0
    steps_to_solve = []
    episode_lengths = []
    overloads = []

    for ep in range(n_episodes):
        rng = np.random.default_rng(base_seed + ep)
        env = ChromaticWhiteEnv(**env_kwargs, seed=base_seed + ep)
        env.reset()
        ep_overloads = 0
        solved_this_ep = False
        while not env.done:
            _, _, _, info = env.step(int(rng.integers(NUM_ACTIONS)))
            if info.get("was_overload"):
                ep_overloads += 1
            if info.get("solved"):
                solved_this_ep = True
        episode_lengths.append(env.steps)
        overloads.append(ep_overloads)
        if solved_this_ep:
            solves += 1
            steps_to_solve.append(env.steps)

    return {
        "eval_success_rate": solves / n_episodes,
        "eval_mean_steps_to_solve": float(np.mean(steps_to_solve)) if steps_to_solve else float("nan"),
        "eval_mean_ep_len": float(np.mean(episode_lengths)),
        "eval_mean_overloads_per_ep": float(np.mean(overloads)),
        "eval_n_episodes": n_episodes,
    }
