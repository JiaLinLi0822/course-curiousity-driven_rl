"""Training driver.

Usage:
    python -m experiments.train_white --condition info_gain --seed 0 --steps 300000
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from chromatic_white.env import ChromaticWhiteEnv
from chromatic_white.env import NUM_ACTIONS
from chromatic_white.agent import ActorCriticGRU, ActorCriticConfig
from chromatic_white.world_model import WorldModel
from chromatic_white.intrinsic import (
    InfoGainIntrinsic, SurprisalIntrinsic, NoveltyIntrinsic, HybridIntrinsic,
    ExpectedInfoGainIntrinsic,
)
from chromatic_white.trainer import PPOTrainer, PPOConfig
from chromatic_white.evaluator import evaluate, evaluate_random
from chromatic_white.rules import edge_key, ALL_EDGES, EDGE_LABELS


CONDITIONS = ("info_gain", "surprisal", "novelty", "random", "hybrid",
              "expected_info_gain")


def make_intrinsic(name: str, world_model: WorldModel, n_states_estimate: int):
    if name == "info_gain":
        return InfoGainIntrinsic(world_model, scale=1.0)
    if name == "surprisal":
        return SurprisalIntrinsic(world_model, scale=1.0)
    if name == "novelty":
        return NoveltyIntrinsic(scale=1.0, n_states_estimate=n_states_estimate)
    if name == "hybrid":
        return HybridIntrinsic(world_model, alpha=0.5, scale=1.0,
                               n_states_estimate=n_states_estimate)
    if name == "expected_info_gain":
        return ExpectedInfoGainIntrinsic(world_model, scale=1.0)
    raise ValueError(name)


def _run_name(condition: str, seed: int, beta: float) -> str:
    run_name = f"{condition}_seed{seed}"
    if beta != 1.0:
        run_name = f"{condition}_beta{beta}_seed{seed}"
    return run_name


def _save_run(output_dir, run_name, history, eval_history, final_eval, agent=None, episode_history=None):
    out_dir = Path(output_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    with open(out_dir / "eval_history.json", "w") as f:
        json.dump(eval_history, f, indent=2)
    with open(out_dir / "final_eval.json", "w") as f:
        json.dump(final_eval, f, indent=2)
    if episode_history is not None:
        with open(out_dir / "episode_history.json", "w") as f:
            json.dump(episode_history, f, indent=2)
    if agent is not None:
        torch.save(agent.state_dict(), out_dir / "agent.pt")
    else:
        with open(out_dir / "random_policy.json", "w") as f:
            json.dump({"policy": "uniform_random", "num_actions": NUM_ACTIONS}, f, indent=2)


def run_random_experiment(
    seed: int,
    total_env_steps: int,
    max_episode_steps: int = 1500,
    output_dir: str = "./runs_white",
    eval_every: int = 10000,
    n_eval_episodes: int = 30,
    beta: float = 1.0,
    n_base_color_tiles_frac: float = 0.25,
) -> Dict:
    rng = np.random.default_rng(seed)
    env = ChromaticWhiteEnv(
        max_steps=max_episode_steps,
        seed=seed,
        n_base_color_tiles_frac=n_base_color_tiles_frac,
    )
    env.reset()
    world_model = WorldModel(prior_alpha=1.0)

    history = []
    episode_history = []
    eval_history = []
    recent_returns, recent_lengths, recent_solves, recent_overloads = [], [], [], []
    ep_return, ep_len, ep_overloads = 0.0, 0, 0
    episode_count = 0
    global_step = 0
    next_eval_step = eval_every
    interval_edge_visits = {e: 0 for e in ALL_EDGES}
    t_start = time.time()

    def append_recent(lst, value):
        lst.append(value)
        if len(lst) > 100:
            lst.pop(0)

    while global_step < total_env_steps:
        action = int(rng.integers(NUM_ACTIONS))
        _, ext_r, done, info = env.step(action)
        global_step += 1
        ep_return += ext_r
        ep_len += 1
        if info.get("was_overload"):
            ep_overloads += 1
        if not info["was_oob"]:
            world_model.update(*info["transition"])
            c_A, c_B, _ = info["transition"]
            interval_edge_visits[edge_key(c_A, c_B)] += 1

        if done:
            episode_count += 1
            episode_history.append({
                "episode": episode_count,
                "global_step": global_step,
                "length": ep_len,
                "return_ext": float(ep_return),
                "solved": bool(info["solved"]),
                "overloads": int(ep_overloads),
            })
            append_recent(recent_returns, ep_return)
            append_recent(recent_lengths, ep_len)
            append_recent(recent_solves, 1 if info["solved"] else 0)
            append_recent(recent_overloads, ep_overloads)
            ep_return, ep_len, ep_overloads = 0.0, 0, 0
            env.reset()

        if global_step >= next_eval_step or global_step >= total_env_steps:
            metrics = evaluate_random(
                env_kwargs={"max_steps": max_episode_steps,
                            "n_base_color_tiles_frac": n_base_color_tiles_frac},
                n_episodes=n_eval_episodes,
                base_seed=100_000 + seed * 10_000 + next_eval_step,
            )
            metrics["global_step"] = global_step
            eval_history.append(metrics)
            edge_visit_log = {f"visit_{EDGE_LABELS[e]}": c
                              for e, c in interval_edge_visits.items()}
            log = {
                "global_step": global_step,
                "episodes": episode_count,
                "mean_return_ext": float(np.mean(recent_returns or [0.0])),
                "mean_ep_len": float(np.mean(recent_lengths or [0.0])),
                "success_rate": float(np.mean(recent_solves or [0.0])),
                "mean_overloads_per_ep": float(np.mean(recent_overloads or [0.0])),
                "mean_int_reward_per_step": 0.0,
                "elapsed_s": time.time() - t_start,
                "entropy": float(np.log(NUM_ACTIONS)),
                **world_model.snapshot(),
                **world_model.edge_snapshot(),
                **edge_visit_log,
            }
            history.append(log)
            print(
                f"[random seed={seed} max_ep={max_episode_steps}] "
                f"step={global_step}/{total_env_steps} "
                f"eval_success={metrics['eval_success_rate']:.3f} "
                f"wm_kl={log['wm_kl_to_truth']:.4g} "
                f"elapsed={log['elapsed_s']:.1f}s",
                flush=True,
            )
            interval_edge_visits = {e: 0 for e in ALL_EDGES}
            while next_eval_step <= global_step:
                next_eval_step += eval_every

    final_eval = evaluate_random(
        env_kwargs={"max_steps": max_episode_steps,
                    "n_base_color_tiles_frac": n_base_color_tiles_frac},
        n_episodes=n_eval_episodes * 2,
        base_seed=200_000 + seed * 10_000,
    )
    _save_run(output_dir, _run_name("random", seed, beta), history, eval_history, final_eval,
              episode_history=episode_history)
    return {"history": history, "eval_history": eval_history, "final_eval": final_eval}


def run_experiment(
    condition: str,
    seed: int,
    total_env_steps: int,
    max_episode_steps: int = 1500,
    output_dir: str = "./runs_white",
    device: str = "cpu",
    eval_every: int = 10000,
    n_eval_episodes: int = 30,
    beta: float = 1.0,
    entropy_coef: float = 0.05,
    entropy_coef_final: float | None = None,
    n_base_color_tiles_frac: float = 0.25,
) -> Dict:
    assert condition in CONDITIONS

    torch.manual_seed(seed)
    np.random.seed(seed)

    if condition == "random":
        return run_random_experiment(
            seed=seed,
            total_env_steps=total_env_steps,
            max_episode_steps=max_episode_steps,
            output_dir=output_dir,
            eval_every=eval_every,
            n_eval_episodes=n_eval_episodes,
            beta=beta,
            n_base_color_tiles_frac=n_base_color_tiles_frac,
        )

    env = ChromaticWhiteEnv(
        max_steps=max_episode_steps,
        seed=seed,
        n_base_color_tiles_frac=n_base_color_tiles_frac,
    )
    env.reset()

    agent_cfg = ActorCriticConfig(
        obs_dim=env.OBS_DIM, num_actions=4, hidden_dim=256, mlp_hidden=256,
    )
    agent = ActorCriticGRU(agent_cfg)

    world_model = WorldModel(prior_alpha=1.0)
    n_states_estimate = 36 * 8 * 8
    intrinsic_mod = make_intrinsic(condition, world_model, n_states_estimate=n_states_estimate)

    ppo_cfg = PPOConfig(
        rollout_length=256,
        n_epochs=4,
        minibatch_size=64,
        learning_rate=3e-4,
        entropy_coef=entropy_coef,
        entropy_coef_final=entropy_coef_final,
        clip_range=0.2,
        gamma_ext=0.99,
        gamma_int=0.95,
        gae_lambda=0.95,
        beta=beta,
        total_env_steps=total_env_steps,
        log_interval=1,
        seed=seed,
        device=device,
    )

    eval_history = []
    next_eval_step = eval_every

    def on_log(log_entry):
        nonlocal next_eval_step
        if log_entry["global_step"] >= next_eval_step:
            metrics = evaluate(
                agent,
                env_kwargs={"max_steps": max_episode_steps,
                            "n_base_color_tiles_frac": n_base_color_tiles_frac},
                n_episodes=n_eval_episodes,
                deterministic=False,
                device=device,
                base_seed=100_000 + seed * 10_000,
            )
            metrics["global_step"] = log_entry["global_step"]
            eval_history.append(metrics)
            print(
                f"[{condition} seed={seed} max_ep={max_episode_steps}] "
                f"step={log_entry['global_step']}/{total_env_steps} "
                f"eval_success={metrics['eval_success_rate']:.3f} "
                f"train_success={log_entry['success_rate']:.3f} "
                f"wm_kl={log_entry['wm_kl_to_truth']:.4g} "
                f"entropy={log_entry['entropy']:.3f} "
                f"elapsed={log_entry['elapsed_s']:.1f}s",
                flush=True,
            )
            next_eval_step += eval_every

    trainer = PPOTrainer(env, agent, intrinsic_mod, world_model, ppo_cfg, eval_callback=on_log)
    history = trainer.train()

    final_eval = evaluate(
        agent,
        env_kwargs={"max_steps": max_episode_steps,
                    "n_base_color_tiles_frac": n_base_color_tiles_frac},
        n_episodes=n_eval_episodes * 2,
        deterministic=False,
        device=device,
        base_seed=200_000 + seed * 10_000,
    )

    _save_run(output_dir, _run_name(condition, seed, beta), history, eval_history, final_eval,
              agent=agent, episode_history=trainer.episode_log)

    return {"history": history, "eval_history": eval_history, "final_eval": final_eval}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=300_000)
    parser.add_argument("--max_ep_steps", type=int, default=1500)
    parser.add_argument("--output_dir", type=str, default="./runs_white")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--eval_every", type=int, default=10000)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--entropy_coef", type=float, default=0.05)
    parser.add_argument("--entropy_coef_final", type=float, default=None)
    parser.add_argument("--n_base_color_tiles_frac", type=float, default=0.25)
    args = parser.parse_args()

    out = run_experiment(
        condition=args.condition, seed=args.seed,
        total_env_steps=args.steps, max_episode_steps=args.max_ep_steps,
        output_dir=args.output_dir, device=args.device,
        eval_every=args.eval_every, beta=args.beta,
        entropy_coef=args.entropy_coef,
        entropy_coef_final=args.entropy_coef_final,
        n_base_color_tiles_frac=args.n_base_color_tiles_frac,
    )
    print(f"\nFinal eval ({args.condition} seed={args.seed}):")
    for k, v in out["final_eval"].items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
