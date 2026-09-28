"""Minimal vectorized PPO using the same actor and observations as BC."""

import argparse
import math
import time

import torch
from torch import nn
from torch.distributions import Normal

from benchmark.common.envs import (
    OBS_MODES,
    TASKS,
    action_spec,
    env_config,
    make_env,
    resolved_config,
)
from benchmark.common.policy import Policy, observation_spec, save_policy
from benchmark.common.utils import log_row, run_directory, seed_everything, write_json


def compute_gae(rewards, values, next_values, terminated, truncated, gamma, gae_lambda):
    """Bootstrap timeouts, never true terminals; stop the trace at either boundary."""
    advantages = torch.zeros_like(rewards)
    trace = torch.zeros_like(rewards[0])
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * (~terminated[t]).float() * next_values[t] - values[t]
        trace = delta + gamma * gae_lambda * (~(terminated[t] | truncated[t])).float() * trace
        advantages[t] = trace
    return advantages, advantages + values


class ActorCritic(nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy
        self.critic = nn.Sequential(nn.Linear(policy.encoder.output_dim, 256), nn.Tanh(), nn.Linear(256, 1))

    def value(self, obs):
        return self.critic(self.policy.encoder(obs)).squeeze(-1)

    def action_value(self, obs, action=None):
        features = self.policy.encoder(obs)
        dist = Normal(self.policy.mean(features), self.policy.log_std.exp())
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action).sum(-1), dist.entropy().sum(-1), self.critic(features).squeeze(-1)


def train(args):
    if min(args.total_timesteps, args.num_steps, args.update_epochs, args.batch_size, args.save_every) < 1:
        raise ValueError("Training counts must be positive")
    seed_everything(args.seed)
    num_envs = args.num_envs if args.num_envs is not None else (256 if args.obs_mode == "state" else 32)
    config = env_config(args.env_id, args.obs_mode, args.control_mode)
    env = make_env(config, num_envs)
    try:
        raw_obs, _ = env.reset(seed=args.seed)
        policy = Policy(observation_spec(raw_obs, args.obs_mode), action_spec(env), resolved_config(config, env)).to(args.device)
        agent = ActorCritic(policy).to(args.device)
        optimizer = torch.optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
        output = run_directory(args.output, "rl", args.env_id, args.obs_mode, args.seed)
        settings = {**vars(args), "num_envs": num_envs, "algorithm": "ppo", "env_config": policy.env_config}
        write_json(output / "config.json", settings)
        obs = policy.prepare(raw_obs)
        step_count, iteration, start = 0, 0, time.monotonic()
        while step_count < args.total_timesteps:
            iteration += 1
            steps = min(args.num_steps, math.ceil((args.total_timesteps - step_count) / num_envs))
            observations = {k: torch.empty((steps, *v.shape), dtype=v.dtype, device=args.device) for k, v in obs.items()}
            shape = (steps, num_envs)
            actions = torch.empty((*shape, policy.mean.out_features), device=args.device)
            log_probs, rewards, values, next_values = [torch.empty(shape, device=args.device) for _ in range(4)]
            terminated, truncated = [torch.zeros(shape, dtype=torch.bool, device=args.device) for _ in range(2)]
            episode_metrics = []
            with torch.no_grad():
                for t in range(steps):
                    for key in obs:
                        observations[key][t] = obs[key]
                    action, log_prob, _, value = agent.action_value(obs)
                    # Store the sampled action/log-probability; only the executed action is clipped.
                    actions[t], log_probs[t], values[t] = action, log_prob, value
                    raw_obs, reward, term, trunc, info = env.step(policy.clip(action).to(env.device))
                    rewards[t] = reward.to(args.device)
                    terminated[t], truncated[t] = term.to(args.device), trunc.to(args.device)
                    obs = policy.prepare(raw_obs)
                    next_values[t] = agent.value(obs)
                    done = terminated[t] | truncated[t]
                    if done.any():
                        final_values = agent.value(policy.prepare(info["final_observation"]))
                        next_values[t] = torch.where(done, final_values, next_values[t])
                        metrics = info["final_info"]["episode"]
                        episode_metrics.append({k: metrics[k][info["_final_info"]].float()
                                                for k in ("return", "success_once")})
                    step_count += num_envs
                advantages, returns = compute_gae(rewards, values, next_values, terminated, truncated, args.gamma, args.gae_lambda)

            flat_obs = {k: v.flatten(0, 1) for k, v in observations.items()}
            flat_actions = actions.flatten(0, 1)
            old_log_probs, advantages, returns = log_probs.flatten(), advantages.flatten(), returns.flatten()
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
            size = steps * num_envs
            for _ in range(args.update_epochs):
                indices = torch.randperm(size, device=args.device)
                for batch in indices.split(args.batch_size):
                    _, log_prob, entropy, value = agent.action_value({k: v[batch] for k, v in flat_obs.items()}, flat_actions[batch])
                    log_ratio = log_prob - old_log_probs[batch]
                    ratio = log_ratio.exp()
                    surrogate = advantages[batch] * ratio
                    clipped = advantages[batch] * ratio.clamp(1 - args.clip_coef, 1 + args.clip_coef)
                    policy_loss = -torch.minimum(surrogate, clipped).mean()
                    value_loss = 0.5 * (value - returns[batch]).square().mean()
                    loss = policy_loss + args.value_coef * value_loss - args.entropy_coef * entropy.mean()
                    if not torch.isfinite(loss):
                        raise RuntimeError("Non-finite PPO loss")
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()
                    approx_kl = ((ratio - 1) - log_ratio).mean().item()
                    if approx_kl > args.target_kl:
                        break
                if approx_kl > args.target_kl:
                    break
            metrics = {k: torch.cat([m[k] for m in episode_metrics]).mean().item()
                       if episode_metrics else None for k in ("return", "success_once")}
            log_row(output / "train.csv", {
                "iteration": iteration, "steps": step_count,
                "seconds": time.monotonic() - start, "policy_loss": policy_loss.item(),
                "value_loss": value_loss.item(), "entropy": entropy.mean().item(),
                "approx_kl": approx_kl, **metrics,
            })
            if iteration % args.save_every == 0 or step_count >= args.total_timesteps:
                save_policy(output / "policy.pt", policy, {**settings, "steps": step_count})
        print(f"Policy saved: {output / 'policy.pt'}")
        return output / "policy.pt"
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-id", choices=TASKS, default="PickCube-v1")
    parser.add_argument("--obs-mode", choices=OBS_MODES, default="state")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--total-timesteps", type=int, default=10_000_000)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=0.03)
    parser.add_argument("--control-mode", default="pd_joint_delta_pos")
    parser.add_argument("--save-every", type=int, default=10, help="Checkpoint interval in PPO iterations")
    parser.add_argument("--output", help="New run directory (must not already exist)")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
