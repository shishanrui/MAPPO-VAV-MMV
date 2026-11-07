from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch

from mmv_env.env import MMVEnv
from mmv_env.multiagent_env import MultiAgentMMVEnv
from mmv_env.signals import N_AC_SP, N_MODES, ZONES
from scripts.run_config import SharedRunConfig, env_kwargs, resolve_model_name

from .diagnostics import CsvMetricLogger, explained_variance, print_sb3_style_log
from .networks import CentralizedCritic, MAPPOActor
from .normalization import RewardNormalizer, RunningMeanStd, load_normalizers, save_normalizers
from .rollout_buffer import MAPPOBuffer


try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover - optional dependency
    SummaryWriter = None


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _stack_actor_obs(obs_dict: dict[str, np.ndarray], zones=ZONES) -> np.ndarray:
    return np.stack([np.asarray(obs_dict[zone], dtype=np.float32) for zone in zones], axis=0)


def _actor_obs_from_normalized_state(state_norm: np.ndarray, zones=ZONES) -> np.ndarray:
    return _stack_actor_obs(MultiAgentMMVEnv.actor_obs_from_state(state_norm, zones=zones), zones=zones)


def _stack_rewards(reward_dict: dict[str, float], zones=ZONES) -> np.ndarray:
    return np.asarray([float(reward_dict[zone]) for zone in zones], dtype=np.float32)


def _stack_masks(mask_dict: dict[str, dict[str, np.ndarray]], zones=ZONES) -> tuple[np.ndarray, np.ndarray]:
    mode_masks = np.stack([np.asarray(mask_dict[zone]["mode"], dtype=bool) for zone in zones], axis=0)
    sp_masks = np.stack([np.asarray(mask_dict[zone]["sp"], dtype=bool) for zone in zones], axis=0)
    return mode_masks, sp_masks


def _actions_to_dict(actions: np.ndarray, zones=ZONES) -> dict[str, tuple[int, int]]:
    arr = np.asarray(actions, dtype=np.int64)
    return {zone: (int(arr[i, 0]), int(arr[i, 1])) for i, zone in enumerate(zones)}


def _count_invalid_actions(actions: np.ndarray, mode_masks: np.ndarray, sp_masks: np.ndarray) -> int:
    invalid = 0
    for i in range(actions.shape[0]):
        if not bool(mode_masks[i, int(actions[i, 0])]):
            invalid += 1
        if not bool(sp_masks[i, int(actions[i, 1])]):
            invalid += 1
    return int(invalid)


def _scheduled_learning_rate(cfg: SharedRunConfig, *, total_env_steps: int) -> float:
    if str(cfg.lr_schedule).strip().lower() == "constant":
        return float(cfg.learning_rate)
    if str(cfg.lr_schedule).strip().lower() != "linear":
        raise ValueError(f"Unsupported lr_schedule={cfg.lr_schedule!r}. Use 'linear' or 'constant'.")
    progress_remaining = max(
        0.0,
        1.0 - float(total_env_steps) / max(1.0, float(cfg.total_timesteps)),
    )
    return float(cfg.final_learning_rate) + (
        float(cfg.learning_rate) - float(cfg.final_learning_rate)
    ) * progress_remaining


def _set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(lr)


def _checkpoint_milestones(cfg: SharedRunConfig) -> list[int]:
    raw = getattr(cfg, "checkpoint_milestone_steps", "")
    if raw is None:
        return []
    if isinstance(raw, (list, tuple, set)):
        parts = list(raw)
    else:
        text = str(raw).strip()
        if not text:
            return []
        parts = [p.strip() for p in text.split(",")]

    milestones: set[int] = set()
    for part in parts:
        if part in ("", None):
            continue
        step = int(float(part))
        if step > 0 and step < int(cfg.total_timesteps):
            milestones.add(step)
    return sorted(milestones)


def _save_checkpoint(
    path: Path,
    *,
    actor: MAPPOActor,
    critic: CentralizedCritic,
    cfg: SharedRunConfig,
    actor_obs_dim: int,
    state_dim: int,
    model_name: str,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "actor_state_dict": actor.state_dict(),
            "critic_state_dict": critic.state_dict(),
            "config": asdict(cfg),
            "model_name": model_name,
            "zones": tuple(ZONES),
            "actor_obs_dim": int(actor_obs_dim),
            "state_dim": int(state_dim),
            "mode_dim": int(getattr(actor, "mode_dim", N_MODES)),
            "sp_dim": int(getattr(actor, "sp_dim", N_AC_SP)),
            "actor_hidden_sizes": tuple(getattr(actor, "hidden_sizes", (256, 256))),
            "actor_n_zones": int(getattr(actor, "n_zones", len(ZONES))),
            "actor_head_arch": str(getattr(actor, "head_arch", "zone_specific")),
            "critic_hidden_sizes": tuple(getattr(critic, "hidden_sizes", (256, 256))),
            "critic_n_agents": int(getattr(critic, "n_agents", len(ZONES))),
        },
        path,
    )


def _save_eval_artifacts(
    *,
    model_dir: Path,
    artifact_name: str,
    actor: MAPPOActor,
    critic: CentralizedCritic,
    cfg: SharedRunConfig,
    actor_obs_dim: int,
    state_dim: int,
    actor_obs_rms: RunningMeanStd,
    state_rms: RunningMeanStd,
    reward_normalizer: RewardNormalizer | None,
) -> tuple[Path, Path, Path]:
    ckpt_path = model_dir / f"{artifact_name}.pt"
    norm_path = model_dir / f"{artifact_name}_obsnorm.pkl"
    cfg_path = model_dir / f"{artifact_name}_config.json"
    _save_checkpoint(
        ckpt_path,
        actor=actor,
        critic=critic,
        cfg=cfg,
        actor_obs_dim=actor_obs_dim,
        state_dim=state_dim,
        model_name=artifact_name,
    )
    save_normalizers(
        norm_path,
        actor_obs_rms=actor_obs_rms,
        state_rms=state_rms,
        reward_normalizer=reward_normalizer,
    )
    with cfg_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)
    return ckpt_path, norm_path, cfg_path


def resolve_tb_run_dir(log_root: Path, run_name: str) -> Path:
    """
    Match SB3's TensorBoard directory style: <run>_1, <run>_2, ...
    """
    log_root = Path(log_root)
    log_root.mkdir(parents=True, exist_ok=True)
    idx = 1
    while True:
        candidate = log_root / f"{run_name}_{idx}"
        if not candidate.exists():
            return candidate
        idx += 1


def train_mappo(
    *,
    cfg: SharedRunConfig,
    model_dir: Path,
    model_name: Optional[str] = None,
    tensorboard_log: Optional[Path] = None,
    tensorboard_run_dir: Optional[Path] = None,
    metrics_csv: Optional[Path] = None,
    device: str = "cpu",
    env_factory: Optional[Callable[[], object]] = None,
    warm_start_checkpoint_path: Optional[Path] = None,
    warm_start_normalizer_path: Optional[Path] = None,
) -> tuple[MAPPOActor, CentralizedCritic]:
    if cfg.ctde_mode != "loose":
        raise ValueError("Initial MAPPO implementation supports only cfg.ctde_mode='loose'.")
    if cfg.reward_mode not in {"shared_team", "hybrid"}:
        raise ValueError("MAPPO supports cfg.reward_mode='shared_team' or 'hybrid'.")

    set_global_seed(cfg.seed)
    device_t = torch.device(device)
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    resolved_name = str(model_name or resolve_model_name(cfg))

    if env_factory is None:
        env = MultiAgentMMVEnv(
            MMVEnv(**env_kwargs(cfg, seed=cfg.seed)),
            zones=ZONES,
            reward_mode=cfg.reward_mode,
        )
    else:
        env = env_factory()

    writer = None
    tb_run_dir = None
    if tensorboard_log is not None and SummaryWriter is not None:
        tb_run_dir = Path(tensorboard_run_dir) if tensorboard_run_dir is not None else resolve_tb_run_dir(Path(tensorboard_log), resolved_name)
        writer = SummaryWriter(log_dir=str(tb_run_dir))
        print(f"[MAPPO] TensorBoard run dir: {tb_run_dir}")
    metric_logger = CsvMetricLogger(Path(metrics_csv)) if metrics_csv is not None else None

    try:
        obs_dict, state, _ = env.reset(seed=cfg.seed)
        actor_obs = _stack_actor_obs(obs_dict)
        actor_obs_dim = int(actor_obs.shape[-1])
        state_dim = int(np.asarray(state).shape[-1])

        actor_obs_rms = RunningMeanStd((state_dim,))
        state_rms = RunningMeanStd((state_dim,))
        reward_normalizer = RewardNormalizer(
            n_agents=len(ZONES),
            gamma=cfg.gamma,
        ) if cfg.norm_reward else None
        actor = MAPPOActor(
            actor_obs_dim,
            N_MODES,
            N_AC_SP,
            n_zones=len(ZONES),
            head_arch="zone_specific",
        ).to(device_t)
        critic = CentralizedCritic(state_dim, n_agents=len(ZONES)).to(device_t)
        if warm_start_checkpoint_path is not None:
            ckpt_path = Path(warm_start_checkpoint_path)
            try:
                ckpt = torch.load(ckpt_path, map_location=device_t, weights_only=False)
            except TypeError:
                ckpt = torch.load(ckpt_path, map_location=device_t)
            actor.load_state_dict(ckpt["actor_state_dict"])
            critic.load_state_dict(ckpt["critic_state_dict"])
            print(f"[MAPPO] warm-started actor/critic from: {ckpt_path}")
            if warm_start_normalizer_path is None:
                candidate = ckpt_path.with_name(f"{ckpt_path.stem}_obsnorm.pkl")
                if candidate.exists():
                    warm_start_normalizer_path = candidate
        if warm_start_normalizer_path is not None:
            norm_path = Path(warm_start_normalizer_path)
            actor_obs_rms_loaded, state_rms_loaded, reward_normalizer_loaded = load_normalizers(norm_path)
            if tuple(actor_obs_rms_loaded.shape) != tuple(actor_obs_rms.shape):
                raise ValueError(
                    f"Warm-start actor obs normalizer shape mismatch: "
                    f"{actor_obs_rms_loaded.shape} != {actor_obs_rms.shape}"
                )
            if tuple(state_rms_loaded.shape) != tuple(state_rms.shape):
                raise ValueError(
                    f"Warm-start state normalizer shape mismatch: "
                    f"{state_rms_loaded.shape} != {state_rms.shape}"
                )
            actor_obs_rms = actor_obs_rms_loaded
            state_rms = state_rms_loaded
            if cfg.norm_reward:
                reward_normalizer = reward_normalizer_loaded
                if reward_normalizer is None:
                    print("[MAPPO] warm-start normalizer has no reward_normalizer; initializing a new one.")
                    reward_normalizer = RewardNormalizer(n_agents=len(ZONES), gamma=cfg.gamma)
            print(f"[MAPPO] warm-started normalizers from: {norm_path}")
        actor_optimizer = torch.optim.Adam(actor.parameters(), lr=float(cfg.learning_rate), eps=1e-5)
        critic_optimizer = torch.optim.Adam(critic.parameters(), lr=float(cfg.learning_rate), eps=1e-5)

        buffer = MAPPOBuffer(
            rollout_steps=cfg.rollout_steps,
            n_agents=len(ZONES),
            obs_dim=actor_obs_dim,
            state_dim=state_dim,
            mode_dim=N_MODES,
            sp_dim=N_AC_SP,
            device=device_t,
        )

        total_env_steps = 0
        update_idx = 0
        checkpoint_milestones = _checkpoint_milestones(cfg)
        saved_checkpoint_milestones: set[int] = set()
        recent_rewards: list[float] = []
        recent_team_rewards: list[float] = []
        recent_zone_rewards: dict[str, list[float]] = {zone: [] for zone in ZONES}
        completed_episode_returns: list[float] = []
        completed_episode_team_returns: list[float] = []
        completed_episode_zone_returns: dict[str, list[float]] = {zone: [] for zone in ZONES}
        completed_episode_lengths: list[int] = []
        current_episode_return = 0.0
        current_episode_team_return = 0.0
        current_episode_zone_returns: dict[str, float] = {zone: 0.0 for zone in ZONES}
        current_episode_len = 0
        n_updates = int(math.ceil(float(cfg.total_timesteps) / float(cfg.rollout_steps)))
        start_time = time.time()

        while total_env_steps < int(cfg.total_timesteps):
            update_idx += 1
            current_lr = _scheduled_learning_rate(cfg, total_env_steps=total_env_steps)
            _set_optimizer_lr(actor_optimizer, current_lr)
            _set_optimizer_lr(critic_optimizer, current_lr)
            buffer.reset()
            invalid_actions = 0
            rollout_return = 0.0
            rollout_team_return = 0.0
            rollout_zone_returns: dict[str, float] = {zone: 0.0 for zone in ZONES}
            rollout_norm_return = 0.0
            remaining_steps = int(cfg.total_timesteps) - total_env_steps
            rollout_limit = int(cfg.rollout_steps)
            next_milestone = next(
                (
                    step
                    for step in checkpoint_milestones
                    if step not in saved_checkpoint_milestones and total_env_steps < step
                ),
                None,
            )
            if next_milestone is not None:
                rollout_limit = min(rollout_limit, int(next_milestone) - total_env_steps)
            rollout_steps = min(rollout_limit, remaining_steps)
            last_done = False

            for _ in range(rollout_steps):
                if cfg.norm_obs:
                    actor_obs_rms.update(state)
                    state_rms.update(state)
                    actor_state_norm = actor_obs_rms.normalize(state, clip=cfg.clip_obs)
                    state_norm = state_rms.normalize(state, clip=cfg.clip_obs)
                else:
                    actor_state_norm = np.asarray(state, dtype=np.float32)
                    state_norm = np.asarray(state, dtype=np.float32)
                actor_obs_norm = _actor_obs_from_normalized_state(actor_state_norm)

                mask_dict = env.action_masks()
                mode_masks, sp_masks = _stack_masks(mask_dict)

                obs_t = torch.as_tensor(actor_obs_norm, dtype=torch.float32, device=device_t)
                state_t = torch.as_tensor(state_norm[None, :], dtype=torch.float32, device=device_t)
                mode_masks_t = torch.as_tensor(mode_masks, dtype=torch.bool, device=device_t)
                sp_masks_t = torch.as_tensor(sp_masks, dtype=torch.bool, device=device_t)

                with torch.no_grad():
                    actor_out = actor.act(
                        obs_t,
                        mode_mask=mode_masks_t,
                        sp_mask=sp_masks_t,
                        deterministic=False,
                    )
                    value = critic(state_t).squeeze(0).detach().cpu().numpy().astype(np.float32)

                actions_np = actor_out.actions.cpu().numpy().astype(np.int64)
                invalid_actions += _count_invalid_actions(actions_np, mode_masks, sp_masks)
                step_out = env.step(_actions_to_dict(actions_np))
                done = bool(step_out.terminated or step_out.truncated)
                rewards = _stack_rewards(step_out.rewards)
                step_reward = float(np.mean(rewards))
                team_step_reward = float(
                    step_out.env_info.get("reward_shared_team", np.sum(rewards))
                )
                rewards_for_training = (
                    reward_normalizer.normalize(
                        rewards,
                        done=done,
                        update=True,
                        clip=cfg.clip_reward,
                    )
                    if reward_normalizer is not None
                    else rewards
                )
                norm_step_reward = float(np.mean(rewards_for_training))
                rollout_return += step_reward
                rollout_team_return += team_step_reward
                rollout_norm_return += norm_step_reward
                recent_rewards.append(step_reward)
                recent_team_rewards.append(team_step_reward)
                for zone_idx, zone in enumerate(ZONES):
                    zone_step_reward = float(rewards[zone_idx])
                    recent_zone_rewards[zone].append(zone_step_reward)
                    rollout_zone_returns[zone] += zone_step_reward
                    current_episode_zone_returns[zone] += zone_step_reward
                current_episode_return += step_reward
                current_episode_team_return += team_step_reward
                current_episode_len += 1

                buffer.add(
                    obs=actor_obs_norm,
                    state=state_norm,
                    actions=actions_np,
                    log_probs=actor_out.log_probs.cpu().numpy(),
                    rewards=rewards_for_training,
                    done=done,
                    value=value,
                    mode_masks=mode_masks,
                    sp_masks=sp_masks,
                )

                obs_dict = step_out.obs
                state = step_out.state
                total_env_steps += 1
                last_done = done

                if done:
                    completed_episode_returns.append(float(current_episode_return))
                    completed_episode_team_returns.append(float(current_episode_team_return))
                    for zone in ZONES:
                        completed_episode_zone_returns[zone].append(
                            float(current_episode_zone_returns[zone])
                        )
                    completed_episode_lengths.append(int(current_episode_len))
                    current_episode_return = 0.0
                    current_episode_team_return = 0.0
                    current_episode_zone_returns = {zone: 0.0 for zone in ZONES}
                    current_episode_len = 0
                    obs_dict, state, _ = env.reset()

            if cfg.norm_obs:
                actor_obs_rms.update(state)
                state_rms.update(state)
                state_next_norm = state_rms.normalize(state, clip=cfg.clip_obs)
            else:
                state_next_norm = np.asarray(state, dtype=np.float32)
            with torch.no_grad():
                next_value = critic(
                    torch.as_tensor(state_next_norm[None, :], dtype=torch.float32, device=device_t)
                ).squeeze(0).detach().cpu().numpy().astype(np.float32)
            buffer.compute_returns_and_advantages(
                next_value=next_value,
                next_done=bool(last_done),
                gamma=cfg.gamma,
                gae_lambda=cfg.gae_lambda,
            )

            policy_losses: list[float] = []
            value_losses: list[float] = []
            entropies: list[float] = []
            approx_kls: list[float] = []
            clip_fracs: list[float] = []
            actor_grad_norms: list[float] = []
            critic_grad_norms: list[float] = []
            early_stop_epoch = 0
            early_stop_kl = 0.0

            for _epoch in range(int(cfg.ppo_epochs)):
                epoch_kls: list[float] = []
                for batch in buffer.iter_actor_minibatches(int(cfg.minibatch_size), shuffle=True):
                    out = actor.act(
                        batch.obs,
                        mode_mask=batch.mode_masks,
                        sp_mask=batch.sp_masks,
                        actions=batch.actions,
                    )
                    new_log_probs = out.log_probs
                    log_ratio = new_log_probs - batch.old_log_probs
                    ratio = torch.exp(log_ratio)
                    pg_loss_1 = -batch.advantages * ratio
                    pg_loss_2 = -batch.advantages * torch.clamp(
                        ratio,
                        1.0 - float(cfg.clip_range),
                        1.0 + float(cfg.clip_range),
                    )
                    policy_loss = torch.max(pg_loss_1, pg_loss_2).mean()
                    entropy = out.entropy.mean()
                    actor_loss = policy_loss - float(cfg.ent_coef) * entropy

                    actor_optimizer.zero_grad(set_to_none=True)
                    actor_loss.backward()
                    actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                        actor.parameters(),
                        float(cfg.max_grad_norm),
                    )
                    actor_optimizer.step()

                    with torch.no_grad():
                        approx_kl = ((ratio - 1.0) - log_ratio).mean()
                        clip_frac = (
                            (torch.abs(ratio - 1.0) > float(cfg.clip_range)).float().mean()
                        )
                    policy_losses.append(float(policy_loss.detach().cpu()))
                    entropies.append(float(entropy.detach().cpu()))
                    approx_kls.append(float(approx_kl.detach().cpu()))
                    epoch_kls.append(float(approx_kl.detach().cpu()))
                    clip_fracs.append(float(clip_frac.detach().cpu()))
                    actor_grad_norms.append(float(torch.as_tensor(actor_grad_norm).detach().cpu()))

                for critic_batch in buffer.iter_critic_minibatches(
                    max(1, int(cfg.minibatch_size) // len(ZONES)),
                    shuffle=True,
                ):
                    values = critic(critic_batch.states)
                    value_loss = 0.5 * torch.square(critic_batch.returns - values).mean()
                    critic_loss = float(cfg.vf_coef) * value_loss

                    critic_optimizer.zero_grad(set_to_none=True)
                    critic_loss.backward()
                    critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                        critic.parameters(),
                        float(cfg.max_grad_norm),
                    )
                    critic_optimizer.step()

                    value_losses.append(float(value_loss.detach().cpu()))
                    critic_grad_norms.append(float(torch.as_tensor(critic_grad_norm).detach().cpu()))

                if float(cfg.target_kl) > 0.0 and epoch_kls:
                    mean_epoch_kl = float(np.mean(epoch_kls))
                    if mean_epoch_kl > float(cfg.target_kl):
                        early_stop_epoch = int(_epoch + 1)
                        early_stop_kl = mean_epoch_kl
                        break

            critic_value_pred = buffer.values[: buffer.pos]
            critic_return = buffer.critic_returns
            time_elapsed = max(1e-9, time.time() - start_time)
            fps = int(total_env_steps / time_elapsed)
            ep_rew_mean = (
                float(np.mean(completed_episode_returns[-100:]))
                if completed_episode_returns
                else float(np.sum(recent_rewards[-rollout_steps:]))
            )
            ep_rew_mean_team = (
                float(np.mean(completed_episode_team_returns[-100:]))
                if completed_episode_team_returns
                else float(np.sum(recent_team_rewards[-rollout_steps:]))
            )
            ep_rew_mean_zone = {
                zone: (
                    float(np.mean(completed_episode_zone_returns[zone][-100:]))
                    if completed_episode_zone_returns[zone]
                    else float(np.sum(recent_zone_rewards[zone][-rollout_steps:]))
                )
                for zone in ZONES
            }
            ep_len_mean = (
                float(np.mean(completed_episode_lengths[-100:]))
                if completed_episode_lengths
                else float(current_episode_len)
            )
            explained_var = explained_variance(critic_value_pred, critic_return)
            row = {
                "update": update_idx,
                "updates_total": n_updates,
                "env_steps": total_env_steps,
                "rollout/ep_len_mean": ep_len_mean,
                "rollout/ep_rew_mean": ep_rew_mean,
                "rollout/ep_rew_mean_team": ep_rew_mean_team,
                "rollout/rollout_return_mean": rollout_return / max(1, rollout_steps),
                "rollout/rollout_return_mean_team": rollout_team_return / max(1, rollout_steps),
                "rollout/recent_reward_mean": float(np.mean(recent_rewards[-100:])) if recent_rewards else 0.0,
                "rollout/recent_reward_mean_team": (
                    float(np.mean(recent_team_rewards[-100:])) if recent_team_rewards else 0.0
                ),
                "rollout/raw_step_reward_mean": rollout_return / max(1, rollout_steps),
                "rollout/raw_step_reward_mean_team": rollout_team_return / max(1, rollout_steps),
                "rollout/norm_step_reward_mean": rollout_norm_return / max(1, rollout_steps),
                "rollout/invalid_actions": invalid_actions,
                "time/fps": fps,
                "time/iterations": update_idx,
                "time/time_elapsed": int(time_elapsed),
                "time/total_timesteps": total_env_steps,
                "train/learning_rate": float(current_lr),
                "train/lr_schedule": str(cfg.lr_schedule),
                "train/norm_reward": float(bool(cfg.norm_reward)),
                "train/target_kl": float(cfg.target_kl),
                "train/early_stop_epoch": int(early_stop_epoch),
                "train/early_stop_kl": float(early_stop_kl),
                "train/reward_ret_var_mean": (
                    float(np.mean(reward_normalizer.ret_rms.var))
                    if reward_normalizer is not None
                    else 0.0
                ),
                "train/entropy_loss": -float(np.mean(entropies)),
                "train/policy_gradient_loss": float(np.mean(policy_losses)),
                "train/value_loss": float(np.mean(value_losses)),
                "train/approx_kl": float(np.mean(approx_kls)),
                "train/clip_fraction": float(np.mean(clip_fracs)),
                "train/clip_range": float(cfg.clip_range),
                "train/explained_variance": explained_var,
                "train/actor_grad_norm": float(np.mean(actor_grad_norms)),
                "train/critic_grad_norm": float(np.mean(critic_grad_norms)),
                # Backward-compatible flat CSV aliases.
                "rollout_return_mean": rollout_return / max(1, rollout_steps),
                "recent_reward_mean": float(np.mean(recent_rewards[-100:])) if recent_rewards else 0.0,
                "policy_loss": float(np.mean(policy_losses)),
                "value_loss": float(np.mean(value_losses)),
                "entropy": float(np.mean(entropies)),
                "approx_kl": float(np.mean(approx_kls)),
                "clip_frac": float(np.mean(clip_fracs)),
                "grad_norm": float(np.mean(actor_grad_norms)),
                "actor_grad_norm": float(np.mean(actor_grad_norms)),
                "critic_grad_norm": float(np.mean(critic_grad_norms)),
                "explained_variance": explained_var,
                "invalid_actions": invalid_actions,
            }
            for zone in ZONES:
                row[f"rollout/ep_rew_mean_{zone}"] = ep_rew_mean_zone[zone]
                row[f"rollout/rollout_return_mean_{zone}"] = (
                    rollout_zone_returns[zone] / max(1, rollout_steps)
                )
                row[f"rollout/recent_reward_mean_{zone}"] = (
                    float(np.mean(recent_zone_rewards[zone][-100:]))
                    if recent_zone_rewards[zone]
                    else 0.0
                )
                row[f"rollout/raw_step_reward_mean_{zone}"] = (
                    rollout_zone_returns[zone] / max(1, rollout_steps)
                )
            if metric_logger is not None:
                metric_logger.write(row)
            if writer is not None:
                for key, value in row.items():
                    if isinstance(value, (int, float)) and np.isfinite(value):
                        writer.add_scalar(key, float(value), total_env_steps)

            print_sb3_style_log(
                {
                    "rollout": {
                        "ep_len_mean": row["rollout/ep_len_mean"],
                        "ep_rew_mean": row["rollout/ep_rew_mean"],
                        "ep_rew_mean_team": row["rollout/ep_rew_mean_team"],
                        "raw_step_reward_mean": row["rollout/raw_step_reward_mean"],
                        "raw_step_reward_mean_team": row["rollout/raw_step_reward_mean_team"],
                        "norm_step_reward_mean": row["rollout/norm_step_reward_mean"],
                        "invalid_actions": row["rollout/invalid_actions"],
                    },
                    "rollout_zone": {
                        f"ep_rew_mean_{zone}": row[f"rollout/ep_rew_mean_{zone}"]
                        for zone in ZONES
                    },
                    "time": {
                        "fps": row["time/fps"],
                        "iterations": row["time/iterations"],
                        "time_elapsed": row["time/time_elapsed"],
                        "total_timesteps": row["time/total_timesteps"],
                    },
                    "train": {
                        "approx_kl": row["train/approx_kl"],
                        "clip_fraction": row["train/clip_fraction"],
                        "clip_range": row["train/clip_range"],
                        "entropy_loss": row["train/entropy_loss"],
                        "explained_variance": row["train/explained_variance"],
                        "learning_rate": row["train/learning_rate"],
                        "target_kl": row["train/target_kl"],
                        "early_stop_epoch": row["train/early_stop_epoch"],
                        "early_stop_kl": row["train/early_stop_kl"],
                        "norm_reward": row["train/norm_reward"],
                        "reward_ret_var_mean": row["train/reward_ret_var_mean"],
                        "policy_gradient_loss": row["train/policy_gradient_loss"],
                        "value_loss": row["train/value_loss"],
                        "actor_grad_norm": row["train/actor_grad_norm"],
                        "critic_grad_norm": row["train/critic_grad_norm"],
                    },
                }
            )

            if int(cfg.checkpoint_freq_updates) > 0 and update_idx % int(cfg.checkpoint_freq_updates) == 0:
                iter_name = f"{resolved_name}_iter_{update_idx:04d}"
                latest_name = f"{resolved_name}_latest"
                iter_paths = _save_eval_artifacts(
                    model_dir=model_dir,
                    artifact_name=iter_name,
                    actor=actor,
                    critic=critic,
                    cfg=cfg,
                    actor_obs_dim=actor_obs_dim,
                    state_dim=state_dim,
                    actor_obs_rms=actor_obs_rms,
                    state_rms=state_rms,
                    reward_normalizer=reward_normalizer,
                )
                latest_paths = _save_eval_artifacts(
                    model_dir=model_dir,
                    artifact_name=latest_name,
                    actor=actor,
                    critic=critic,
                    cfg=cfg,
                    actor_obs_dim=actor_obs_dim,
                    state_dim=state_dim,
                    actor_obs_rms=actor_obs_rms,
                    state_rms=state_rms,
                    reward_normalizer=reward_normalizer,
                )
                print(f"[MAPPO] saved periodic checkpoint: {iter_paths[0]}")
                print(f"[MAPPO] updated latest checkpoint: {latest_paths[0]}")

            if total_env_steps in checkpoint_milestones and total_env_steps not in saved_checkpoint_milestones:
                step_name = f"{resolved_name}_step_{total_env_steps:07d}"
                latest_name = f"{resolved_name}_latest"
                step_paths = _save_eval_artifacts(
                    model_dir=model_dir,
                    artifact_name=step_name,
                    actor=actor,
                    critic=critic,
                    cfg=cfg,
                    actor_obs_dim=actor_obs_dim,
                    state_dim=state_dim,
                    actor_obs_rms=actor_obs_rms,
                    state_rms=state_rms,
                    reward_normalizer=reward_normalizer,
                )
                latest_paths = _save_eval_artifacts(
                    model_dir=model_dir,
                    artifact_name=latest_name,
                    actor=actor,
                    critic=critic,
                    cfg=cfg,
                    actor_obs_dim=actor_obs_dim,
                    state_dim=state_dim,
                    actor_obs_rms=actor_obs_rms,
                    state_rms=state_rms,
                    reward_normalizer=reward_normalizer,
                )
                saved_checkpoint_milestones.add(total_env_steps)
                print(f"[MAPPO] saved step checkpoint: {step_paths[0]}")
                print(f"[MAPPO] updated latest checkpoint: {latest_paths[0]}")

        ckpt_path, norm_path, _cfg_path = _save_eval_artifacts(
            model_dir=model_dir,
            artifact_name=resolved_name,
            actor=actor,
            critic=critic,
            cfg=cfg,
            actor_obs_dim=actor_obs_dim,
            state_dim=state_dim,
            actor_obs_rms=actor_obs_rms,
            state_rms=state_rms,
            reward_normalizer=reward_normalizer,
        )
        print(f"[MAPPO] saved checkpoint: {ckpt_path}")
        print(f"[MAPPO] saved normalizers: {norm_path}")
        return actor, critic
    finally:
        if writer is not None:
            writer.close()
        env.close()
