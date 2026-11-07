"""
Run loose-CTDE MAPPO training for the multizone PMV MMV pipeline.
"""

from __future__ import annotations

import argparse
import shutil
import socket
import subprocess
import sys
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
_ROOT = _THIS_FILE.parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agents.mappo_train import train_mappo
from agents.mappo_train import resolve_tb_run_dir
from scripts.run_config import (
    DEFAULT_RUN_CONFIG,
    make_model_name,
    model_dir,
    resolve_model_name,
    tensorboard_log_dir,
    with_overrides,
)


def _port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def _next_free_port(start_port: int, *, max_tries: int = 20) -> int:
    for port in range(int(start_port), int(start_port) + int(max_tries)):
        if not _port_in_use(port):
            return port
    raise RuntimeError(f"No free TensorBoard port found from {start_port} to {start_port + max_tries - 1}")


def start_tensorboard(logdir: Path, port: int = 6100) -> None:
    logdir = logdir.resolve()
    tb_exe = shutil.which("tensorboard")
    logdir.mkdir(parents=True, exist_ok=True)

    requested_port = int(port)
    launch_port = _next_free_port(requested_port)
    if launch_port != requested_port:
        print(f"TensorBoard port {requested_port} is already in use.")
        print(
            "Starting a new TensorBoard instance on "
            f"http://localhost:{launch_port}/ so the logdir is not inherited from an older run."
        )

    if tb_exe is None:
        print("TensorBoard executable not found in the active environment; continuing without it.")
        return

    subprocess.Popen(
        [tb_exe, "--logdir", str(logdir), "--port", str(launch_port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
    )
    print(f"TensorBoard started at http://localhost:{launch_port}/")
    print(f"TensorBoard logdir: {logdir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train loose-CTDE MAPPO for multizone PMV MMV.")
    parser.add_argument("--run-tag", type=str, default="", help="Tag appended to model name.")
    parser.add_argument("--model-name", type=str, default="", help="Base model name override.")
    parser.add_argument("--seed", type=int, default=None, help="Seed override.")
    parser.add_argument("--month", type=int, default=None, help="Month override (1-12).")
    parser.add_argument("--full-year", action="store_true", help="Use full-year random starts instead of the default month window.")
    parser.add_argument("--total-timesteps", type=int, default=None, help="Training env-step override.")
    parser.add_argument("--rollout-steps", type=int, default=None, help="MAPPO rollout length.")
    parser.add_argument("--minibatch-size", type=int, default=None, help="PPO minibatch size.")
    parser.add_argument("--ppo-epochs", type=int, default=None, help="PPO epochs per rollout.")
    parser.add_argument("--learning-rate", type=float, default=None, help="Initial Adam learning rate.")
    parser.add_argument("--final-learning-rate", type=float, default=None, help="Final learning rate for linear schedule.")
    parser.add_argument("--lr-schedule", choices=("linear", "constant"), default=None, help="Learning-rate schedule.")
    parser.add_argument("--gamma", type=float, default=None, help="Discount factor.")
    parser.add_argument("--gae-lambda", type=float, default=None, help="GAE lambda.")
    parser.add_argument("--clip-range", type=float, default=None, help="PPO clip range.")
    parser.add_argument("--target-kl", type=float, default=None, help="Stop PPO epochs early when approximate KL exceeds this value. Use 0 to disable.")
    parser.add_argument("--vf-coef", type=float, default=None, help="Value loss coefficient.")
    parser.add_argument("--ent-coef", type=float, default=None, help="Entropy coefficient.")
    parser.add_argument("--max-grad-norm", type=float, default=None, help="Gradient clipping norm.")
    parser.add_argument("--no-norm-obs", action="store_true", help="Disable actor/state observation normalization.")
    parser.add_argument("--no-norm-reward", action="store_true", help="Disable VecNormalize-style reward normalization.")
    parser.add_argument("--clip-obs", type=float, default=None, help="Observation normalization clip value.")
    parser.add_argument("--clip-reward", type=float, default=None, help="Reward normalization clip value.")
    parser.add_argument(
        "--checkpoint-freq-updates",
        type=int,
        default=None,
        help="Save eval checkpoints every N PPO updates. Use 0 to disable periodic checkpoints.",
    )
    parser.add_argument(
        "--checkpoint-milestone-steps",
        type=str,
        default=None,
        help="Comma-separated env-step milestones to save, e.g. 800000. Use an empty string to disable.",
    )
    parser.add_argument("--w-energy", type=float, default=None, help="Reward energy weight.")
    parser.add_argument("--w-pmv", type=float, default=None, help="PMV comfort penalty weight.")
    parser.add_argument("--w-switch", type=float, default=None, help="Switch penalty weight.")
    parser.add_argument("--w-sp-jump", type=float, default=None, help="Setpoint jump penalty weight.")
    parser.add_argument("--w-co2", type=float, default=None, help="Occupied CO2 excess penalty weight.")
    parser.add_argument("--co2-setpoint-ppm", type=float, default=None, help="Occupied CO2 setpoint in ppm.")
    parser.add_argument("--co2-scale-ppm", type=float, default=None, help="CO2 excess normalization scale in ppm.")
    parser.add_argument("--nv-bonus-beta", type=float, default=None, help="Occupied NV bonus.")
    parser.add_argument("--nv-bonus-unocc-beta", type=float, default=None, help="Unoccupied NV bonus.")
    parser.add_argument("--w-nv-unbenefit", type=float, default=None, help="Unbeneficial NV penalty.")
    parser.add_argument("--tb-port", type=int, default=6301, help="TensorBoard port.")
    parser.add_argument("--no-tb", action="store_true", help="Disable TensorBoard auto-start and writer.")
    args = parser.parse_args()

    overrides = {}
    for arg_name, cfg_name in [
        ("seed", "seed"),
        ("month", "month"),
        ("total_timesteps", "total_timesteps"),
        ("rollout_steps", "rollout_steps"),
        ("minibatch_size", "minibatch_size"),
        ("ppo_epochs", "ppo_epochs"),
        ("learning_rate", "learning_rate"),
        ("final_learning_rate", "final_learning_rate"),
        ("lr_schedule", "lr_schedule"),
        ("gamma", "gamma"),
        ("gae_lambda", "gae_lambda"),
        ("clip_range", "clip_range"),
        ("target_kl", "target_kl"),
        ("vf_coef", "vf_coef"),
        ("ent_coef", "ent_coef"),
        ("max_grad_norm", "max_grad_norm"),
        ("clip_obs", "clip_obs"),
        ("clip_reward", "clip_reward"),
        ("checkpoint_freq_updates", "checkpoint_freq_updates"),
        ("checkpoint_milestone_steps", "checkpoint_milestone_steps"),
        ("w_energy", "w_energy"),
        ("w_pmv", "w_pmv"),
        ("w_switch", "w_switch"),
        ("w_sp_jump", "w_sp_jump"),
        ("w_co2", "w_co2"),
        ("co2_setpoint_ppm", "co2_setpoint_ppm"),
        ("co2_scale_ppm", "co2_scale_ppm"),
        ("nv_bonus_beta", "nv_bonus_beta"),
        ("nv_bonus_unocc_beta", "nv_bonus_unocc_beta"),
        ("w_nv_unbenefit", "w_nv_unbenefit"),
    ]:
        value = getattr(args, arg_name)
        if value is not None:
            overrides[cfg_name] = value
    if args.model_name:
        overrides["model_name"] = args.model_name
    if args.full_year:
        overrides["month"] = None
    if args.no_norm_obs:
        overrides["norm_obs"] = False
    if args.no_norm_reward:
        overrides["norm_reward"] = False

    cfg = with_overrides(DEFAULT_RUN_CONFIG, **overrides)
    base_model_name = resolve_model_name(cfg)
    model_name = make_model_name(
        base_model_name,
        run_tag=args.run_tag if args.run_tag else None,
        seed=cfg.seed,
    )
    print(f"[MAPPO-TRAIN] model_name={model_name}")
    print(f"[MAPPO-TRAIN] ctde_mode={cfg.ctde_mode} reward_mode={cfg.reward_mode}")
    print(f"[MAPPO-TRAIN] norm_obs={cfg.norm_obs} norm_reward={cfg.norm_reward}")
    print(f"[MAPPO-TRAIN] dt_comm_s={cfg.dt_comm_s} rollout_steps={cfg.rollout_steps}")

    tb_logdir = tensorboard_log_dir()
    models_dir = model_dir()
    models_dir.mkdir(parents=True, exist_ok=True)
    tb_run_dir = None
    if args.no_tb:
        metrics_csv = tb_logdir / f"{model_name}_metrics.csv"
    else:
        # Mirror SB3's TensorBoard run suffix for the sidecar CSV too, so
        # repeated runs with the same model name do not append to old metrics.
        tb_run_dir = resolve_tb_run_dir(tb_logdir, model_name)
        metrics_csv = tb_run_dir.with_name(f"{tb_run_dir.name}_metrics.csv")

    if not args.no_tb:
        start_tensorboard(tb_logdir, port=args.tb_port)

    train_mappo(
        cfg=cfg,
        model_dir=models_dir,
        model_name=model_name,
        tensorboard_log=None if args.no_tb else tb_logdir,
        tensorboard_run_dir=tb_run_dir,
        metrics_csv=metrics_csv,
        device="cpu",
    )


if __name__ == "__main__":
    main()
