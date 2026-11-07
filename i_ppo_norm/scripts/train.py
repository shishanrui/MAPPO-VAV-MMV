"""
Run PPO training for the PMV MMV pipeline.
"""

import sys
import argparse
import shutil
import socket
import subprocess
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
_ROOT = _THIS_FILE.parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agents.ppo_train import train_ppo
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


def start_tensorboard(logdir: Path, port: int = 6101) -> None:
    logdir = logdir.resolve()
    tb_exe = shutil.which("tensorboard")
    logdir.mkdir(parents=True, exist_ok=True)

    if _port_in_use(port):
        print(f"TensorBoard already running at http://localhost:{port}/")
        print(f"TensorBoard logdir: {logdir}")
        return

    if tb_exe is None:
        print("TensorBoard executable not found in the active environment; continuing without it.")
        return

    subprocess.Popen(
        [tb_exe, "--logdir", str(logdir), "--port", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
    )
    print(f"TensorBoard started at http://localhost:{port}/")
    print(f"TensorBoard logdir: {logdir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO for PMV MMV with optional per-run overrides.")
    parser.add_argument("--run-tag", type=str, default="", help="Tag appended to model name.")
    parser.add_argument("--model-name", type=str, default="", help="Base model name override.")
    parser.add_argument("--seed", type=int, default=None, help="Seed override.")
    parser.add_argument("--month", type=int, default=None, help="Month override (1-12).")
    parser.add_argument("--total-timesteps", type=int, default=None, help="Training timesteps override.")
    parser.add_argument(
        "--checkpoint-milestone-steps",
        type=str,
        default=None,
        help="Comma-separated env-step milestones to save, e.g. 800000. Use an empty string to disable.",
    )
    parser.add_argument("--w-energy", type=float, default=None, help="Reward weight override.")
    parser.add_argument("--w-pmv", type=float, default=None, help="PMV comfort penalty weight.")
    parser.add_argument("--w-nv-blocked", type=float, default=None, help="Penalty for requesting blocked NV.")
    parser.add_argument("--w-min-ac-hold", type=float, default=None, help="Penalty when policy tries to leave AC during minimum AC-on hold.")
    parser.add_argument("--w-unocc-ac", type=float, default=None, help="Per-step penalty when AC is applied while unoccupied.")
    parser.add_argument("--w-switch", type=float, default=None, help="Switch penalty weight override.")
    parser.add_argument("--w-sp-jump", type=float, default=None, help="Large AC setpoint jump penalty weight override.")
    parser.add_argument("--w-co2", type=float, default=None, help="Occupied CO2 excess penalty weight.")
    parser.add_argument("--co2-setpoint-ppm", type=float, default=None, help="Occupied CO2 setpoint in ppm.")
    parser.add_argument("--co2-scale-ppm", type=float, default=None, help="CO2 excess normalization scale in ppm.")
    parser.add_argument("--comfort-bonus-beta", type=float, default=None, help="All-zones-comfortable inner PMV bonus.")
    parser.add_argument("--nv-bonus-beta", type=float, default=None, help="Occupied NV bonus override.")
    parser.add_argument("--nv-bonus-unocc-beta", type=float, default=None, help="Unoccupied beneficial-NV bonus override.")
    parser.add_argument("--w-nv-unbenefit", type=float, default=None, help="Penalty for allowed but unbeneficial NV requests.")
    parser.add_argument("--ent-coef", type=float, default=None, help="PPO entropy coefficient override.")
    parser.add_argument("--tb-port", type=int, default=6303, help="TensorBoard port.")
    parser.add_argument("--no-tb", action="store_true", help="Disable TensorBoard auto-start.")
    args = parser.parse_args()

    overrides = {}
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.month is not None:
        overrides["month"] = args.month
    if args.total_timesteps is not None:
        overrides["total_timesteps"] = args.total_timesteps
    if args.checkpoint_milestone_steps is not None:
        overrides["checkpoint_milestone_steps"] = args.checkpoint_milestone_steps
    if args.w_energy is not None:
        overrides["w_energy"] = args.w_energy
    if args.w_pmv is not None:
        overrides["w_pmv"] = args.w_pmv
    if args.w_nv_blocked is not None:
        overrides["w_nv_blocked"] = args.w_nv_blocked
    if args.w_min_ac_hold is not None:
        overrides["w_min_ac_hold"] = args.w_min_ac_hold
    if args.w_unocc_ac is not None:
        overrides["w_unocc_ac"] = args.w_unocc_ac
    if args.w_switch is not None:
        overrides["w_switch"] = args.w_switch
    if args.w_sp_jump is not None:
        overrides["w_sp_jump"] = args.w_sp_jump
    if args.w_co2 is not None:
        overrides["w_co2"] = args.w_co2
    if args.co2_setpoint_ppm is not None:
        overrides["co2_setpoint_ppm"] = args.co2_setpoint_ppm
    if args.co2_scale_ppm is not None:
        overrides["co2_scale_ppm"] = args.co2_scale_ppm
    if args.comfort_bonus_beta is not None:
        overrides["comfort_bonus_beta"] = args.comfort_bonus_beta
    if args.nv_bonus_beta is not None:
        overrides["nv_bonus_beta"] = args.nv_bonus_beta
    if args.nv_bonus_unocc_beta is not None:
        overrides["nv_bonus_unocc_beta"] = args.nv_bonus_unocc_beta
    if args.w_nv_unbenefit is not None:
        overrides["w_nv_unbenefit"] = args.w_nv_unbenefit
    if args.ent_coef is not None:
        overrides["ent_coef"] = args.ent_coef
    if args.model_name:
        overrides["model_name"] = args.model_name

    cfg = with_overrides(DEFAULT_RUN_CONFIG, **overrides)
    base_model_name = resolve_model_name(cfg)
    model_name = make_model_name(
        base_model_name,
        run_tag=args.run_tag if args.run_tag else None,
        seed=cfg.seed,
    )
    print(f"[TRAIN] model_name={model_name}")
    print(f"[TRAIN] month={cfg.month if cfg.month is not None else 'all-year'}")
    print(f"[TRAIN] Using dt_comm_s={cfg.dt_comm_s}")

    tb_logdir = tensorboard_log_dir()
    models_dir = model_dir()

    if not args.no_tb:
        start_tensorboard(tb_logdir, port=args.tb_port)

    train_ppo(
        month=cfg.month,
        dt_comm_s=cfg.dt_comm_s,
        total_timesteps=cfg.total_timesteps,
        w_energy=cfg.w_energy,
        w_pmv=cfg.w_pmv,
        w_nv_blocked=cfg.w_nv_blocked,
        w_min_ac_hold=cfg.w_min_ac_hold,
        w_unocc_ac=cfg.w_unocc_ac,
        w_switch=cfg.w_switch,
        w_sp_jump=cfg.w_sp_jump,
        nv_bonus_beta=cfg.nv_bonus_beta,
        nv_bonus_unocc_beta=cfg.nv_bonus_unocc_beta,
        w_nv_unbenefit=cfg.w_nv_unbenefit,
        w_co2=cfg.w_co2,
        co2_setpoint_ppm=cfg.co2_setpoint_ppm,
        co2_scale_ppm=cfg.co2_scale_ppm,
        comfort_bonus_beta=cfg.comfort_bonus_beta,
        ent_coef=cfg.ent_coef,
        model_dir=str(models_dir),
        model_name=model_name,
        tensorboard_log=str(tb_logdir),
        seed=cfg.seed,
        checkpoint_milestone_steps=cfg.checkpoint_milestone_steps,
    )


if __name__ == "__main__":
    main()
