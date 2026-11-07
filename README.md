# Reinforcement learning for multizone mixed-mode ventilation

Research implementations of G-MAPPO, I-PPO, and U-PPO for supervisory control of a three-zone variable-air-volume (VAV) system with mixed-mode ventilation (MMV). Controllers coordinate OFF, natural ventilation (NV), and air-conditioning (AC) modes, with state-dependent feasibility masks.

## Controllers and code

| Controller | Formulation | Normal schedule | Heterogeneous schedule |
| --- | --- | --- | --- |
| G-MAPPO | Globally observed zonal policies, zonal PPO updates, and rewards combining local service with shared energy feedback | [g_mappo_norm](experiments_code/g_mappo_norm/) | [g_mappo_hetero](experiments_code/g_mappo_hetero/) |
| I-PPO | Centralized PPO with independent zonal modes and AC setpoints | [i_ppo_norm](experiments_code/i_ppo_norm/) | [i_ppo_hetero](experiments_code/i_ppo_hetero/) |
| U-PPO | Centralized PPO with one shared mode and independent zonal AC setpoints | [u_ppo_norm](experiments_code/u_ppo_norm/) | [u_ppo_hetero](experiments_code/u_ppo_hetero/) |

Each variant is self-contained: `agents/` implements learning, `mmv_env/` contains the environment and FMU interface, `scripts/` provides training and short evaluation, and `year_eval/` provides annual evaluation. Run commands from the selected variant's directory.

## Setup

The current runtime uses Python 3.11. Install the Python dependencies in a dedicated environment:

```powershell
python -m pip install numpy pandas scipy gymnasium torch stable-baselines3 sb3-contrib fmpy tensorboard
```

## Training

From the repository root, for example:

```powershell
cd experiments_code/g_mappo_hetero
python scripts/train.py --help
python scripts/train.py --seed 42 --w-energy 0.10 --no-tb
```

The same training command works in the other variant directories. Review `scripts/run_config.py` for training settings; current defaults vary between variants. The example explicitly selects the manuscript's reference energy weight, 0.10. Omit `--no-tb` to enable TensorBoard.

## Annual evaluation

After training, replace `MODEL` below with the saved checkpoint stem. G-MAPPO requires its matching observation/reward normalizer:

```powershell
python year_eval/evaluate_year.py --model-path scripts/models/MODEL.pt --normalizer-path scripts/models/MODEL_obsnorm.pkl --eval-window year
```

For I-PPO and U-PPO, use a `.zip` checkpoint and `--vecnorm-path scripts/models/MODEL_vecnorm.pkl` instead of `--normalizer-path`. Use `--fmu-path` and `--input-csv` to override evaluation inputs. Short-evaluation options are available through `python scripts/evaluate.py --help`.

## Citation
