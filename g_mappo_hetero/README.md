# g_mappo_hetero

## Run

Run commands from this directory with the MMV environment activated. On this workstation, the interpreter is `M:\Anaconda\envs\MMV\python.exe`.

```powershell
M:\Anaconda\envs\MMV\python.exe .\scripts\train.py --no-tb
M:\Anaconda\envs\MMV\python.exe .\scripts\evaluate.py --help
M:\Anaconda\envs\MMV\python.exe .\year_eval\evaluate_year.py --help
```

Remove `--no-tb` to start TensorBoard on the branch-specific default port `6300`. Evaluation expects a trained model in `scripts/models/` unless a model path is supplied on the command line.

## Isolation contract

- Imports resolve from this branch through the direct-run bootstraps in `scripts/` and `year_eval/`.
- Training models, TensorBoard logs, and short-evaluation CSVs stay under `scripts/`.
- Full-year evaluation output stays under `year_eval/out/`.
- Reward aggregation uses the mean per-zone PMV and CO2 penalties without worst-zone penalty multipliers; `co2_worst_excess` remains diagnostic only.
- Runtime data: `input_data_train_random.csv` for training and short evaluation; `input_data_eval_random.csv` for full-year evaluation.
- Both runtime FMUs are local `MultizoneSetpointsTrain.fmu` and `MultizoneSetpointsEval.fmu` files.
- Saved artifacts are MAPPO checkpoints (`.pt`) and observation normalizers.

## Retained runtime surface

- `agents/`: policy/training implementation
- `mmv_env/`: environment, reward, safety, and FMU backend
- `scripts/train.py`: training entrypoint
- `scripts/evaluate.py`: short evaluation entrypoint
- `scripts/run_config.py`: branch defaults and local output paths
- `year_eval/evaluate_year.py`: full-year evaluation entrypoint
- `year_eval/env_factory.py` and `year_eval/eval_config.py`: full-year runtime support
- `data/`: only the schedule CSVs and FMUs used by these entrypoints
