# flyvis-closedloop (`flyloop`)

A closed-loop virtual fly whose steering comes only from a connectome-constrained
visual system. Photoreceptors -> T4/T5 (pretrained [flyvis](https://github.com/TuragaLab/flyvis))
-> HS/VS lobula plate tangential cells (connectome weights) -> steering controller ->
body -> rendered environment. See `CLAUDE.md` for the full project description.

## Layout

```
src/flyloop/
  stimuli/     gratings, moving edges, rotating gratings, 1/f^alpha textures
  eyes/        scene -> hex lattice sampling (closed loop, later)
  vision/      pretrained flyvis wrapper returning T4/T5 time series
  lptc/        HS/VS layer (later)
  control/     steering controllers (later)
  sim/         kinematic body + closed-loop driver (later)
  connectome/  FlyWire / neuPrint loaders (later)
  perturb/     noise, tau scaling, ablation hooks (later)
  metrics/     DSI, preferred direction, behavioral metrics
configs/       Hydra configs, one per experiment
scripts/       entry points
slurm/         Adroit job templates
notebooks/     exploration
tests/         pytest
runs/          outputs (gitignored)
```

## Setup

```bash
make env     # create the env and download the pretrained flyvis ensemble
make test
```

On the Mac, `make env` builds a uv venv in `.venv` (Python 3.10). Where uv is
missing (Adroit), it builds the conda env from `environment.yml`, which you must
activate first (`conda activate flyloop`). Pretrained models land in
`data/flyvis` (gitignored); `flyloop` sets `FLYVIS_ROOT_DIR` there unless you
export your own.

flyvis picks its device at import (`cuda` if available, else `cpu`); it never
selects Apple MPS by itself. See `scripts/benchmark_speed.py` for MPS timings.
