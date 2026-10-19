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
make env     # create the conda env and download the pretrained flyvis ensemble
make test
```
