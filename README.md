# Residual Model Criticism for Fitted SDE Implementations

Reproduction code for the paper, including simulations, residual diagnostics,
figure builders and small result tables. Manuscript sources are not included.
Some figures and SIR analyses require the separate data and model bundle
described below.

The main paper develops residual comparisons with a fixed fitted numerical
implementation, studies propagation and weighting in a linear family, and
applies the diagnostic to reconstructed SIR paths. Nonlinear weight selection
and neural fitting comparisons are reported in the supplement.

## Installation

Use Python 3.12. Clone the repository and run commands from its root:

```text
git clone https://github.com/clemmentin/Residual-Model-Criticism-for-Fitted-SDE-Implementations.git
cd Residual-Model-Criticism-for-Fitted-SDE-Implementations
```

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

On Linux/macOS, replace `.\.venv\Scripts\python.exe` with `.venv/bin/python`.

## Reproduce the paper

```powershell
.\.venv\Scripts\python.exe reproduce.py tables        # Nonlinear and neural comparisons
.\.venv\Scripts\python.exe reproduce.py figures       # Figures, including archived comparisons; requires the bundle
.\.venv\Scripts\python.exe reproduce.py sir-settings  # Print the reported SIR settings
.\.venv\Scripts\python.exe reproduce.py sir-fit       # Fit the SIR model; requires the data
```

Tables are rebuilt from included per-fit estimates and written to
`output/tables/`; this does not rerun the experiments. Figures go to
`output/figures/`. Missing CZE/GRC residual paths are regenerated from saved
models and seeds, so the first figure run can take longer.
The current paper cites five main figures and seven supplementary figures.
Figure 2 is a conceptual diagram supplied in `assets/score_comparisons.pdf`;
the figure command copies it to the output directory without simulation.
The builder also retains three figures
from earlier analyses; these are not cited in the current paper.

SIR fitting writes to `output/sir_fit/` and reuses an existing fit there.
Use `--out-dir output/fresh_run` for a fresh fit. Full simulation and neural
training commands, seeds, and saved-output limitations are in
[REPRODUCE.md](REPRODUCE.md).

The feedback extension in Supplementary Section S3.7 is self-contained:

```powershell
.\.venv\Scripts\python.exe experiments/feedback_weight_selection/summarize.py
.\.venv\Scripts\python.exe experiments/feedback_weight_selection/joint_point_check.py
```

The first command rebuilds selection bounds and replays the 16 final tests
from included counts and seeds. The second rebuilds the supplementary
two-parameter exclusion table. Outputs go to `output/feedback_weight_selection/`.
See the experiment's [README](experiments/feedback_weight_selection/README.md)
for full simulation commands and the scope of each result.

## Data and models

Unpack the matching bundle's `cache/` and `models/` directories beside
`reproduce.py`. The bundle contains the archived OWID snapshot, checkpoints
and simulation results. On Windows, use a short path such as `C:\repro\`.
The source checkout alone cannot reproduce every result.

The [data and model deposit](https://doi.org/10.5281/zenodo.22212995)
currently has restricted file access. The included tables and self-contained
feedback study can be reproduced from this repository; analyses requiring
the archived data or fitted models need access to the separate bundle.
See [DATA_SOURCES.md](DATA_SOURCES.md) for sources and reuse terms.

## Files and documentation

- `reproduce.py`: main reproduction commands.
- `neural_sde.py`, `trajectory.py`, `losses.py`, `brownian_inversion.py`: model and residual calculations.
- `sir_*.py`: data preparation, training and evaluation.
- `experiments/`: paper experiments and figure scripts.
- `docs/`: reproduction notes, figure/table mapping and retained numerical checks.
- [Reproduction map](docs/REPRODUCTION_MAP.md): inputs and scripts for each figure and table.
- [Settings](SETTINGS.md): reported settings and differences from the development defaults in `config.py`.

`config1.py` only provides compatibility with archived pickle caches.

## Citation and licence

See [CITATION.cff](CITATION.cff) for citation metadata. Cite the code release
and matching data archive together. Original code is licensed under
[GPL-3.0-only](LICENSE); separate terms apply to data and third-party material.
