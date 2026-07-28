# TRACE-CRC Reproducibility Parameters

This document records the parameters used by the canonical COPA 2026
reproducibility pipeline:

- [`diu_conformal_all_in_one_reproducibility.ipynb`](diu_conformal_all_in_one_reproducibility.ipynb)
- [`parameter_selection.json`](final_comparison_revision_all_in_one_reproducibility/parameter_selection.json)
- [`robustness_summary.json`](final_comparison_revision_all_in_one_reproducibility/robustness_summary.json)

The notebook is the executable source of truth. The JSON artifacts are the
machine-readable records produced by a successful run. Notebooks under
`archive/` are retained only as development history.

## Environment

The pipeline was tested with CPython 3.13.7. Exact package versions are pinned
in [`requirements.txt`](../requirements.txt).

From the repository root:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m ipykernel install --user --name trace-crc --display-name "TRACE-CRC"
jupyter lab
```

The full dependency file includes TensorFlow and Sionna for CSI dataset
generation. They are not called when the provided dataset or validated compact
inference caches are available.

## Required Inputs

| Input | Expected path | Purpose |
|---|---|---|
| CSI trajectories | `data/csi/CSI_Dataset_1000_diverse.pt` | 1,000 trajectory-level CSI samples |
| DiU checkpoint | `DiU/models/models_900.pt` | Pretrained diffusion predictor |
| Model configuration | `DiU/config.yml` | ConvLSTM, UNet, scheduler, and normalization settings |
| Inference helper | `DiU/inference_helpers.py` | Checkpoint loading and deterministic inference |

The dataset and checkpoint are intentionally excluded from Git because they are
large. A public release must provide download instructions and checksums for
both files. If the five validated compact caches are distributed, the notebook
can reproduce conformal tuning, evaluation, tables, and figures without
rerunning diffusion inference.

Expected SHA-256 hashes:

| File | SHA-256 |
|---|---|
| `CSI_Dataset_1000_diverse.pt` | `cb19854166b662e4badcbc1f5374f4a424c93013ad5147d3e5bed078236a3d0c` |
| `models_900.pt` | `800227c94359977bd80c4cef33a418fb06eb1ef5a0bddf52e5a7b73ba2e5f37b` |
| `compact_seed_20260724.npz` | `d84ee36674b4526120704eed213bd9b2816472180111b86e32fd5e60aa52d10b` |
| `compact_seed_20260824.npz` | `2b30b3fd0cf2b499c498505ddc58b8e1755f9f8b047566b4eb8f38496d604315` |
| `compact_seed_20260924.npz` | `d16daecbe54121af356496251513f3f1c9395f5adce18426fc9fa62961940f25` |
| `compact_seed_20261024.npz` | `f31184748343115be092d39042c6da5337bd953887a8aadfe10e0b6321f5c1b6` |
| `compact_seed_20261124.npz` | `da3c44737610a2176e8aa7a76af7b579fee85eed10753f46a64a00fc6f2024ab` |

## Diffusion Inference

| Parameter | Value |
|---|---:|
| Number of trajectories | 1,000 |
| Frames stored per source trajectory | 100 |
| Frames used per experiment | first 30 |
| Input frames, `T_in` | 10 |
| Forecast frames, `T_out` | 20 |
| CSI matrix shape | `2 x 16 x 16` real/imaginary representation |
| Inference SNR | 20 dB |
| DDIM sampling steps | 20 |
| DDIM `eta` | 0.0 |
| Guidance scale | 1.0 |
| Noise level | 100.0 |
| Inference batch size | 16 |
| Checkpoint averaging | EMA weights |

The inference seeds are:

```text
20260724, 20260824, 20260924, 20261024, 20261124
```

For every seed, Python, NumPy, PyTorch, CUDA when available, and MPS when
available are seeded. Deterministic PyTorch algorithms are requested with
`warn_only=True`; therefore exact floating-point identity across operating
systems and accelerators is not guaranteed. Reusing the validated compact
caches avoids accelerator-dependent regeneration.

Each compact cache stores:

- Frobenius residuals with shape `(1000, 20)`
- predicted CSI Frobenius norms with shape `(1000, 20)`
- horizon-wise NMSE
- JSON metadata containing the seed, data paths, checkpoint, horizon, SNR,
  sampling steps, and batch size

`REBUILD_SEED_CACHES=False` is the default. A cache is loaded only when its
metadata exactly matches the active inference settings. Set it to `True` only
when intentionally regenerating all seed caches.

## Data Splits

The primary experiment uses a random trajectory-level split, not a
chronological split.

| Parameter | Value |
|---|---:|
| Calibration fraction | 0.30 |
| Calibration trajectories | 300 |
| Test trajectories | 700 |
| Primary split seed | 314159 |

`np.random.default_rng(314159).permutation(1000)` determines the split. The
first 300 permuted indices form calibration. The remaining 700 test indices are
sorted before evaluation to give online adaptive baselines a stable traversal
order. Calibration and test trajectories are disjoint.

TRACE-CRC split robustness uses ten additional predeclared random split seeds:

```text
1101, 1102, 1103, 1104, 1105, 1106, 1107, 1108, 1109, 1110
```

Together with the five inference seeds, this produces 50 TRACE-CRC robustness
runs. These runs do not participate in the main all-method comparison.

## Shared Statistical Targets

| Parameter | Value |
|---|---:|
| Miscoverage/risk target, `alpha` | 0.10 |
| Target coverage, `1 - alpha` | 0.90 |
| LTT error level, `delta_ltt` | 0.10 |
| Numerical tolerance, `eps` | `1e-8` |

All LTT losses are trajectory-failure indicators:

```text
loss_i(lambda) = 1 if any forecast horizon exceeds its radius, otherwise 0
```

Thus, the LTT stage targets trajectory-level risk, not mean stepwise risk.

## Baseline Parameter Tuning

Baseline tuning uses only the 300 primary calibration trajectories for each
inference seed:

| Subset | Per-seed size |
|---|---:|
| Tuning training | 210 |
| Tuning validation | 90 |
| Test used during tuning | 0 |

Metrics are calculated separately for each inference seed and then aggregated;
the five seed datasets are not pooled as 1,500 independent trajectories.
Horizon methods target mean horizon coverage of at least 0.90 on every seed.
Trajectory methods target trajectory coverage of at least 0.90 on every seed.
Among all-seed-feasible candidates, the selected candidate minimizes mean
validation AFR, then maximizes minimum coverage, then minimizes AFR standard
deviation.

When no candidate in a prespecified grid is feasible on every seed, the
documented fallback minimizes:

1. maximum seed-level coverage shortfall;
2. mean seed-level coverage shortfall;
3. mean validation AFR.

This fallback is an empirical tuning rule and is not a formal coverage
certificate.

### Baseline Search Grids

| Family | Candidate grid |
|---|---|
| EW | `beta` in `{0.003, 0.005, 0.01, 0.02, 0.03, 0.05}` |
| EnbPI | `B` in `{50, 100}`; window in `{None, 100, 200, 300}`; center in `{mean, oob_quantile}` |
| ACI | `gamma` in `{0.001, 0.003, 0.005, 0.01, 0.02, 0.05}` |
| AgACI | three prespecified `Gamma` sets; `eta` in `{0.05, 0.10, 0.20, 0.50}` |
| CopulaCPTS | split in `{0.50, 0.60, 0.70, 0.80}`; epochs in `{400, 800, 1200}` |

The AgACI candidate sets are:

```text
{0, 0.0005, 0.001, 0.002, 0.005, 0.01}
{0, 0.001, 0.003, 0.005, 0.01, 0.02, 0.03}
{0, 0.005, 0.01, 0.02, 0.03, 0.05}
```

### Frozen Baseline Parameters

| Method | Selected parameters | Selection status |
|---|---|---|
| EW-Horizon | `beta=0.02` | fallback |
| EW-Trajectory | `beta=0.02` | fallback |
| EnbPI-Horizon | `B=50`, full history, mean center, Ridge penalty 1, seed 123 | all-seed feasible |
| EnbPI-Trajectory | `B=50`, window 100, mean center, Ridge penalty 1, seed 456 | all-seed feasible |
| ACI-Horizon | `gamma=0.05` | fallback |
| ACI-Trajectory | `gamma=0.02` | fallback |
| AgACI-Horizon | `Gamma={0, 0.005, 0.01, 0.02, 0.03, 0.05}`, `eta=0.05` | fallback |
| AgACI-Trajectory | `Gamma={0, 0.005, 0.01, 0.02, 0.03, 0.05}`, `eta=0.20` | fallback |
| CopulaCPTS | split `0.50/0.50`, 1,200 epochs, seed 123 | all-seed feasible |

Standard split conformal, residual quantile, Bonferroni-CRC, and Sidak-CRC
have no validation-selected hyperparameters in this pipeline.

## TRACE-CRC Structural Search

TRACE-CRC structural tuning uses the same five inference seeds and only the
primary calibration split. The prespecified search family is:

| Parameter | Candidate values |
|---|---|
| Profile fraction | `{0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40}` |
| CP fraction within the post-profile remainder | `{0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40}` |
| Number of trajectory groups | `2` fixed |
| Smoothing window, `K` | `3` |
| Stabilization ratio, `rho` | `{0.15, 0.25}` |
| Ridge penalty, `eta` | `{0.1, 1.0}` |
| Search multiplier grid | 10 uniform values on `[0.70, 1.30]` |
| Maximum mean-horizon AFR | 25.0 |

This gives 196 structural configurations and 1,960
configuration/multiplier hypotheses. For each hypothesis, the joint p-value is
the maximum Hoeffding-Bentkus p-value across the five inference seeds.
Holm stepdown correction is applied globally to the complete 1,960-hypothesis
family. Among certified candidates satisfying the AFR bound, selection
minimizes mean calibration AFR.

The selected and frozen TRACE-CRC structure is:

```text
trace_profile_fraction = 0.10
trace_cp_fraction_within_rest = 0.15
trace_num_groups = 2
trace_smooth_window = 3
trace_floor_ratio = 0.15
trace_ridge_alpha = 1.0
```

For 300 calibration trajectories, the resulting disjoint partition is:

| Subset | Purpose | Size |
|---|---|---:|
| `D_prof` | horizon profile and difficulty model | 30 |
| `D_cp` | global or groupwise conformal quantiles | 40 |
| `D_val` | trajectory-level LTT selection | 230 |

The split is taken in the order of the randomly permuted calibration indices;
it is not chronological.

## CRC Ablations

All risk-control methods use the same disjoint 30/40/230 calibration partition.
The default is:

```text
use_all_calib_global_crc = False
```

Therefore Global-CRC also uses only `D_val`, rather than all 300 calibration
trajectories, for LTT selection.

| Method | Radius before LTT multiplication | Data-dependent components |
|---|---|---|
| Global-CRC | `q_global` | `q_global` from `D_cp`; lambda from `D_val` |
| Horizon-Profile CRC | `q_global * w_j` | `w_j` from `D_prof`; `q_global` from `D_cp`; lambda from `D_val` |
| Trajectory-Stratified CRC | `q_g_raw` | grouping model from `D_prof`; raw group quantiles from `D_cp`; lambda from `D_val` |
| TRACE-CRC | `q_g * w_j` | profile/group model from `D_prof`; normalized group quantiles from `D_cp`; lambda from `D_val` |

The final radius is the corresponding preliminary radius multiplied by the
formally selected, seed-specific `lambda_star`.

## Final LTT Family

After structures and baseline parameters are frozen, every CRC method uses the
same prespecified 27-candidate multiplier family:

```text
10 uniform candidates on [0.70, 1.30]
plus 1.40, 1.50, ..., 3.00
```

Hoeffding-Bentkus p-values and a Holm-style stepdown correction are applied to
the complete family. The smallest accepted multiplier is selected. The
extension is part of the prespecified family and is not an adaptive empirical
fallback.

```text
lambda_grid_max_cap = 3.00
lambda_grid_expand_step = 0.10
empirical_fallback_allowed = False
```

`lambda_star`, `q_global`, `q_g`, and `w_j` are calibrated separately for each
inference seed. They are outputs, not globally frozen hyperparameters. Their
per-seed values are stored in
`tables/method_inference_seed_metrics.csv` and the serialized robustness
artifacts.

## Output Artifacts

All generated files are written only under:

```text
DiU/final_comparison_revision_all_in_one_reproducibility/
```

Important outputs:

| Artifact | Contents |
|---|---|
| `parameter_selection.json` | Tuning protocol, selected baselines, TRACE structure, and ablation settings |
| `robustness_summary.json` | Experiment configuration and aggregate robustness summaries |
| `reproduction_validation.json` | Checks against the previous validated artifact |
| `robustness_outputs.pkl` | Complete Python results and metadata |
| `robustness_arrays.npz` | Numeric arrays |
| `tables/baseline_tuning_candidates.csv` | Every baseline candidate and aggregate score |
| `tables/baseline_tuning_seed_metrics.csv` | Candidate metrics for every inference seed |
| `tables/method_inference_seed_metrics.csv` | Per-method, per-seed test results |
| `tables/method_inference_seed_per_horizon.csv` | Per-horizon coverage and AFR |
| `tables/method_inference_seed_summary.csv` | Mean and standard deviation across inference seeds |
| `figures/*.svg` and `figures/*.pdf` | Matplotlib paper and robustness figures |

In the main method figures, each line is the mean test curve across five
diffusion inference seeds and each shade is mean plus or minus one population
standard deviation across those five seed-level curves. The shade is not a
confidence interval and does not represent split-seed variability.

## Reproduction Checks

Before treating a run as paper-ready:

1. restart the kernel and run all cells in order;
2. confirm all five compact caches pass metadata validation;
3. confirm baseline and TRACE searches finish before test evaluation;
4. confirm no CRC method uses empirical fallback;
5. confirm every method has five primary test rows;
6. confirm TRACE robustness has 50 seed/split rows;
7. inspect `reproduction_validation.json`;
8. run `python -m pip check` in the active environment.
