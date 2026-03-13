# FL Lab README

## Overview

This repository contains a federated learning experimentation setup built on top of the local `fluke_package` framework.
It currently covers:

- horizontal federated learning on tabular medical data
- private horizontal federated learning with differential privacy
- true vertical federated learning with aligned samples and feature splits
- metric and fairness analysis utilities
- plotting utilities for run summaries and round-wise progressions

The current experiment set is centered on the medical dataset family under `data/medical`.

Dataset source:
`data/medical/smoking.csv` comes from the Kaggle dataset
https://www.kaggle.com/datasets/kukuroo3/body-signal-of-smoking

Important note:
The current `medical` dataset loader default is `data/medical/smoking.csv` with target column `smoking`.
The configs in `config/` currently rely on that default because they only specify `name: medical` and `path: ./data`.
If you want to use `data/medical/medical.csv`, you must explicitly override `filename` and `target_col` in the dataset config.

## Main locations

Project-level files:

- `commands.txt`: ready-to-run experiment commands
- `config/`: experiment and method YAML files
- `data/`: local datasets
- `runs/`: run outputs, plotting scripts, generated figures and CSV summaries
- `README.md`: this file

Core framework files:

- `fluke_package/fluke/run.py`: CLI entry points for centralized, federation, decentralized, and sweep runs
- `fluke_package/fluke/data/datasets.py`: dataset loaders, including the `medical` loader
- `fluke_package/fluke/evaluation.py`: evaluation logic and fairness metrics
- `fluke_package/fluke/utils/log.py`: CSV logging and output file generation
- `fluke_package/fluke/nets.py`: model definitions
- `fluke_package/fluke/algorithms/`: FL algorithm implementations
- `fluke_package/fluke/algorithms/vertical.py`: true VFL implementation

## Configurations and scenarios

### Horizontal scenarios

Scenario files:

- `config/medical-data-iid.yaml`
- `config/medical-data-non-iid.yaml`

What they define:

- `medical-data-iid.yaml`: horizontal IID split across 10 clients
- `medical-data-non-iid.yaml`: horizontal non-IID split across 10 clients using Dirichlet distribution with `beta: 1`

Common characteristics:

- 20 rounds
- all clients eligible each round (`eligible_perc: 1.0`)
- client local test evaluation enabled
- server/global evaluation enabled
- CSV logging enabled

### Vertical scenarios

Scenario files:

- `config/medical-data-vertical-disjoint.yaml`
- `config/medical-data-vertical-overlap.yaml`
- `config/medical-data-vertical-disjoint-5-clients.yaml`
- `config/medical-data-vertical-overlap-5-clients.yaml`

What they define:

- `vertical-disjoint`: aligned samples, feature blocks are disjoint across parties
- `vertical-overlap`: aligned samples, feature blocks can overlap across parties
- `*-5-clients.yaml`: same idea with 5 parties and custom feature splits

Important semantic note:
These vertical modes are not horizontal IID/non-IID variants. They are feature-partition variants:

- `vertical-disjoint` = no shared columns
- `vertical-overlap` = some shared columns are allowed

### Method configs

Method files:

- `config/fl-config-svm.yaml`
- `config/fl-config-svm-dp.yaml`
- `config/fl-config-vfl.yaml`

Current active method usage in this repository:

- `fl-config-svm.yaml` uses `fluke.algorithms.scaffold.SCAFFOLD`
- `fl-config-svm-dp.yaml` uses `fluke.algorithms.dpscaffold.DPSCAFFOLD`
- `fl-config-vfl.yaml` uses `fluke.algorithms.vertical.VerticalFL`

Important note:
Although some comments in older conversations or command labels may mention "FedAvg baseline", the current YAML files in `config/` are configured for `SCAFFOLD` and `DPSCAFFOLD`, not plain `FedAVG` and `DPFedAVG`.

## Models used

Currently used by the active configs:

- `Medical_SVM`
    - file: `fluke_package/fluke/nets.py`
    - current horizontal baseline model
    - linear tabular classifier
    - paired with `MultiMarginLoss`

- `Medical_VFL`
    - file: `fluke_package/fluke/nets.py`
    - split-model factory for true vertical FL
    - used with `VerticalFL`
    - creates per-party encoders and a server-side head

Available but not currently used in `commands.txt`:

- `Medical_ResMLP`
    - file: `fluke_package/fluke/nets.py`
    - stronger lightweight residual MLP for tabular data
    - intended to be paired with `CrossEntropyLoss`

## Aggregation / coordination methods used

### SCAFFOLD

Used in:

- horizontal IID baseline runs
- horizontal non-IID baseline runs

It was used in both IID and non-IID scenarios to make the comparison as valid as possible.

Config file:

- `config/fl-config-svm.yaml`

Why it matters:

- SCAFFOLD is a control-variate-based alternative to FedAvg
- it reduces client drift, especially under heterogeneity
- it is the current horizontal non-private baseline in this repo

### DPSCAFFOLD

Used in:

- private horizontal IID runs
- private horizontal non-IID runs

Config file:

- `config/fl-config-svm-dp.yaml`

Why it matters:

- same SCAFFOLD-style optimization logic
- adds differential privacy through Opacus on the client side
- key DP parameters are `target_epsilon`, `target_delta`, and `max_grad_norm`

Important DP note:

- `target_epsilon` is the privacy budget target
- `target_delta` is the small failure probability term in `(epsilon, delta)`-DP
- the actual noise level is derived from those values by Opacus

### VerticalFL

Used in:

- vertical disjoint runs
- vertical overlap runs
- 5-client vertical variants

Config file:

- `config/fl-config-vfl.yaml`

Why it matters:

- this is a true VFL coordinator, not masked-feature horizontal FL
- it trains party-specific encoders on aligned samples
- the server combines embeddings with a server-side head
- it exchanges embeddings and embedding gradients instead of averaging full client models

How aggregation works in VerticalFL:

- there is no FedAvg-style model averaging step
- each party computes a local embedding from its own feature subset
- those embeddings are sent to the server
- the server concatenates the embeddings and applies the server-side head to produce logits
- the loss is computed at the server level
- backpropagation produces gradients with respect to each party embedding
- the server sends each embedding gradient back to the corresponding party
- each party updates only its own encoder with its own optimizer
- the server updates only the server head with its own optimizer

So in VerticalFL, aggregation is really representation fusion at the server, followed by split backpropagation, not parameter averaging across clients.

## Metrics logged and computed

### Standard predictive metrics

Defined in `fluke_package/fluke/evaluation.py`.

Logged metrics include:

- `accuracy`
- `macro_precision`
- `macro_recall`
- `macro_f1`
- `micro_precision`
- `micro_recall`
- `micro_f1`
- `loss`

### Fairness metrics

Fairness is computed only if a binary sensitive attribute is available.

Sensitive-column detection in the medical loader:

- explicit `sensitive_col` if provided
- otherwise first match among:
    - `Sex`
    - `sex`
    - `gender`
    - `Gender`

If the sensitive attribute is binary, the evaluator additionally computes:

- `statistical_parity_difference`
- `equal_opportunity_difference`
- `tp`, `fp`, `tn`, `fn`
- `group_0_tp`, `group_0_fp`, `group_0_tn`, `group_0_fn`
- `group_1_tp`, `group_1_fp`, `group_1_tn`, `group_1_fn`
- `group_0_positive_rate`, `group_1_positive_rate`
- `group_0_true_positive_rate`, `group_1_true_positive_rate`

Formulas:

- `SPD = positive_rate(group_1) - positive_rate(group_0)`
- `EOD = TPR(group_1) - TPR(group_0)`

### Aggregated fairness summaries used in plots

`runs/generate_fl_comparisons.py` aggregates client-level fairness metrics from `locals_metrics.csv` and produces round-wise and final summaries:

- `fairness_final_spd_mean`
- `fairness_final_spd_std`
- `fairness_final_spd_gap`
- `fairness_final_eod_mean`
- `fairness_final_eod_std`
- `fairness_final_eod_gap`

The single fairness value used in final bar plots is the last-round mean across clients.

### Cost and runtime metrics

Logged and later plotted:

- communication costs from `comm_costs.csv`
- runtime from `run_metrics.csv`

Communication cost is based on message sizes exchanged through the communication channel.

## Files generated by each run

Each run directory under `runs/<run_name>/` may contain:

- `global_metrics.csv`: server/global metrics per round
- `locals_metrics.csv`: per-client evaluation metrics per round
- `prefit_metrics.csv`: per-client pre-fit metrics if enabled
- `postfit_metrics.csv`: per-client post-fit metrics if enabled
- `comm_costs.csv`: communication cost per round
- `metrics.csv`: additional logged scalar metrics
- `local_test_metrics.csv`: client-local evaluation exports when available
- `shared_test_metrics.csv`: shared-test evaluation exports when available
- `run_metrics.csv`: run-level scalars such as runtime

## Files generated by the plotting scripts

Primary plotting entry point:

- `runs/generate_fl_comparisons.py`

Main outputs under `runs/plots/`:

- `run_summary.csv`
- `roundwise_global_metrics.csv`
- `roundwise_fairness_metrics.csv`
- `roundwise_client_metrics.csv`
- `predictive_metric_vs_round.png`
- `utility_loss_vs_round.png`
- `predictive_client_mean_vs_round.png`
- `predictive_server_vs_clients_mean_vs_round.png`
- `fairness_spd_vs_round.png`
- `fairness_eod_vs_round.png`
- `final_macro_f1_bar.png`
- `final_fairness_spd_bar.png`
- `final_fairness_eod_bar.png`
- `final_fairness_spd_eod_bar.png`
- `run_time_seconds_bar.png`
- `total_comm_cost_bar.png`
- `total_comm_cost_bar_log.png`
- `dp_epsilon_vs_predictive_metric.png`
- `dp_epsilon_vs_runtime.png`

Round-wise progression outputs under `runs/plots/iid_noise_progression/`:

- one folder per setting:
    - `Horizontal-IID/`
    - `Horizontal-NonIID/`
    - `Vertical-Disjoint/`
    - `Vertical-Overlap/`
    - `Vertical-Comparison/`
- `roundwise_setting_privacy_metrics.csv`

Important note:
All plotting now goes through `runs/generate_fl_comparisons.py`.

## How to launch runs

Prerequisite:
Activate the environment where the `fluke` CLI is available.

### Horizontal IID baseline

```bash
fluke federation config/medical-data-iid.yaml config/fl-config-svm.yaml logger.log_dir=runs/medical-SVM-iid
```

### Horizontal IID private sweep

```bash
fluke federation config/medical-data-iid.yaml config/fl-config-svm-dp.yaml method.hyperparameters.client.target_epsilon=0.2 method.hyperparameters.client.max_grad_norm=1.0 logger.log_dir=runs/medical-dp-svm-iid-eps-0p2-mgn-1p0
fluke federation config/medical-data-iid.yaml config/fl-config-svm-dp.yaml method.hyperparameters.client.target_epsilon=0.5 method.hyperparameters.client.max_grad_norm=1.0 logger.log_dir=runs/medical-dp-svm-iid-eps-0p5-mgn-1p0
fluke federation config/medical-data-iid.yaml config/fl-config-svm-dp.yaml method.hyperparameters.client.target_epsilon=1.0 method.hyperparameters.client.max_grad_norm=1.0 logger.log_dir=runs/medical-dp-svm-iid-eps-1p0-mgn-1p0
fluke federation config/medical-data-iid.yaml config/fl-config-svm-dp.yaml method.hyperparameters.client.target_epsilon=5.0 method.hyperparameters.client.max_grad_norm=1.0 logger.log_dir=runs/medical-dp-svm-iid-eps-5p0-mgn-1p0
```

### Horizontal non-IID baseline

```bash
fluke federation config/medical-data-non-iid.yaml config/fl-config-svm.yaml logger.log_dir=runs/medical-SVM-non-iid
```

### Horizontal non-IID private sweep

```bash
fluke federation config/medical-data-non-iid.yaml config/fl-config-svm-dp.yaml method.hyperparameters.client.target_epsilon=0.2 method.hyperparameters.client.max_grad_norm=1.0 logger.log_dir=runs/medical-dp-svm-non-iid-eps-0p2-mgn-1p0
fluke federation config/medical-data-non-iid.yaml config/fl-config-svm-dp.yaml method.hyperparameters.client.target_epsilon=0.5 method.hyperparameters.client.max_grad_norm=1.0 logger.log_dir=runs/medical-dp-svm-non-iid-eps-0p5-mgn-1p0
fluke federation config/medical-data-non-iid.yaml config/fl-config-svm-dp.yaml method.hyperparameters.client.target_epsilon=1.0 method.hyperparameters.client.max_grad_norm=1.0 logger.log_dir=runs/medical-dp-svm-non-iid-eps-1p0-mgn-1p0
fluke federation config/medical-data-non-iid.yaml config/fl-config-svm-dp.yaml method.hyperparameters.client.target_epsilon=5.0 method.hyperparameters.client.max_grad_norm=1.0 logger.log_dir=runs/medical-dp-svm-non-iid-eps-5p0-mgn-1p0
```

### Vertical disjoint

```bash
fluke federation config/medical-data-vertical-disjoint.yaml config/fl-config-vfl.yaml logger.log_dir=runs/medical-vfl-vertical-disjoint
```

### Vertical overlap

```bash
fluke federation config/medical-data-vertical-overlap.yaml config/fl-config-vfl.yaml logger.log_dir=runs/medical-vfl-vertical-overlap
```

### Vertical disjoint with 5 clients

```bash
fluke federation config/medical-data-vertical-disjoint-5-clients.yaml config/fl-config-vfl.yaml logger.log_dir=runs/medical-vfl-vertical-disjoint-5-clients
```

### Vertical overlap with 5 clients

```bash
fluke federation config/medical-data-vertical-overlap-5-clients.yaml config/fl-config-vfl.yaml logger.log_dir=runs/medical-vfl-vertical-overlap-5-clients
```

## How to launch the plotting

Recommended unified plotting entry point:

```bash
python runs/generate_fl_comparisons.py --dataset medical
```

Optional progression controls:

```bash
python runs/generate_fl_comparisons.py --dataset medical --epsilon-levels 0.2 0.5 1.0 5.0
python runs/generate_fl_comparisons.py --dataset medical --progression-out-dir runs/plots/iid_noise_progression
python runs/generate_fl_comparisons.py --dataset medical --skip-progression
```

## Important practical notes

- For CLI overrides on the algorithm config, use `method.hyperparameters...`, not `hyperparameters...`.
- For vertical runs, use `VerticalFL` with a vertical distribution config.
- Vertical runs require `eligible_perc: 1.0` so that all parties stay aligned.
- `Medical_SVM` should be paired with `MultiMarginLoss`.
- `Medical_ResMLP` should be paired with `CrossEntropyLoss`.
- Horizontal and vertical communication costs are on very different scales. Use the log-scale communication plot for clearer comparison.
- The 5-client `vertical-overlap` config uses adjacent-party overlap
