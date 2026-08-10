# Output Summaries

This folder keeps lightweight result summaries and figures used to check the
current analysis scripts against historical experiment outputs.

## Layout

```text
outputs/
|-- exp1_data_scaling/
|   |-- old_results/
|   |-- new_results/
|   `-- raw_runs/
|-- exp2_gridsearch/
|   |-- old_results/
|   |-- new_results/
|   `-- raw_runs/
|-- exp3_optuna_stage1/
|   |-- old_results/
|   `-- figures/
|-- exp4_optuna_stage2/
|   |-- old_results/
|   `-- figures/
`-- exp5_final_eval/
    |-- old_results/
    `-- figures/
```

- `old_results/`: historical outputs from the original saved results.
- `new_results/` or `figures/`: summaries and figures generated with the
  current scripts.
- `raw_runs/`: local job logs, per-run metrics, predictions, and model
  artifacts. These are kept for traceability and are not intended for GitHub.

## Regenerate Figures

```bash
# Experiment 1
python analysis/01_data_scaling_analysis.py --results-file outputs/exp1_data_scaling/old_results/results.json --output-file outputs/exp1_data_scaling/old_results/data_scaling_analysis.png
python analysis/01_data_scaling_analysis.py

# Experiment 2
python analysis/02_grid_search_analysis.py --results-file outputs/exp2_gridsearch/old_results/results.json --output-dir outputs/exp2_gridsearch/old_results
python analysis/02_grid_search_analysis.py

# Experiment 3
python analysis/03_parameter_optuna_analysis.py --input-file outputs/exp3_optuna_stage1/old_results/top_configs.json --output-dir outputs/exp3_optuna_stage1/old_results
python analysis/03_parameter_optuna_analysis.py --input-file outputs/exp3_optuna_stage1/results.csv --output-dir outputs/exp3_optuna_stage1/figures

# Experiment 4
python analysis/04_optuna_stage2_validation_analysis.py --results-file outputs/exp4_optuna_stage2/old_results/validation_results.csv --output-dir outputs/exp4_optuna_stage2/old_results --label old
python analysis/04_optuna_stage2_validation_analysis.py

# Experiment 5
python analysis/05_final_eval_analysis.py --input-file outputs/exp5_final_eval/old_results/final_results.json --output-dir outputs/exp5_final_eval/old_results --label old
python analysis/05_final_eval_analysis.py
```

## Notes

The current-script summaries are the main files for reporting. Historical
outputs are kept so the result-checking process is transparent.
