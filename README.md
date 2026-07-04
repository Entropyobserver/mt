# LoRA Fine-Tuning for English-Norwegian Petroleum Translation

This repository contains experiments for adapting
`facebook/nllb-200-distilled-600M` to English-to-Norwegian petroleum-domain
translation using LoRA, plus a controlled comparison against full
fine-tuning.

The project uses Norwegian Petroleum Directorate (NPD) parallel data and
focuses on low-resource domain adaptation, data scaling, LoRA hyperparameter
sensitivity, and LoRA-vs-full-fine-tuning trade-offs.

## Repository Layout

```text
mt_oil_no/
|-- config.yaml                         # shared model, data, and training config
|-- Environment.yml                     # conda environment specification
|-- data/
|   |-- final_splits_npd/               # JSON/TSV train/val/test splits
|   |-- final_splits_npd_bokmal/        # JSONL Bokmal-filtered splits
|   |-- processed/                      # processing notebooks and reports
|   `-- source/                         # source NPD TMX and converted data
|-- experiments/en_no_expert/
|   |-- a_data_scaling.py               # Exp A: data scaling
|   |-- a_exp1.py                       # Slurm array wrapper for Exp A
|   |-- b_gridsearch.py                 # Exp B: LoRA grid search
|   |-- b_exp2.py                       # Slurm array wrapper for Exp B
|   |-- c_optuna_stage1.py              # Exp C1: Optuna coarse search
|   |-- c_optuna_stage2.py              # Exp C2: validate top configs
|   |-- d_final_eval.py                 # Exp D: final LoRA evaluation
|   |-- e_lora_vs_ft.py                 # Exp E: LoRA vs full fine-tuning
|   `-- *.sh                            # UPPMAX/Pelle Slurm launch scripts
|-- scripts/
|   |-- data/                           # dataset and DataManager utilities
|   |-- evaluation/                     # BLEU, chrF, optional COMET evaluation
|   `-- model/                          # BaseTrainer, LoRATrainer, FullTrainer
|-- analysis/                           # analysis utilities
|-- test/                               # walkthrough notebooks
`-- outputs/                            # generated locally; not tracked
```

## Data

The processed data is included in this repository.

Main configured splits:

```text
data/final_splits_npd/train.json
data/final_splits_npd/val.json
data/final_splits_npd/test.json
```

Additional Bokmal-filtered JSONL splits are available at:

```text
data/final_splits_npd_bokmal/train.jsonl
data/final_splits_npd_bokmal/val.jsonl
data/final_splits_npd_bokmal/test.jsonl
```

`TranslationDataset` supports both `.json` and `.jsonl` via
`TranslationDataset.from_file()`.

The source corpus is derived from the NPD/ELRC petroleum translation data.
Please check the original data provider terms before redistributing or using
the data outside research/course contexts.

## Setup

The recommended environment is provided in `Environment.yml`.

```bash
conda env create -f Environment.yml
conda activate mt26
```

For a minimal manual setup:

```bash
conda create -n mt26 python=3.10
conda activate mt26
pip install torch transformers peft datasets evaluate pandas pyyaml optuna sacrebleu
```

For COMET evaluation in the final experiment:

```bash
pip install unbabel-comet
```

## Configuration

Shared settings live in `config.yaml`.

Important defaults:

```yaml
model:
  pretrained: facebook/nllb-200-distilled-600M
  src_lang: eng_Latn
  tgt_lang: nob_Latn
  max_length: 128

training:
  epochs: 3
  batch_size: 4
  grad_accumulation: 4
  lr: 5.0e-4

generation:
  max_length: 128
  num_beams: 5
```

## Running Experiments

Run from the repository root.

### A. Data Scaling

Single-process run:

```bash
python experiments/en_no_expert/a_data_scaling.py
```

UPPMAX Slurm array:

```bash
sbatch experiments/en_no_expert/a_exp1.sh
```

This tests multiple training sizes and seeds. Current config uses:

```text
train_sizes = [100, 500, 1000, 2000, 4000, 6000, 8000, 10000, full]
seeds = [42, 123, 456]
```

### B. LoRA Grid Search

Single-process run:

```bash
python experiments/en_no_expert/b_gridsearch.py
```

UPPMAX Slurm array:

```bash
sbatch experiments/en_no_expert/b_exp2.sh
```

The grid searches LoRA rank, alpha, and dropout using the training size
specified in `config.yaml`.

### C. Optuna Hyperparameter Search

```bash
python experiments/en_no_expert/c_optuna_stage1.py
python experiments/en_no_expert/c_optuna_stage2.py
```

Or on UPPMAX:

```bash
sbatch experiments/en_no_expert/c_optuna_en_no.sh
```

Stage 1 performs a coarse search. Stage 2 validates top configurations and
writes:

```text
outputs/exp3_optuna_stage2/best_config.json
```

### D. Final LoRA Evaluation

```bash
python experiments/en_no_expert/d_final_eval.py
```

Or:

```bash
sbatch experiments/en_no_expert/d_final_eval_en_no.sh
```

This loads the best Stage 2 LoRA configuration when available and evaluates
the final model with BLEU, chrF, and optional COMET.

### E. LoRA vs Full Fine-Tuning

Run LoRA and full fine-tuning as separate jobs:

```bash
python experiments/en_no_expert/e_lora_vs_ft.py --method lora
python experiments/en_no_expert/e_lora_vs_ft.py --method ft
```

Or on UPPMAX:

```bash
sbatch experiments/en_no_expert/e_lora.sh
sbatch experiments/en_no_expert/e_ft.sh
```

The comparison controls the backbone, data subsets, seeds, epochs, batch
settings, validation/test generation, and test set. It uses method-appropriate
learning rates:

```text
LoRA:    5e-4, fp16
Full FT: 5e-5, fp32
```

Full fine-tuning is run in fp32 for numerical stability; LoRA uses fp16 for
efficiency.

## Reproducibility Notes

- The Slurm scripts are configured for the UPPMAX Pelle cluster and the
  project path used during this course project.
- Hugging Face caches are placed under the configured project storage path in
  the Slurm scripts.
- Generated outputs, logs, checkpoints, model weights, and caches are excluded
  from Git.
- Validation and test generation both force the NLLB target language token to
  `nob_Latn`.

## Outputs

Experiment outputs are written under:

```text
outputs/
```

This directory is ignored by Git because it may contain large checkpoints and
generated result files. Keep important summary tables separately if they need
to be archived.

## License

Code in this repository is licensed under the Apache License 2.0. See
`LICENSE`.

The NLLB model is not redistributed here. See the model card for
`facebook/nllb-200-distilled-600M` for its license and use restrictions.
