# Experiment Rerun Notes

This repository contains a rerun of the English-to-Norwegian petroleum-domain
LoRA experiments.

## Summary

The Slurm-array entry points now support explicit post-processing modes:

```bash
python experiments/en_no_expert/a_exp1.py --summarize
python experiments/en_no_expert/b_exp2.py --summarize
```

These commands collect per-job `metrics.json` files and write compact summary
artifacts under `outputs/`.

## Experiment 1: Data Scaling

Current rerun summary:

| Training size | Test BLEU mean | Test chrF mean |
|---:|---:|---:|
| 100 | 0.3728 | 62.1734 |
| 500 | 0.4838 | 70.2907 |
| 1000 | 0.5203 | 73.1143 |
| 2000 | 0.5276 | 73.7723 |
| 4000 | 0.5701 | 76.1616 |
| 6000 | 0.5850 | 77.0723 |
| 8000 | 0.5994 | 77.9257 |
| 10000 | 0.6085 | 78.5826 |
| 13935 | 0.6184 | 79.2503 |

Best run:

```text
size=13935, seed=123, test_bleu=0.6198
```

## Experiment 2: LoRA Grid Search

Current rerun top configurations by validation BLEU:

| Rank | r | alpha | dropout | Validation BLEU |
|---:|---:|---:|---:|---:|
| 1 | 16 | 64 | 0.0 | 0.6181 |
| 2 | 32 | 64 | 0.0 | 0.6177 |
| 3 | 8 | 64 | 0.0 | 0.6169 |
| 4 | 16 | 64 | 0.1 | 0.6160 |
| 5 | 32 | 64 | 0.1 | 0.6157 |

Recommended configuration from the rerun:

```text
r=16, alpha=64, dropout=0.0
```

## Notes on Original vs Rerun Results

The rerun reproduces the main experimental trends, but not the exact original
numeric scores. Scores in the rerun are consistently higher than the earlier
records. The most likely reason is a difference in generation or evaluation
settings; the current code explicitly forces the target language token
(`nob_Latn`) during generation, which is the appropriate setting for NLLB-style
multilingual translation models.

For reporting, use the rerun results as the final corrected results and describe
the earlier numbers as historical records rather than exact reproduced values.

