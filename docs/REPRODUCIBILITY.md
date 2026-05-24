# Reproducibility guide for FTQ-DETR final model

This document describes how to reproduce the final FTQ-DETR model experiments reported in the manuscript.

## 1. SHEL5K primary experiment

Prepare the processed SHEL5K COCO-format dataset under `data/SHEL5K_COCO6/` with the 3500/1000/500 train/val/test split.

Check the dataset:

```bash
python scripts/check_coco_structure.py --coco_path data/SHEL5K_COCO6 --splits train val test
```

Train the final FTQ-DETR model:

```bash
bash run_scripts/shel5k/train_ftq_detr_shel5k.sh
```

Evaluate the selected checkpoint on the reserved test set:

```bash
bash run_scripts/shel5k/eval_ftq_detr_shel5k_test.sh runs/shel5k_ftq_detr_r50_b2_e36/checkpoint0035.pth
```

## 2. SHWD supplementary validation

Prepare the converted SHWD COCO-format dataset under `data/SHWD_COCO/` with the 6065/1516 train/val split.

Check the dataset:

```bash
python scripts/check_coco_structure.py --coco_path data/SHWD_COCO --splits train val
```

Train and evaluate the final model:

```bash
bash run_scripts/shwd/train_ftq_detr_shwd.sh
bash run_scripts/shwd/eval_ftq_detr_shwd_val.sh runs/shwd_ftq_detr_r50_b2_e36/checkpoint0035.pth
```

## 3. Notes on category IDs and `--num_classes`

This code follows the DETR/H-Deformable-DETR convention in which target labels are used as integer class indices. If the COCO category IDs are one-based, the classifier output dimension should be `max_category_id + 1`.

For the processed SHEL5K setting used in the manuscript, category IDs are 1-6 after removing the rare `person` category, so `--num_classes 7` is used.

For SHWD, the provided scripts assume one-based category IDs 1-2, so `--num_classes 3` is used. If SHWD labels are remapped to zero-based IDs, change the scripts to `--num_classes 2`.

## 4. What to place in the public repository

The GitHub repository should be the single entry point for reproducibility. It should include the source code, final-model scripts, processed COCO annotations, split files, dataset acquisition or reconstruction instructions, and result-verification files when available.
