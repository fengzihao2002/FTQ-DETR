# FTQ-DETR: Feature-Compensated, Task-Decoupled, and Quality-Aligned End-to-End Safety Helmet Detection

This repository provides the source code and dataset preparation materials for the final FTQ-DETR model used in the manuscript:

**FTQ-DETR: Feature-Compensated, Task-Decoupled, and Quality-Aligned End-to-End Safety Helmet Detection**

The repository is intended for reproducible review. It contains the final-model implementation, training and evaluation scripts, dataset conversion scripts, COCO-format annotation files, split files, and documentation required to reproduce the reported SHEL5K and SHWD experiments.

FTQ-DETR is implemented on top of H-Deformable-DETR and includes three final-model components:

- **FGBC**: Fine-Grained Boundary Compensation before Transformer encoding;
- **TDDFD**: Task-Decoupled Dual-Field Deformable Decoder with delayed task decoupling;
- **QACR**: Quality-Aligned Classification and Ranking with quality-aware labels, explicit quality prediction, local ranking, and score recalibration.

This public package is organized for the **final FTQ-DETR model only**. It does not include ablation-only scripts.

---

## 1. Repository organization

```text
FTQ-DETR/
├── benchmark.py
├── main.py                         # training and evaluation entry point
├── engine.py                       # training and evaluation loops
├── requirements.txt
├── README.md
├── LICENSE
├── models/
│   ├── deformable_detr.py          # FTQ-DETR detector, FGBC, QACR losses, and post-processing
│   ├── deformable_transformer.py   # TDDFD decoder implementation
│   ├── matcher.py
│   ├── backbone.py
│   └── ops/                        # multi-scale deformable attention CUDA operators
├── datasets/
│   ├── coco.py                     # COCO-format dataset loader; supports train/val/test splits
│   ├── coco_eval.py
│   └── transforms.py
├── util/
├── mmcv_custom/
├── tools/
├── run_scripts/
│   ├── shel5k/
│   │   ├── train_ftq_detr_shel5k.sh
│   │   └── eval_ftq_detr_shel5k_test.sh
│   └── shwd/
│       ├── train_ftq_detr_shwd.sh
│       └── eval_ftq_detr_shwd_val.sh
├── scripts/
│   ├── check_coco_structure.py
│   ├── convert_shel5k_voc_to_coco6.py
│   └── convert_shwd_voc_to_coco.py
├── docs/
│   ├── METHOD_IMPLEMENTATION_MAP.md
│   └── REPRODUCIBILITY.md
└── data/
    ├── DATASET_LAYOUT.md
    ├── SHEL5K_COCO6/
    │   ├── README.md
    │   ├── annotations/
    │   │   ├── instances_train2017.json
    │   │   ├── instances_val2017.json
    │   │   └── instances_test2017.json
    │   └── splits/
    │       ├── train.txt
    │       ├── val.txt
    │       └── test.txt
    └── SHWD_COCO/
        ├── README.md
        ├── annotations/
        │   ├── instances_train2017.json
        │   └── instances_val2017.json
        └── splits/
            ├── train.txt
            └── val.txt
```

The `data/` directory in the repository provides annotation and split files. The complete processed image folders can be downloaded from the Release page of this repository and placed under the corresponding dataset directories.

---

## 2. Environment

The experiments in the manuscript were conducted under the following environment:

```text
Operating system: Linux
GPU: NVIDIA GeForce RTX 3090 24GB
Python: 3.9.25
PyTorch: 1.13.1
CUDA: 11.7
```

A compatible environment can be created as follows:

```bash
conda create -n ftq_detr python=3.9 -y
conda activate ftq_detr

pip install torch==1.13.1 torchvision==0.14.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
```

If your local CUDA/PyTorch version differs, install the PyTorch build that matches your CUDA runtime.

---

## 3. Compile multi-scale deformable attention operators

Before training or evaluation, compile the CUDA operators used by multi-scale deformable attention:

```bash
cd models/ops
bash make.sh
cd ../..
```

A quick check can be performed by running:

```bash
python models/ops/test.py
```

If compilation fails, check that `CUDA_HOME`, `nvcc`, PyTorch, and the local CUDA toolkit are correctly configured.

---

## 4. Dataset preparation

This code expects COCO-format datasets. The processed annotations and split files are included under `data/`. Complete processed image folders should be placed in the same dataset directories.

### 4.1 SHEL5K

SHEL5K is used as the primary dataset. The original dataset contains seven categories, but the extremely rare `person` category was removed during conversion. The retained six semantic categories are:

```text
1 helmet
2 head_with_helmet
3 head
4 person_with_helmet
5 face
6 person_no_helmet
```

The converted annotation keeps one-based category IDs from 1 to 6. Therefore, the detector uses:

```bash
--num_classes 7
```

Class ID 0 is unused; this setting is required by the DETR/H-Deformable-DETR indexing convention when COCO category IDs start from 1.

The split protocol is:

```text
train: 3500 images
val:   1000 images
test:   500 images
```

Expected directory layout:

```text
data/SHEL5K_COCO6/
├── train2017/
├── val2017/
├── test2017/
├── annotations/
│   ├── instances_train2017.json
│   ├── instances_val2017.json
│   └── instances_test2017.json
└── splits/
    ├── train.txt
    ├── val.txt
    └── test.txt
```

If starting from the original VOC-style SHEL5K release containing `Annotations/` and `Images/`, convert it with:

```bash
python scripts/convert_shel5k_voc_to_coco6.py \
  --src "/path/to/Safety Helmet Wearing Dataset" \
  --out data/SHEL5K_COCO6 \
  --seed 42 \
  --overwrite
```

The script removes the `person` category, generates the 3500/1000/500 split, copies images, and writes the COCO-format annotation files.

### 4.2 SHWD

SHWD is used as supplementary validation under its two-class annotation protocol:

```text
1 person
2 hat
```

The split protocol is:

```text
train: 6065 images
val:   1516 images
```

Expected directory layout:

```text
data/SHWD_COCO/
├── train2017/
├── val2017/
├── annotations/
│   ├── instances_train2017.json
│   └── instances_val2017.json
└── splits/
    ├── train.txt
    └── val.txt
```

The provided SHWD scripts assume one-based category IDs, namely 1 and 2. Therefore, they use:

```bash
--num_classes 3
```

If starting from the original VOC-style SHWD release containing `Annotations/`, `ImageSets/`, and `JPEGImages/`, convert it with:

```bash
python scripts/convert_shwd_voc_to_coco.py \
  --src "/path/to/VOC2028" \
  --out data/SHWD_COCO \
  --seed 42 \
  --overwrite \
  --force_random_split
```

The conversion script generates the 6065/1516 split and ignores non-target labels that are not part of the two-class SHWD setting.

---

## 5. Dataset integrity check

After placing the processed datasets under `data/`, verify the COCO structure:

```bash
python scripts/check_coco_structure.py --coco_path data/SHEL5K_COCO6 --splits train val test
python scripts/check_coco_structure.py --coco_path data/SHWD_COCO --splits train val
```

The checker reports image counts, annotation counts, category IDs, category names, and missing image files. The expected image counts are:

```text
SHEL5K: train=3500, val=1000, test=500
SHWD:   train=6065, val=1516
```

---

## 6. Train the final FTQ-DETR model on SHEL5K

Run:

```bash
bash run_scripts/shel5k/train_ftq_detr_shel5k.sh
```

The script uses the following final-model training command:

```bash
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python main.py \
  --dataset_file coco \
  --coco_path data/SHEL5K_COCO6 \
  --output_dir runs/shel5k_ftq_detr_r50_b2_e36 \
  --backbone resnet50 \
  --num_classes 7 \
  --epochs 36 \
  --lr 1e-4 \
  --lr_backbone 1e-5 \
  --weight_decay 1e-4 \
  --lr_drop 30 \
  --batch_size 2 \
  --num_workers 2 \
  --device cuda \
  --seed 42 \
  --num_feature_levels 4 \
  --with_box_refine \
  --two_stage \
  --num_queries_one2one 300 \
  --num_queries_one2many 300 \
  --k_one2many 3 \
  --lambda_one2many 1.0 \
  --use_fp16 \
  --use_fgbc \
  --fgbc_levels 0,1 \
  --fgbc_alpha 0.10 \
  --fgbc_kernel 3 \
  --fgbc_warmup_epochs 2 \
  --use_tddfd \
  --tddfd_start_layer 2 \
  --use_qa_cls \
  --qa_iou_power 1.0 \
  --use_quality_branch \
  --quality_loss_coef 1.0 \
  --use_quality_ranking \
  --quality_rank_loss_coef 0.05 \
  --quality_rank_margin 0.1 \
  --quality_rank_topk 3 \
  --use_quality_rescore \
  --quality_score_alpha 1.0 \
  --quality_score_beta 0.5
```

---

## 7. Evaluate the final FTQ-DETR model on SHEL5K

Evaluate a trained checkpoint on the reserved SHEL5K test split:

```bash
bash run_scripts/shel5k/eval_ftq_detr_shel5k_test.sh runs/shel5k_ftq_detr_r50_b2_e36/checkpoint0035.pth
```

The script runs:

```bash
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python main.py \
  --eval \
  --eval_split test \
  --resume runs/shel5k_ftq_detr_r50_b2_e36/checkpoint0035.pth \
  --dataset_file coco \
  --coco_path data/SHEL5K_COCO6 \
  --output_dir runs/shel5k_ftq_detr_eval_test \
  --backbone resnet50 \
  --num_classes 7 \
  --batch_size 2 \
  --num_workers 2 \
  --device cuda \
  --seed 42 \
  --num_feature_levels 4 \
  --with_box_refine \
  --two_stage \
  --num_queries_one2one 300 \
  --num_queries_one2many 300 \
  --k_one2many 3 \
  --lambda_one2many 1.0 \
  --use_fp16 \
  --use_fgbc \
  --fgbc_levels 0,1 \
  --fgbc_alpha 0.10 \
  --fgbc_kernel 3 \
  --fgbc_warmup_epochs 2 \
  --use_tddfd \
  --tddfd_start_layer 2 \
  --use_qa_cls \
  --qa_iou_power 1.0 \
  --use_quality_branch \
  --quality_loss_coef 1.0 \
  --use_quality_ranking \
  --quality_rank_loss_coef 0.05 \
  --quality_rank_margin 0.1 \
  --quality_rank_topk 3 \
  --use_quality_rescore \
  --quality_score_alpha 1.0 \
  --quality_score_beta 0.5
```

The manuscript reports the following SHEL5K test-set results for the final FTQ-DETR model:

```text
mAP@[0.50:0.95] = 56.54
AP50             = 89.52
Recall           = 86.36
```

Small numerical differences may occur because of CUDA/cuDNN nondeterminism, GPU differences, or checkpoint selection.

---

## 8. Train and evaluate the final FTQ-DETR model on SHWD

Train on SHWD:

```bash
bash run_scripts/shwd/train_ftq_detr_shwd.sh
```

The script uses:

```bash
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python main.py \
  --dataset_file coco \
  --coco_path data/SHWD_COCO \
  --output_dir runs/shwd_ftq_detr_r50_b2_e36 \
  --backbone resnet50 \
  --num_classes 3 \
  --epochs 36 \
  --lr 1e-4 \
  --lr_backbone 1e-5 \
  --weight_decay 1e-4 \
  --lr_drop 30 \
  --batch_size 2 \
  --num_workers 2 \
  --device cuda \
  --seed 42 \
  --num_feature_levels 4 \
  --with_box_refine \
  --two_stage \
  --num_queries_one2one 300 \
  --num_queries_one2many 300 \
  --k_one2many 3 \
  --lambda_one2many 1.0 \
  --use_fp16 \
  --use_fgbc \
  --fgbc_levels 0,1 \
  --fgbc_alpha 0.10 \
  --fgbc_kernel 3 \
  --fgbc_warmup_epochs 2 \
  --use_tddfd \
  --tddfd_start_layer 2 \
  --use_qa_cls \
  --qa_iou_power 1.0 \
  --use_quality_branch \
  --quality_loss_coef 1.0 \
  --use_quality_ranking \
  --quality_rank_loss_coef 0.05 \
  --quality_rank_margin 0.1 \
  --quality_rank_topk 3 \
  --use_quality_rescore \
  --quality_score_alpha 1.0 \
  --quality_score_beta 0.5
```

Evaluate on SHWD validation split:

```bash
bash run_scripts/shwd/eval_ftq_detr_shwd_val.sh runs/shwd_ftq_detr_r50_b2_e36/checkpoint0035.pth
```

---

## 9. Method-to-code correspondence

### FGBC

Implemented in:

```text
models/deformable_detr.py
FineGrainedBoundaryCompensation
```

Final-model flags:

```bash
--use_fgbc --fgbc_levels 0,1 --fgbc_alpha 0.10 --fgbc_kernel 3 --fgbc_warmup_epochs 2
```

### TDDFD

Implemented in:

```text
models/deformable_transformer.py
DeformableTransformerDecoderLayer
DeformableTransformerDecoder
```

Final-model flags:

```bash
--use_tddfd --tddfd_start_layer 2
```

`--tddfd_start_layer 2` means that decoder layers 0 and 1 remain shared, while layers from index 2 onward use task-decoupled classification and localization fields.

### QACR

Implemented in:

```text
models/deformable_detr.py
SetCriterion.loss_labels
SetCriterion.loss_quality
SetCriterion.loss_quality_rank
PostProcess.forward
```

Final-model flags:

```bash
--use_qa_cls \
--qa_iou_power 1.0 \
--use_quality_branch \
--quality_loss_coef 1.0 \
--use_quality_ranking \
--quality_rank_loss_coef 0.05 \
--quality_rank_margin 0.1 \
--quality_rank_topk 3 \
--use_quality_rescore \
--quality_score_alpha 1.0 \
--quality_score_beta 0.5
```

---

## 10. Public data access

For journal review, this GitHub repository is the single public entry point for code and data.

The repository contains:

- source code;
- final-model training and evaluation scripts;
- dataset conversion scripts;
- processed COCO-format annotation files;
- train/val/test split files;
- dataset organization and reproduction instructions.

The complete processed image datasets should be provided through the Release page of this repository:

```text
SHEL5K_COCO6.zip
SHWD_COCO.zip
```

After downloading and extracting the data archives, the expected final layout is:

```text
data/SHEL5K_COCO6/train2017/
data/SHEL5K_COCO6/val2017/
data/SHEL5K_COCO6/test2017/
data/SHWD_COCO/train2017/
data/SHWD_COCO/val2017/
```

The annotation and split files in the repository should match the files included in the released dataset archives.

---

## 11. Citation

If this repository is used, please cite the manuscript after publication.
