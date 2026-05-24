# Dataset organization for FTQ-DETR reproducibility

This repository expects SHEL5K and SHWD to be converted to COCO detection format and placed under `data/`.

## 1. SHEL5K_COCO6

SHEL5K is the primary dataset in the manuscript. The processed version removes the extremely rare `person` category and keeps six semantic categories.

### Split protocol

```text
train2017: 3500 images
val2017:   1000 images
test2017:   500 images
```

### Directory layout

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

### Category IDs

The processed annotation keeps the following one-based COCO category IDs:

```text
1 helmet
2 head_with_helmet
3 head
4 person_with_helmet
5 face
6 person_no_helmet
```

Because the labels are one-based and the maximum category ID is 6, the code should be launched with:

```bash
--num_classes 7
```

This setting is not the number of semantic categories retained in the dataset; it is the classifier output dimension required by this DETR/H-Deformable-DETR implementation when COCO category IDs start from 1. Class ID 0 is unused.

## 2. SHWD_COCO

SHWD is used as supplementary validation and preserves the original two-class annotation setting.

### Split protocol

```text
train2017: 6065 images
val2017:   1516 images
```

### Directory layout

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

If the converted COCO annotations use one-based category IDs 1 and 2, use:

```bash
--num_classes 3
```

If the converted annotations are remapped to zero-based category IDs 0 and 1, use:

```bash
--num_classes 2
```

The provided SHWD scripts use `--num_classes 3`, assuming one-based COCO IDs.

## 3. Validation before training

After placing the datasets, run:

```bash
python scripts/check_coco_structure.py --coco_path data/SHEL5K_COCO6 --splits train val test
python scripts/check_coco_structure.py --coco_path data/SHWD_COCO --splits train val
```

The checker reports split sizes, category IDs, category names, annotation counts, and missing images.

## 4. Materials for public reproducibility

For the journal resubmission, the public repository should include or link from one entry point:

1. the source code;
2. this dataset layout description;
3. processed COCO-format annotation JSON files;
4. split files or explicit reconstruction instructions;
5. original dataset download instructions if raw images cannot be redistributed;
6. final-model training and evaluation scripts under `run_scripts/`;
7. logs, checkpoints, or result CSV files used to verify the manuscript results, if available.
