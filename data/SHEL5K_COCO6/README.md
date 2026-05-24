# SHEL5K_COCO6

Place the processed SHEL5K six-category COCO-format dataset in this directory.

Expected structure:

```text
SHEL5K_COCO6/
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

Split sizes used in the manuscript:

```text
train2017: 3500 images
val2017:   1000 images
test2017:   500 images
```

The retained categories are:

```text
1 helmet
2 head_with_helmet
3 head
4 person_with_helmet
5 face
6 person_no_helmet
```

The original `person` category is removed. Because category IDs are kept as 1-6, use `--num_classes 7` when training or evaluating FTQ-DETR on this dataset.
