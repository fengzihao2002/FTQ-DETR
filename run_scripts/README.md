# Final-model run scripts

This directory contains only scripts for training and evaluating the final FTQ-DETR model reported in the manuscript.

```text
run_scripts/
├── shel5k/
│   ├── train_ftq_detr_shel5k.sh
│   └── eval_ftq_detr_shel5k_test.sh
└── shwd/
    ├── train_ftq_detr_shwd.sh
    └── eval_ftq_detr_shwd_val.sh
```

The scripts enable all final FTQ-DETR components: FGBC, TDDFD, and QACR. They are intended for reproducing the main SHEL5K experiment and the supplementary SHWD validation.
