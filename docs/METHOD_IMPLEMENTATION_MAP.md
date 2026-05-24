# Method-to-code mapping

This document maps the final FTQ-DETR components described in the manuscript to their implementation locations.

## FGBC: Fine-Grained Boundary Compensation

Paper section: `3.2 Fine-Grained Boundary Compensation Module`

Code:

```text
models/deformable_detr.py
class FineGrainedBoundaryCompensation
```

Main flags used by the final-model scripts:

```text
--use_fgbc
--fgbc_levels
--fgbc_alpha
--fgbc_kernel
--fgbc_warmup_epochs
```

## TDDFD: Task-Decoupled Dual-Field Deformable Decoder

Paper section: `3.3 Task-Decoupled Dual-Field Deformable Decoder`

Code:

```text
models/deformable_transformer.py
class DeformableTransformerDecoderLayer
class DeformableTransformerDecoder
```

Main flags used by the final-model scripts:

```text
--use_tddfd
--tddfd_start_layer
```

## QACR: Quality-Aligned Classification and Ranking

Paper section: `3.4 Quality-Aligned Classification and Ranking Framework`

Code:

```text
models/deformable_detr.py
SetCriterion.loss_labels
SetCriterion.loss_quality
SetCriterion.loss_quality_rank
PostProcess.forward
```

Main flags used by the final-model scripts:

```text
--use_qa_cls
--qa_iou_power
--use_quality_branch
--quality_loss_coef
--use_quality_ranking
--quality_rank_loss_coef
--quality_rank_margin
--quality_rank_topk
--use_quality_rescore
--quality_score_alpha
--quality_score_beta
```
