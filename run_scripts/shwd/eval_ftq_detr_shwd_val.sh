#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: bash run_scripts/shwd/eval_ftq_detr_shwd_val.sh /path/to/checkpoint.pth"
  exit 1
fi

CKPT=$1
PYTHON=${PYTHON:-python}
DATA_ROOT=${DATA_ROOT:-data/SHWD_COCO}
OUT_DIR=${OUT_DIR:-runs/shwd_ftq_detr_eval_val}
mkdir -p "${OUT_DIR}"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} $PYTHON main.py \
  --eval \
  --eval_split val \
  --resume "${CKPT}" \
  --dataset_file coco \
  --coco_path "${DATA_ROOT}" \
  --output_dir "${OUT_DIR}" \
  --backbone resnet50 \
  --num_classes 3 \
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
