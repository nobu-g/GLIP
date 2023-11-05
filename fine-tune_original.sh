#!/usr/bin/env bash

set -euo pipefail
set -x

GLOBAL_BATCH_SIZE=8
DEVICES=4
MAX_EPOCH=2
EXPR_NAME="pretrained_bert_mixed_${MAX_EPOCH}e_b${GLOBAL_BATCH_SIZE}"

python -m torch.distributed.launch --nproc_per_node="${DEVICES}" tools/finetune.py \
  --config-file ./configs/finetune/glip_Swin_T_GoldG.yaml \
  --skip-test \
  --use-tensorboard \
  --evaluate_only_best_on_test \
  OUTPUT_DIR "./OUTPUT/${EXPR_NAME}" \
  MODEL.WEIGHT ./MODEL/glip_tiny_model_o365_goldg.pth \
  MODEL.DYHEAD.SCORE_AGG "MEAN" \
  MODEL.DYHEAD.USE_CHECKPOINT True \
  MODEL.BACKBONE.FREEZE_CONV_BODY_AT -1 \
  TEST.DURING_TRAINING False \
  TEST.IMS_PER_BATCH "${GLOBAL_BATCH_SIZE}" \
  TEST.EVAL_TASK detection \
  DATASETS.SHUFFLE_SEED 3 \
  SOLVER.IMS_PER_BATCH "${GLOBAL_BATCH_SIZE}" \
  SOLVER.USE_AMP True \
  SOLVER.MAX_EPOCH "${MAX_EPOCH}" \
  SOLVER.FIND_UNUSED_PARAMETERS False \
  SOLVER.BASE_LR 0.00001 \
  SOLVER.LANG_LR 0.00001 \
  SOLVER.STEPS \(0.67,0.89\) \
  SOLVER.SEED 10 \
  SOLVER.STEP_PATIENCE 3 \
  SOLVER.CHECKPOINT_PER_EPOCH 1.0 \
  SOLVER.AUTO_TERMINATE_PATIENCE 8 \
  SOLVER.MODEL_EMA 0.0
  # SOLVER.MAX_ITER 10 \
