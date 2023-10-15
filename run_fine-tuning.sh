#!/usr/bin/env bash

set -euo pipefail
set -x

poetry run python -m torch.distributed.launch --nproc_per_node=4 tools/finetune.py \
  --config-file ./configs/finetune/glip_Swin_T_O365_GoldG_flickr-ja.yaml \
  --skip-test \
  --use-tensorboard \
  --evaluate_only_best_on_test \
  MODEL.BACKBONE.FREEZE_CONV_BODY_AT 2 \
  MODEL.DYHEAD.USE_CHECKPOINT True \
  TEST.DURING_TRAINING False \
  TEST.IMS_PER_BATCH 12 \
  TEST.EVAL_TASK detection \
  DATASETS.USE_OVERRIDE_CATEGORY True \
  DATASETS.SHUFFLE_SEED 3 \
  DATASETS.USE_CAPTION_PROMPT True \
  SOLVER.MAX_EPOCH 2 \
  SOLVER.WARMUP_ITERS 100 \
  SOLVER.USE_AMP True \
  SOLVER.IMS_PER_BATCH 12 \
  SOLVER.WEIGHT_DECAY 0.05 \
  SOLVER.FIND_UNUSED_PARAMETERS False \
  SOLVER.TEST_WITH_INFERENCE False \
  SOLVER.USE_AUTOSTEP True \
  SOLVER.SEED 10 \
  SOLVER.STEP_PATIENCE 3 \
  SOLVER.CHECKPOINT_PER_EPOCH 1.0 \
  SOLVER.AUTO_TERMINATE_PATIENCE 8 \
  SOLVER.MODEL_EMA 0.0 \
  SOLVER.TUNING_HIGHLEVEL_OVERRIDE full
