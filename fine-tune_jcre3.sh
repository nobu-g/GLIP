#!/usr/bin/env bash

set -euo pipefail
set -x

readonly GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-"32"}"
readonly DEVICES="${DEVICES:-"4"}"
readonly MAX_EPOCH="${MAX_EPOCH:-"2"}"
readonly BASE_EXPR_NAME="${BASE_EXPR_NAME:-"pretrained_roberta_flickr_ja_mixed_2e_b36"}"
EXPR_NAME="${BASE_EXPR_NAME}_jcre3_${MAX_EPOCH}e_b${GLOBAL_BATCH_SIZE}"

poetry run python -m torch.distributed.launch --nproc_per_node="${DEVICES}" tools/finetune.py \
  --config-file ./configs/finetune/jcre3.yaml \
  --skip-test \
  --use-tensorboard \
  --evaluate_only_best_on_test \
  OUTPUT_DIR "./OUTPUT/${EXPR_NAME}" \
  MODEL.WEIGHT "./OUTPUT/${BASE_EXPR_NAME}/ft_task_1/model_0032040.pth" \
  MODEL.DYHEAD.USE_CHECKPOINT True \
  MODEL.LANGUAGE_BACKBONE.USE_CHECKPOINT True \
  MODEL.BACKBONE.USE_CHECKPOINT True \
  TEST.DURING_TRAINING False \
  TEST.IMS_PER_BATCH 8 \
  TEST.EVAL_TASK detection \
  DATASETS.USE_OVERRIDE_CATEGORY True \
  DATASETS.SHUFFLE_SEED 3 \
  DATASETS.USE_CAPTION_PROMPT True \
  SOLVER.MAX_EPOCH "${MAX_EPOCH}" \
  SOLVER.WARMUP_ITERS 1000 \
  SOLVER.USE_AMP True \
  SOLVER.IMS_PER_BATCH "${GLOBAL_BATCH_SIZE}" \
  SOLVER.WEIGHT_DECAY 0.05 \
  SOLVER.FIND_UNUSED_PARAMETERS False \
  SOLVER.TEST_WITH_INFERENCE False \
  SOLVER.USE_AUTOSTEP True \
  SOLVER.SEED 10 \
  SOLVER.STEP_PATIENCE 3 \
  SOLVER.CHECKPOINT_PER_EPOCH 1.0 \
  SOLVER.AUTO_TERMINATE_PATIENCE 8 \
  SOLVER.MODEL_EMA 0.0 \
  "${@}"
