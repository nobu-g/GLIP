#!/usr/bin/env bash

set -euo pipefail
set -x

readonly GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-"4"}"
readonly DEVICES="${DEVICES:-"1"}"
readonly EXPR_NAME="pretrained_roberta_flickr_ja_mixed_debug"

# poetry run python -m torch.distributed.launch --nproc_per_node="${DEVICES}" tools/finetune.py \
poetry run python -m ipdb -c continue tools/finetune.py \
  --config-file ./configs/finetune/flickr_ja_mixed.yaml \
  --skip-test \
  --use-tensorboard \
  --evaluate_only_best_on_test \
  --skip_loading_text_encoder \
  OUTPUT_DIR "./OUTPUT/${EXPR_NAME}" \
  MODEL.DYHEAD.USE_CHECKPOINT True \
  MODEL.LANGUAGE_BACKBONE.USE_CHECKPOINT True \
  MODEL.BACKBONE.USE_CHECKPOINT True \
  TEST.DURING_TRAINING False \
  TEST.IMS_PER_BATCH 4 \
  TEST.EVAL_TASK detection \
  DATASETS.USE_OVERRIDE_CATEGORY True \
  DATASETS.SHUFFLE_SEED 3 \
  DATASETS.USE_CAPTION_PROMPT True \
  DATALOADER.NUM_WORKERS 0 \
  SOLVER.MAX_EPOCH 0 \
  SOLVER.MAX_ITER 10 \
  SOLVER.WARMUP_ITERS 500 \
  SOLVER.USE_AMP True \
  SOLVER.IMS_PER_BATCH "${GLOBAL_BATCH_SIZE}" \
  SOLVER.WEIGHT_DECAY 0.05 \
  SOLVER.FIND_UNUSED_PARAMETERS False \
  SOLVER.TEST_WITH_INFERENCE False \
  SOLVER.USE_AUTOSTEP True \
  SOLVER.SEED 10 \
  SOLVER.STEP_PATIENCE 3 \
  SOLVER.CHECKPOINT_PER_EPOCH 0.1 \
  SOLVER.AUTO_TERMINATE_PATIENCE 8 \
  SOLVER.MODEL_EMA 0.0 \
  "${@}"
