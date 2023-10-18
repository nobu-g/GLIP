#!/usr/bin/env bash

set -euo pipefail
set -x

python -m torch.distributed.launch --nproc_per_node=2 tools/train_net.py \
  --config-file ./configs/pretrain/glip_Swin_T_O365_GoldG.yaml \
  --skip-test \
  MODEL.WEIGHT ./MODEL/glip_tiny_model_o365_goldg.pth \
  DATASETS.TRAIN '("coco_grounding_train", )' \
  MODEL.BACKBONE.FREEZE_CONV_BODY_AT -1 \
  SOLVER.IMS_PER_BATCH 4 \
  SOLVER.USE_AMP True \
  SOLVER.MAX_EPOCH 0 \
  SOLVER.MAX_ITER 10 \
  TEST.DURING_TRAINING False \
  TEST.IMS_PER_BATCH 4 \
  TEST.EVAL_TASK detection \
  SOLVER.FIND_UNUSED_PARAMETERS False \
  SOLVER.BASE_LR 0.00001 \
  SOLVER.LANG_LR 0.00001 \
  SOLVER.STEPS \(0.67,0.89\) \
  DATASETS.DISABLE_SHUFFLE True \
  MODEL.DYHEAD.SCORE_AGG "MEAN"
