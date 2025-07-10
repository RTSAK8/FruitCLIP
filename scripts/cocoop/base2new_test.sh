#!/bin/bash

# custom config
DATA=data
TRAINER=CoCoOp
# TRAINER=CoOp

DATASET=$1
SEED=$2

CFG=$3
# CFG=vit_b16_ctxv1  # uncomment this when TRAINER=CoOp
SHOTS=64
LOADEP=50
SUB=new


COMMON_DIR=${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
MODEL_DIR=output/CoCoOp/train_base/${COMMON_DIR}
DIR=output/CoCoOp/test_${SUB}/${COMMON_DIR}
if [ -d "$DIR" ]; then
    echo "Oops! The results exist at ${DIR} (so skip this job)"
else
    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --model-dir ${MODEL_DIR} \
    --load-epoch ${LOADEP} \
    --eval-only \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES ${SUB}
fi