#!/usr/bin/env bash

# Data Parameters
DATASET='shapenet'
INPTS=1024
NGTPTS=2048
NPTS=$((1*(2048)))
NSAUCE=2048

# Model Parameters
NET='AtlasNet'
CODE_NFTS=1024
DIST_FUN='chamfer'
NB_PRIMITIVES=16

# Training Parameters
MODE='train'
RESUME=0
OPTIM='adagrad'
LR=1e-2
EPOCHS=300
SAVE_EPOCH=5
TEST_EPOCH=$SAVE_EPOCH
BATCH_SIZE=32
NWORKERS=4


PROGRAM="main.py"

python -u $PROGRAM --epochs $EPOCHS --lr $LR --batch_size $BATCH_SIZE \
    --nworkers  $NWORKERS --NET $NET --dataset $DATASET \
    --mode $MODE --optim $OPTIM --code_nfts $CODE_NFTS --resume $RESUME --dist_fun $DIST_FUN \
    --npts $NPTS --nsauce $NSAUCE --inpts $INPTS --ngtpts $NGTPTS --nb_primitives $NB_PRIMITIVES
