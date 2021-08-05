#!/bin/bash

batch_size=64
lr=0.01
epochs=2
image_size=256

save_dir="../results/ReconNet/ImageSize${image_size}/bs${batch_size}lr${lr}e${epochs}"

python ../src/trainReconNet.py \
    --save-dir $save_dir \
    --batchSize $batch_size \
    --lr $lr \
    --epochs $epochs \
    --imageSize $image_size \
    |& tee -a "${save_dir}/train.log"