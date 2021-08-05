#!/bin/bash

batch_size=2
lr=0.001
epochs=90
save_test=True

save_dir="../results/SegNetPlusClass/bs${batch_size}lr${lr}e${epochs}"

python ../src/trainSegNetPlusClass.py \
    --save-dir $save_dir \
    --batchSize $batch_size \
    --lr $lr \
    --epochs $epochs \
    --saveTest $save_test \
    |& tee -a "${save_dir}/train.log"