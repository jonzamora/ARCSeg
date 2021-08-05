#!/bin/bash

model="unet"

batch_size=3
lr=1e-3
epochs=50

dataset="synapse"
data_dir="/home/jonzamora/Desktop/arclab/ARCSeg/src/data/datasets/${dataset}"
json_path="/home/jonzamora/Desktop/arclab/ARCSeg/src/data/classes/synapseSegClasses.json"

save_dir="../results/SegNet/${dataset}/bs_${batch_size}_lr${lr}_e${epochs}"

mkdir -p $save_dir

python ../src/trainSegNet.py \
    --model $model \
    --batchSize $batch_size \
    --lr $lr \
    --epochs $epochs \
    --dataset $dataset \
    --data_dir $data_dir \
    --json_path $json_path \
    --save_dir $save_dir \
    |& tee -a "${save_dir}/debug.log"
