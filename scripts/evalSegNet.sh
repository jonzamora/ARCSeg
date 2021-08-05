#!/bin/bash

save_test=True
checkpoint_dir="save_cholecSegNet"
checkpoint_title="checkpoint_40_segnet_augs2_batch32"
checkpoint="${checkpoint_dir}/${checkpoint_title}.tar"
save_dir="../results/${checkpoint_dir}/${checkpoint_title}"


python ../src/evaluate.py \
--save-dir $save_dir \
--saveTest $save_test \
--model $checkpoint \
|& tee -a "${save_dir}/eval.log"