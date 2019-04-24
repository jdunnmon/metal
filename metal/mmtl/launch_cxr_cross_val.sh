#!/bin/bash

NUM_RUNS=$1
RUN_NAME=$2
USE_SLICES=$3

SEEDS=(000 1701 123 7 42 1000000)
for run in $(seq 1 1 $NUM_RUNS)
do
CMD="python -W ignore launch_cxr.py --tasks CXR8-DRAIN_ALL --batch_size 16 --n_epochs 2 --lr 0.0001 --l2 0.000 --lr_scheduler linear --run_name ${RUN_NAME}_cv_run_${run} --pretrained 1 --drop_rate 0.2 --warmup_steps 0 --warmup_unit epochs --min_lr 1e-6 --res 224 --test_split test --progress_bar 0 --use_slices $USE_SLICES --seed ${SEEDS[$run]} --num_workers 8"

echo "Launching cross-validation run $run with command:"
echo $CMD
sleep 10
eval $CMD

done
