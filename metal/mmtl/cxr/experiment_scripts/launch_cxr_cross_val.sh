#!/bin/bash

NUM_RUNS=$1
RUN_NAME=$2

SEEDS=(000 1701 123 7 42 1000000)
LRS=(0.001 0.0001 0.00001 0.000001 0.0001)
L2S=(0.00001 0.00001 0.00001 0.00001 0.0001)
for run in $(seq 1 1 $NUM_RUNS)
do
CMD="python -W ignore launch_cxr.py --tasks CXR8-DRAIN_ALL --batch_size 16 --n_epochs 3 --lr ${LRS[$run]} --l2 ${L2S[$run]} --lr_scheduler linear --pretrained 1 --drop_rate 0.2 --warmup_steps 0 --warmup_unit epochs --min_lr 1e-6 --res 224 --test_split test --progress_bar 0 --seed 1701 --num_workers 6 --log_every 1.0 --checkpoint_every 1.0 --slice_model 1 --use_slices 1 --run_name sps_drain_slice_neg_test_fine_tune --model_weights /home/jdunnmon/Research/repos/metal/logs/2019_04_29/slice_model_debug_23_59_42/best_model.pth --fine_tune CXR8-DRAIN_PNEUMOTHORAX_slice:chest_drain_cnn_neg:pred,CXR8-DRAIN_PNEUMOTHORAX_slice:chest_drain_cnn_neg:BASE --freeze heads --checkpoint_metric CXR8-DRAIN_PNEUMOTHORAX_slice:chest_drain_cnn_neg:pred/CXR8-DRAIN_valid/CXR8-DRAIN_PNEUMOTHORAX_slice:chest_drain_cnn_neg:pred/roc-auc --seed ${SEEDS[$run]} --checkpoint_metric_mode max"

echo "Launching run $run with command:"
echo $CMD
sleep 10
eval $CMD
done

