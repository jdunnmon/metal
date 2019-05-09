MODEL_TYPE=$1

N_EPOCHS=10
TASKS=CXR8-DRAIN_ALL
BATCH_SIZE=16
LR=0.0001
L2=0.000
LR_SCHEDULER=linear
PRETRAINED=1
DROP_RATE=0.2
WARMUP_STEPS=0
WARMUP_UNIT=epochs
MIN_LR=1e-6
RES=224
TEST_SPLIT=test
PROGRESS_BAR=0
SEED=1701
NUM_WORKERS=8
LOG_EVERY=1.0
CHECKPOINT_EVERY=1.0
MODEL_TYPES=(hard_param soft_param soft_param_rep, soft_param_ens)
SLICE_DICT=("'{\"CXR8-DRAIN_PNEUMOTHORAX\":[\"chest_drain_cnn_pos\"]}'" \
	    "'{\"CXR8-DRAIN_PNEUMOTHORAX\": [\"chest_drain_cnn_pos\"]}'" \
            "'{\"CXR8-DRAIN_PNEUMOTHORAX\": [\"chest_drain_cnn_neg\"]}'" \
            "'{\"CXR8-DRAIN_PNEUMOTHORAX\": [\"chest_drain_cnn_neg\"]}'" \
	    )
SLICE_POS_ONLY=("chest_drain_cnn_pos"\
	  "NONE" \
	  "chest_drain_cnn_neg" \
	  "NONE"\
	  )
EXP_NAMES=("${MODEL_TYPE}_positive_drain_slice_cnn_all_classes"\
	   "${MODEL_TYPE}_positive_drain_slice_cnn_pneumos_only"\
	   "${MODEL_TYPE}_negative_drain_slice_cnn_all_classes"\
	   "${MODEL_TYPE}_negative_drain_slice_cnn_pneumos_only"\
)

NUM_RUNS=$((${#SLICE_DICT[@]}-1))

for run in $(seq 0 1 $NUM_RUNS)
do
CMD="python -W ignore launch_cxr.py --tasks $TASKS --batch_size $BATCH_SIZE --n_epochs $N_EPOCHS --lr $LR --l2 $L2 --lr_scheduler $LR_SCHEDULER --pretrained $PRETRAINED --drop_rate $DROP_RATE --warmup_steps $WARMUP_STEPS --warmup_unit $WARMUP_UNIT --min_lr $MIN_LR --res $RES --test_split $TEST_SPLIT --progress_bar $PROGRESS_BAR --seed $SEED --num_workers $NUM_WORKERS --log_every $LOG_EVERY --checkpoint_every $CHECKPOINT_EVERY --model_type $MODEL_TYPE --run_name ${EXP_NAMES[$run]} --slice_dict ${SLICE_DICT[$run]} --slice_pos_only ${SLICE_POS_ONLY[$run]}"
echo "Launching run $(($run+1)) with command:"
echo $CMD
#sleep 10
#eval $CMD
done


# ORIGINAL COMMAND
### python -W ignore launch_cxr.py --tasks CXR8-DRAIN_ALL --batch_size 16 --n_epochs 10 --lr 0.0001 --l2 0.000 --lr_scheduler linear --pretrained 1 --drop_rate 0.2 --warmup_steps 0 --warmup_unit epochs --min_lr 1e-6 --res 224 --test_split test --progress_bar 0 --seed 1701 --num_workers 6 --log_every 1.0 --checkpoint_every 1.0 --model_type soft_param_rep --run_name test_sps_rep_new_code --slice_dict '{"CXR8-DRAIN_PNEUMOTHORAX": ["chest_drain_cnn_pos"]}'
