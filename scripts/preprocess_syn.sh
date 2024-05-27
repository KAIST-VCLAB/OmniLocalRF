# Copyright (c) Dongyoung Choi
# Copyright (c) Meta Platforms, Inc. and affiliates.

DATA_SYN_DIR=../../../data/omnilocalrf/synthetic
SCENES=(lone_monk_on lone_monk_off pavillion_on pavillion_off sponza_on sponza_off)

for (( JOB_COMPLETION_INDEX=0; JOB_COMPLETION_INDEX<${#SCENES[@]}; JOB_COMPLETION_INDEX++ )) 
do
    SCENE_DIR=${DATA_SYN_DIR}/${SCENES[$JOB_COMPLETION_INDEX]}
    echo "Preprocessing $SCENE_DIR"

    python scripts/run_flow.py --data_dir ${SCENE_DIR} --frame_step 1
    python DPT/run_monodepth.py --input_path ${SCENE_DIR}/images --output_path ${SCENE_DIR}/depth --model_type dpt_large
done
