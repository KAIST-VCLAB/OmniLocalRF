# Copyright (c) Dongyoung Choi
# Copyright (c) Meta Platforms, Inc. and affiliates.

DATA_REAL_DIR=../../../data/omnilocalrf/real
SCENES=(dormitory library red_building rocket temple yongsil)

for (( JOB_COMPLETION_INDEX=0; JOB_COMPLETION_INDEX<${#SCENES[@]}; JOB_COMPLETION_INDEX++ )) 
do
    SCENE_DIR=${DATA_REAL_DIR}/${SCENES[$JOB_COMPLETION_INDEX]}
    echo "Preprocessing $SCENE_DIR"

    python scripts/run_flow.py --data_dir ${SCENE_DIR} --frame_step 4
    python DPT/run_monodepth.py --input_path ${SCENE_DIR}/images --output_path ${SCENE_DIR}/depth --model_type dpt_large
done