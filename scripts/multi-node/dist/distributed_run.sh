#!/bin/bash
# Launch tmux
tmux new -ds ${TMUX_NAME}
ENV_PATH=/mnt/data01/huangyufei/miniconda3/envs/moe-data01
tmux send-keys -t ${TMUX_NAME} "conda activate ${ENV_PATH}" C-m
# get global vars
WORK_DIR=$(realpath "$(dirname "$0")")
. ${WORK_DIR}/global_vars.sh

# Train
SCRIPT=${1}
# TRAIN_CMD="cd ${CUR_DIR}; bash ${SCRIPT} ${GPUS_PER_NODE} ${NNODES} ${NODE_RANK} ${MASTER_ADDR}"
SET_GLOBAL_VAR="export GPUS_PER_NODE=${GPUS_PER_NODE}; export NNODES=${NNODES}; export NODE_RANK=${NODE_RANK}; export MASTER_ADDR=${MASTER_ADDR}"

tmux send-keys -t ${TMUX_NAME} "${SET_GLOBAL_VAR}" C-m

TRAIN_CMD="cd ${CUR_DIR}; bash ${SCRIPT}"
tmux send-keys -t ${TMUX_NAME} "${TRAIN_CMD}" C-m
