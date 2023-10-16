#!/bin/bash

WORK_DIR=$(realpath "$(dirname "$0")")
CUR_DIR=$(pwd)

SCRIPT=${1}
HOST_FILE=${2:-"${WORK_DIR}/hostfiles/host_file"}
TMUX_NAME=${3:-"instruction"}

HOST_FILE=$(realpath ${HOST_FILE})

echo ${WORK_DIR}
echo ${CUR_DIR}
echo ${HOST_FILE}

pssh -ih ${HOST_FILE} "export HOST_FILE=${HOST_FILE}; export CUR_DIR=${CUR_DIR}; export TMUX_NAME=${TMUX_NAME}; bash ${WORK_DIR}/dist/distributed_run.sh ${SCRIPT}"
# parallel-ssh -ih host_file "export WORK_DIR=${WORK_DIR}; bash ${WORK_DIR}/dist/cancel.sh"
