#!/bin/bash

WORK_DIR=$(realpath "$(dirname "$0")")
CUR_DIR=$(pwd)

HOST_FILE=${1:-"${WORK_DIR}/hostfiles/host_file"}
TMUX_NAME=${2:-"instruction"}

HOST_FILE=$(realpath ${HOST_FILE})

echo ${WORK_DIR}
echo ${CUR_DIR}
echo ${HOST_FILE}

pssh -ih ${HOST_FILE} "tmux send-keys -t ${TMUX_NAME} C-c C-m"
# parallel-ssh -ih host_file "export WORK_DIR=${WORK_DIR}; bash ${WORK_DIR}/dist/cancel.sh"
