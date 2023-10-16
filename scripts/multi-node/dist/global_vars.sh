#!/bin/bash

# HOST_FILE=${WORK_DIR}/host_file
local_ips=$(hostname -I)
host_ips=$(grep -oE '([0-9]{1,3}\.){3}[0-9]{1,3}' ${HOST_FILE})

# Setting up the variables required for distributed training
GPUS_PER_NODE=8
NNODES=$(awk 'END{print NR}' ${HOST_FILE})
NODE_RANK=-1
MASTER_ADDR=$(echo "$host_ips" | head -n1)

# Get the NODE_RANK value of the current machine
# The current machine must also be one of the training machines
LOCAL_ADDR=""
counter=0
for host_ip in $host_ips; do
    for local_ip in $local_ips; do
        if [[ $local_ip == "${host_ip}" ]]; then
            if [ -z "$LOCAL_ADDR" ]; then
                LOCAL_ADDR=$host_ip
                NODE_RANK=$counter
            else echo "Found multiple local IP matches, please check local_ips." && exit 1; fi
        fi
    done
    counter=$((counter+1))
done
if [ -z "$LOCAL_ADDR" ]; then echo "Not found local IP in ${HOST_FILE}." && exit 1; fi
