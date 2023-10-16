#!/bin/bash

folder_path=""

for file in "$folder_path"/*; do
    if [[ -f "$file" ]]; then
        file_name=$(basename "$file")
        file_name="${file_name%.*}"
        dir_path=$(dirname "$file")
        echo ${file_name}
        python3 tools/preprocess_data.py --input ${file}
    fi
done
