#!/bin/bash

python /home/zanqi/LLaVA/scripts/merge_lora_weights.py \
    --model-path /home/zanqi/LLaVA/llava/checkpoints/llava-v1.5-7b-task-lora \
    --model-base liuhaotian/llava-v1.5-7b \
    --save-model-path /home/zanqi/LLaVA/llava/checkpoints/llava-v1.5-7b-task-lora-merged