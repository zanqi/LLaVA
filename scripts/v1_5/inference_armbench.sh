#!/bin/bash

python llava/eval/run_llava.py --model-path ~/LLaVA/llava/checkpoints/llava-2-7b-chat-task-qlora \
    --model-base ~/LLaVA/llava-v1.5-7b \
    --image-file ~/LLaVA/armbench/images/9bb5a473-f0b9-4771-87e4-21ad2503c519.jpg \
    --query 'What are the objects and their bounding boxes in the image? Output in json format: [{\"category_id\": 1, \"bbox\": [x_min, y_min, width, height]}, {\"category_id\": 2, \"bbox\": [x_min, y_min, width, height]}, ...]'



