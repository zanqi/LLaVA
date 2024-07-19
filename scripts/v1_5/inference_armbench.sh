#!/bin/bash

python llava/eval/run_llava.py --model-path ~/LLaVA/llava/checkpoints/llava-2-7b-chat-task-qlora \
    --model-base ~/LLaVA/llava-v1.5-7b \
    --image-file ~/LLaVA/armbench/images/b6370b0b-58de-4500-bd81-6cd4a6337225.jpg \
    --query 'What are the objects and their bounding boxes in the image? Output in json format: [{\"bbox\": [x_min, y_min, width, height]}, {\"bbox\": [x_min, y_min, width, height]}, ...]'



