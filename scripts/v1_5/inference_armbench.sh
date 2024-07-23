#!/bin/bash

python llava/eval/run_llava.py --model-path ~/LLaVA/llava/checkpoints/llava-2-7b-chat-task-qlora \
    --model-base ~/LLaVA/llava-v1.5-7b \
    --image-file ~/LLaVA/armbench/images/828.jpg \
    --query 'You are given a 512x384 image. Perform object detection on it. What are the objects and their bounding boxes in the image? Output in json format, x_min and y_min is the coordinate of the top left corner of an object bounding box, width and height are the width and height of the bounding box: [{\"bbox\": [x_min, y_min, width, height]}, {\"bbox\": [x_min, y_min, width, height]}, ...]'



