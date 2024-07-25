#!/bin/bash

python llava/eval/run_llava.py --model-path ~/LLaVA/llava/checkpoints/llava-v1.5-7b-task-lora \
    --model-base liuhaotian/llava-v1.5-7b \
    --image-file ~/LLaVA/armbench/images/828.jpg \
    --query 'You are given a 512 by 384 image. Perform object detection on it. What are the objects and their bounding boxes in the image? Output in the following format: [[x_min, y_min, width, height], [x_min, y_min, width, height], ...] \n\nx_min and y_min are placeholder for the coordinate of the top left corner of an object bounding box, width and height are placeholder for the width and height of the bounding box.'



