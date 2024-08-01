#!/bin/bash

python llava/eval/run_llava.py --model-path llava/checkpoints/llava-v1.5-7b-task-lora \
    --model-base liuhaotian/llava-v1.5-7b \
    --image-file armbench/images/87.jpg \
    --query 'Perform object detection on the given image. First, resize the image to 1x1, then answer: what are the bounding boxes of the objects in the image? Output should be in  the format of a list of lists, where each list represents a bounding box: [[x_min, y_min, width, height], [x_min, y_min, width, height], ...] \n\nx_min and y_min are placeholder for the coordinate of the top left corner of an object bounding box, width and height are placeholder for the width and height of the bounding box. \n\nThe x_min, y_min, width and height of the bounding box should be positive decimal numbers between 0 and 1. \n\n \n\nFor example, if there are two objects in the image, the bounding boxes of the objects could be [[0.13, 0.24, 0.37, 0.4], [0.5, 0.55, 0.23, 0.25]].'



