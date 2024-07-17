#!/bin/bash

python llava/eval/run_llava.py --model-path ~/LLaVA/llava/checkpoints/llama-2-7b-chat-task-qlora/checkpoint-80 \
    --image-file ~/LLaVA/armbench/images/9bb5a473-f0b9-4771-87e4-21ad2503c519.jpg \
    --query 'why was this photo taken?'



