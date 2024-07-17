#!/bin/bash

python -m llava.serve.cli \
    --model-path ~/LLaVA/llava/checkpoints/llama-2-7b-chat-task-qlora/checkpoint-80 \
    --model-base lmsys/vicuna-7b-v1.5 \
    --image-file ~/LLaVA/armbench/images/9bb5a473-f0b9-4771-87e4-21ad2503c519.jpg \


