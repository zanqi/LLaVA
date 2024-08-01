#!/bin/bash

#SBATCH --job-name=llava_train
#SBATCH --mail-type=END
#SBATCH --mail-user=zanqil@cs.washington.edu

#SBATCH --account=sciencehub
#SBATCH --partition=gpu-a40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=120G
#SBATCH --gpus=1
#SBATCH --time=2-00:00:00

#SBATCH --chdir=/gscratch/sciencehub/zanqil/LLaVA
#SBATCH --export=all
#SBATCH --output=/gscratch/sciencehub/zanqil/LLaVA/output
#SBATCH --error=/gscratch/sciencehub/zanqil/LLaVA/error

# Modules to use (optional).
apptainer exec --pwd /gscratch/sciencehub/zanqil/LLaVA  --nv  -B /gscratch /mmfs1/gscratch/sciencehub/amazon_perception/pytorch.sif bash -c 'source ~/.bashrc ; conda activate /gscratch/sciencehub/zanqil/llava ; export HF_HOME=/gscratch/sciencehub/zanqil/cache ; export WANDB_API_KEY=4247a6d61ee555dcc2835bbe242de06d61ed7d09 ; sh -x scripts/v1_5/finetune_task_lora_armbench.sh'

# Done
