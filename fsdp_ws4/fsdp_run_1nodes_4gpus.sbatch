#!/bin/bash
#SBATCH -A m4431_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 01:00:00
#SBATCH -N 1                    # or 1, depending on how many nodes you want
#SBATCH -n 4                    # total tasks = 4 per node * 2 nodes
#SBATCH -G 4                    # total GPUs = 4 per node * 2 nodes
#SBATCH -J blip2_fsdp
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

module load python/3.10
module load pytorch/2.8.0

# Activate your env (update this path to your actual venv if different)
source /pscratch/sd/r/rs2531/Course_Project/env/bin/activate

cd /pscratch/sd/r/rs2531/Course_Project/FSDP/fsdp_ws4/

srun -N 1 \
     --ntasks-per-node=4 \
     --gpus-per-node=4 \
     python -u train_fsdp.py --config fsdp_blip2_roco.json