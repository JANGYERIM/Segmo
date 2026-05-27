#!/usr/bin/bash

#SBATCH -J Segmo_gen_sample300
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_grad
#SBATCH -w ariel-v2
#SBATCH -t 2-0
#SBATCH -o /data/dpfla3573/code/Segmo/logs/slurm-%A_gen_sample300.out

cd /data/dpfla3573/code/Segmo
export PYTHONPATH=/data/dpfla3573/code/Segmo:$PYTHONPATH

/data/dpfla3573/anaconda3/envs/momask/bin/python run/gen_t2m.py \
  --name MTRANS_V1 \
  --res_name rtrans_V1 \
  --gpu_id 0 \
  --ext SegMo_Baseline_inference300