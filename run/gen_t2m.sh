#!/usr/bin/bash

#SBATCH -J Segmo_gen_sample500
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_grad
#SBATCH -w ariel-v2
#SBATCH -t 2-0
#SBATCH -o /nas2/data/dpfla3573/code/Segmo/logs/slurm-%A_gen_sample500.out

cd /nas2/data/dpfla3573/code/Segmo
export PYTHONPATH=/nas2/data/dpfla3573/code/Segmo:$PYTHONPATH

/nas2/data/dpfla3573/anaconda3/envs/momask/bin/python run/gen_t2m.py \
  --name MTRANS_Baseline \
  --res_name RTRANS_Baseline \
  --gpu_id 0 \
  --ext Seg_test_500