#!/usr/bin/bash

#SBATCH -J Segmo_eval_V1-3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_grad
#SBATCH -w ariel-v2
#SBATCH -t 2-0
#SBATCH -o /data/dpfla3573/code/Segmo/logs/slurm-%A_eval_V1-3.out

cd /data/dpfla3573/code/Segmo
export PYTHONPATH=/data/dpfla3573/code/Segmo:$PYTHONPATH

/data/dpfla3573/anaconda3/envs/momask/bin/python run/eval_t2m_trans_res.py \
  --name MTRANS_V1-3 \
  --gpu_id 0 \
  --use_res_model \
  --dataset_name t2m \
  --which_epoch all \
  --time_steps 10 \
  --res_name tres_nlayer8_ld384_ff1024_rvq6ns_cdp0.2_sw