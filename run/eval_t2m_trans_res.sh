#!/usr/bin/bash

#SBATCH -J Segmo_eval_t2m_P2_M_V1-5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_grad
#SBATCH -w ariel-v2
#SBATCH -t 2-0
#SBATCH -o /nas2/data/dpfla3573/code/Segmo/logs/slurm-%A_eval_t2m_P2_M_V1-5.out

cd /nas2/data/dpfla3573/code/Segmo
export PYTHONPATH=/nas2/data/dpfla3573/code/Segmo:$PYTHONPATH

/nas2/data/dpfla3573/anaconda3/envs/momask/bin/python run/eval_t2m_trans_res.py \
  --name t2m_P2_M_V1-5 \
  --gpu_id 0 \
  --dataset_name t2m \
  --which_epoch all \
  --time_steps 10 \
  --use_res_model \
  --res_name tres_nlayer8_ld384_ff1024_rvq6ns_cdp0.2_sw