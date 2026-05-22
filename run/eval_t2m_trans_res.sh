#!/usr/bin/bash

#SBATCH -J Segmo_eval_baseline
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_grad
#SBATCH -w ariel-v2
#SBATCH -t 2-0
#SBATCH -o /data/dpfla3573/code/Segmo/logs/slurm-%A_eval_baseline.out

cd /data/dpfla3573/code/Segmo
export PYTHONPATH=/data/dpfla3573/code/Segmo:$PYTHONPATH

/data/dpfla3573/anaconda3/envs/momask/bin/python run/eval_t2m_trans_res.py \
  --name mtrans_test \
  --gpu_id 0 \
  --dataset_name t2m \
  --use_res_model \
  --res_name rtrans_test \
  --cond_scale 4 \
  --time_steps 10 \
  --which_epoch all