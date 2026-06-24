#!/usr/bin/bash

#SBATCH -J Segmo_CoarseVQ
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_grad
#SBATCH -w ariel-v2
#SBATCH -t 2-0
#SBATCH -o /nas2/data/dpfla3573/code/Segmo/logs/slurm-%A_coarse_vq.out

cd /nas2/data/dpfla3573/code/Segmo
export PYTHONPATH=/nas2/data/dpfla3573/code/Segmo:$PYTHONPATH

/nas2/data/dpfla3573/anaconda3/envs/momask/bin/python run/train_coarse_vq.py \
  --name coarse_vq_v1 \
  --gpu_id 0 \
  --dataset_name t2m \
  --batch_size 64 \
  --vq_name rvq_nq6_dc512_nc512_noshare_qdp0.2 \
  --max_epoch 100 \
  --log_every 50 \
