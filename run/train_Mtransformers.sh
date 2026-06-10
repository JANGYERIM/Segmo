#!/usr/bin/bash

#SBATCH -J Segmo_MTRANS_P5_test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_grad
#SBATCH -w ariel-v2
#SBATCH -t 4-0
#SBATCH -o /nas2/data/dpfla3573/code/Segmo/logs/slurm-%A_MTRANS_P5_test.out

cd /nas2/data/dpfla3573/code/Segmo
export PYTHONPATH=/nas2/data/dpfla3573/code/Segmo:$PYTHONPATH

/nas2/data/dpfla3573/anaconda3/envs/momask/bin/python run/train_t2m_transformer.py \
  --name MTRANS_P5_test\
  --gpu_id 0 \
  --dataset_name t2m \
  --batch_size 64 \
  --lambda_align 0.1 \
  --seg_captions /data4/local_datasets/HumanML3D/SegmentedCaption \
  --vq_name rvq_nq6_dc512_nc512_noshare_qdp0.2 \
