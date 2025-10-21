#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:TeslaL40s:1
#SBATCH --job-name=dino
#SBATCH --output=/auto/home/users/d/a/darimez/MIRO/vllmcluster/job.out
#SBATCH --error=/auto/home/users/d/a/darimez/MIRO/vllmcluster/job.err

module load releases/2023b
module load Python/3.11.5-GCCcore-13.2.0
module load CUDA/12.5.0
cd MIRO/vllmcluster

python train.py
