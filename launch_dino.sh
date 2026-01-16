#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=2G
#SBATCH --job-name=dino
#SBATCH --output=/auto/home/users/d/a/darimez/MIRO/vllmcluster/job.out
#SBATCH --error=/auto/home/users/d/a/darimez/MIRO/vllmcluster/job.err

module load releases/2023b
module load Python/3.11.5-GCCcore-13.2.0
module load CUDA/12.5.0
cd ~/MIRO/vllmcluster

# --- Define cleanup function ---
# cleanup() {
#     echo "Cleaning up W&B runs..."
#     wandb sync --sync-all --clean
# }
# --- Trap signals ---
# trap cleanup SIGINT SIGTERM EXIT

# cd $GLOBALSCRATCH/dino
# rm -rf embeddings/*vec*
# rm -rf distances/*vec*
# rm -rf proj/*
# cd ~/MIRO/vllmcluster

source dino/bin/activate 

# python test_sqlvector_refactor.py
# pytest -q test_cocodataset_augmentation_integration.py -s
# python main.py --model="dino,dino_attention,dinov3,dinov3_attention,clip" 
python main.py --visu
# python main.py --knn

# cleanup

# --partition=gpu
# --gres=gpu:1
# --constraint='TeslaL40s|TeslaA100|TeslaA100_80'
# --qos=preemptible