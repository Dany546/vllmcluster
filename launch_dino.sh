#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
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
# rm -rf embeddings/*clip*
# rm -rf distances/*clip*
# cd ~/MIRO/vllmcluster

source dino/bin/activate
# python main.py --model="dino"
# source visu/bin/activate
# pip install ipywidgets IPython --quiet
# python main.py --visu
python main.py --knn

# cleanup

# --partition=gpu
# --gres=gpu:1
# --constraint='TeslaL40s|TeslaA100|TeslaA100_80'
# --qos=preemptible
