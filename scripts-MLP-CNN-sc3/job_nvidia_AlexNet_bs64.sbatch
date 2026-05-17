#!/bin/bash
#SBATCH --job-name=nvidia_AlexNet_bs64
#SBATCH --partition=gpu_titan
#SBATCH --nodelist=felix
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --mem=10000
#SBATCH --gres=gpu:1
#SBATCH --output=/home/sgflores/datos2/nvidia_AlexNet_bs64.%j.out
#SBATCH --error=/home/sgflores/datos2/nvidia_AlexNet_bs64.%j.err

module purge
source /home/sgflores/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_nvidia
cd /home/sgflores/datos2

# 1) Entrenamiento
python3 entrenamiento_nvidia.py --batch_size 64 --arch AlexNet
echo "[JOB nvidia_AlexNet_bs64] Entrenamiento terminado"
wc -l /home/sgflores/datos2/entrenamiento_nvidia_AlexNet_bs64.csv 2>/dev/null
wc -l /home/sgflores/datos2/curvas_nvidia_AlexNet_bs64.csv 2>/dev/null

# 2) Profiling
python3 profiling_nvidia.py --batch_size 64 --arch AlexNet
echo "[JOB nvidia_AlexNet_bs64] Profiling terminado"
wc -l /home/sgflores/datos2/profiling_nvidia_AlexNet_bs64.csv 2>/dev/null
