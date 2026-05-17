#!/bin/bash
#SBATCH --job-name=nvidia_LeNet5_bs16
#SBATCH --partition=gpu_titan
#SBATCH --nodelist=felix
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --mem=10000
#SBATCH --gres=gpu:1
#SBATCH --output=/home/sgflores/datos2/nvidia_LeNet5_bs16.%j.out
#SBATCH --error=/home/sgflores/datos2/nvidia_LeNet5_bs16.%j.err

module purge
source /home/sgflores/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_nvidia
cd /home/sgflores/datos2

# 1) Entrenamiento
python3 entrenamiento_nvidia.py --batch_size 16 --arch LeNet-5
echo "[JOB nvidia_LeNet5_bs16] Entrenamiento terminado"
wc -l /home/sgflores/datos2/entrenamiento_nvidia_LeNet_5_bs16.csv 2>/dev/null
wc -l /home/sgflores/datos2/curvas_nvidia_LeNet_5_bs16.csv 2>/dev/null

# 2) Profiling
python3 profiling_nvidia.py --batch_size 16 --arch LeNet-5
echo "[JOB nvidia_LeNet5_bs16] Profiling terminado"
wc -l /home/sgflores/datos2/profiling_nvidia_LeNet_5_bs16.csv 2>/dev/null
