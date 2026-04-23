#!/bin/bash
#SBATCH --job-name=train_VGG_16_bs128
#SBATCH --partition=gpu_titan
#SBATCH --nodelist=felix
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --mem=10000
#SBATCH --gres=gpu:1
#SBATCH --output=/home/sgflores/tesis/nvidia_VGG_16_bs128.%j.out
#SBATCH --error=/home/sgflores/tesis/nvidia_VGG_16_bs128.%j.err

module purge
source /home/sgflores/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_nvidia
cd /home/sgflores/tesis

python3 entrenamiento_2.py --batch_size 128 --arch VGG-16

echo "[JOB train_VGG_16_bs128] Entrenamiento terminado"
wc -l /home/sgflores/tesis/entrenamiento_nvidia_VGG_16_bs128.csv 2>/dev/null
wc -l /home/sgflores/tesis/curvas_nvidia_VGG_16_bs128.csv 2>/dev/null
