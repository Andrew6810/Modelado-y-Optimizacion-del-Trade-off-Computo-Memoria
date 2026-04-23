#!/bin/bash
#SBATCH --job-name=train_LeNet_5_bs32
#SBATCH --partition=gpu_titan
#SBATCH --nodelist=felix
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --mem=10000
#SBATCH --gres=gpu:1
#SBATCH --output=/home/sgflores/tesis/nvidia_LeNet_5_bs32.%j.out
#SBATCH --error=/home/sgflores/tesis/nvidia_LeNet_5_bs32.%j.err

module purge
source /home/sgflores/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_nvidia
cd /home/sgflores/tesis

python3 entrenamiento_2.py --batch_size 32 --arch LeNet-5

echo "[JOB train_LeNet_5_bs32] Entrenamiento terminado"
wc -l /home/sgflores/tesis/entrenamiento_nvidia_LeNet_5_bs32.csv 2>/dev/null
wc -l /home/sgflores/tesis/curvas_nvidia_LeNet_5_bs32.csv 2>/dev/null
