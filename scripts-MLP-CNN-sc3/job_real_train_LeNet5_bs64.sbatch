#!/bin/bash
#SBATCH --job-name=real_train_LeNet5_bs64
#SBATCH --partition=amd
#SBATCH --nodelist=smexa
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=128
#SBATCH --mem=10000
#SBATCH --gres=gpu:1
#SBATCH --output=/home/sgflores/datos2/real_train_LeNet5_bs64.%j.out
#SBATCH --error=/home/sgflores/datos2/real_train_LeNet5_bs64.%j.err

module purge
source /home/sgflores/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_amd
cd /home/sgflores/datos2

# 1) Entrenamiento
python3 entrenamiento_cnn_sc3.py --batch_size 64 --arch LeNet-5
echo "[JOB real_train_LeNet5_bs64] Entrenamiento terminado"
wc -l /home/sgflores/datos2/entrenamiento_LeNet_5_bs64.csv 2>/dev/null
wc -l /home/sgflores/datos2/curvas_LeNet_5_bs64.csv 2>/dev/null

# 2) Profiling
python3 profiling_cnn_sc3.py --batch_size 64 --arch LeNet-5
echo "[JOB real_train_LeNet5_bs64] Profiling terminado"
wc -l /home/sgflores/datos2/profiling_LeNet_5_bs64.csv 2>/dev/null
