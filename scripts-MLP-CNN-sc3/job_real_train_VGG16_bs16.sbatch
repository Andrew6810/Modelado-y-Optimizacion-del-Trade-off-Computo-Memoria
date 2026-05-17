#!/bin/bash
#SBATCH --job-name=real_train_VGG16_bs16
#SBATCH --partition=amd
#SBATCH --nodelist=smexa
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=128
#SBATCH --mem=10000
#SBATCH --gres=gpu:1
#SBATCH --output=/home/sgflores/datos2/real_train_VGG16_bs16.%j.out
#SBATCH --error=/home/sgflores/datos2/real_train_VGG16_bs16.%j.err

module purge
source /home/sgflores/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_amd
cd /home/sgflores/datos2

# 1) Entrenamiento
python3 entrenamiento_cnn_sc3.py --batch_size 16 --arch VGG-16
echo "[JOB real_train_VGG16_bs16] Entrenamiento terminado"
wc -l /home/sgflores/datos2/entrenamiento_VGG_16_bs16.csv 2>/dev/null
wc -l /home/sgflores/datos2/curvas_VGG_16_bs16.csv 2>/dev/null

# 2) Profiling
python3 profiling_cnn_sc3.py --batch_size 16 --arch VGG-16
echo "[JOB real_train_VGG16_bs16] Profiling terminado"
wc -l /home/sgflores/datos2/profiling_VGG_16_bs16.csv 2>/dev/null
