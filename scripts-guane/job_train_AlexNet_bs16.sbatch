#!/bin/bash
#SBATCH --job-name=train_AlexNet_bs16
#SBATCH --partition=amd
#SBATCH --nodelist=smexa
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=128
#SBATCH --mem=10000
#SBATCH --gres=gpu:1
#SBATCH --output=/home/sgflores/tesis/train_AlexNet_bs16.%j.out
#SBATCH --error=/home/sgflores/tesis/train_AlexNet_bs16.%j.err

module purge
source /home/sgflores/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_amd
cd /home/sgflores/tesis

python3 entrenamiento_cnn_sc3.py --batch_size 16 --arch AlexNet

echo "[JOB train_AlexNet_bs16] Entrenamiento terminado"
wc -l /home/sgflores/tesis/entrenamiento_AlexNet_bs16.csv 2>/dev/null
wc -l /home/sgflores/tesis/curvas_AlexNet_bs16.csv 2>/dev/null
