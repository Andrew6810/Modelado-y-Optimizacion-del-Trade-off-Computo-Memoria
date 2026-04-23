#!/bin/bash
#SBATCH --job-name=prof_VGG_16_bs128
#SBATCH --partition=amd
#SBATCH --nodelist=smexa
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=128
#SBATCH --mem=10000
#SBATCH --gres=gpu:1
#SBATCH --output=/home/sgflores/tesis/prof_VGG_16_bs128.%j.out
#SBATCH --error=/home/sgflores/tesis/prof_VGG_16_bs128.%j.err

module purge
source /home/sgflores/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_amd
cd /home/sgflores/tesis

python3 profiling_cnn_sc3.py --batch_size 128 --arch VGG-16

echo "[JOB prof_VGG_16_bs128] Profiling terminado"
wc -l /home/sgflores/tesis/profiling_VGG_16_bs128.csv 2>/dev/null
