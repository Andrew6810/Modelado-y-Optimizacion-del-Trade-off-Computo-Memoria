#!/bin/bash
#SBATCH --job-name=prof_VGG_16_bs64
#SBATCH --partition=gpu_titan
#SBATCH --nodelist=felix
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --mem=10000
#SBATCH --gres=gpu:1
#SBATCH --output=/home/sgflores/tesis/prof_nvidia_VGG_16_bs64.%j.out
#SBATCH --error=/home/sgflores/tesis/prof_nvidia_VGG_16_bs64.%j.err

module purge
source /home/sgflores/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_nvidia
cd /home/sgflores/tesis

python3 profiling_2.py --batch_size 64 --arch VGG-16

echo "[JOB prof_VGG_16_bs64] Profiling terminado"
wc -l /home/sgflores/tesis/profiling_nvidia_VGG_16_bs64.csv 2>/dev/null
