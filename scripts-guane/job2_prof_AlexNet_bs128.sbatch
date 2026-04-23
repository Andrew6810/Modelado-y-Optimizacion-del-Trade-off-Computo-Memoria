#!/bin/bash
#SBATCH --job-name=prof_AlexNet_bs128
#SBATCH --partition=gpu_titan
#SBATCH --nodelist=felix
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --mem=10000
#SBATCH --gres=gpu:1
#SBATCH --output=/home/sgflores/tesis/prof_nvidia_AlexNet_bs128.%j.out
#SBATCH --error=/home/sgflores/tesis/prof_nvidia_AlexNet_bs128.%j.err

module purge
source /home/sgflores/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_nvidia
cd /home/sgflores/tesis

python3 profiling_2.py --batch_size 128 --arch AlexNet

echo "[JOB prof_AlexNet_bs128] Profiling terminado"
wc -l /home/sgflores/tesis/profiling_nvidia_AlexNet_bs128.csv 2>/dev/null
