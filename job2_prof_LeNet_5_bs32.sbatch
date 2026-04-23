#!/bin/bash
#SBATCH --job-name=prof_LeNet_5_bs32
#SBATCH --partition=gpu_titan
#SBATCH --nodelist=felix
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --mem=10000
#SBATCH --gres=gpu:1
#SBATCH --output=/home/sgflores/tesis/prof_nvidia_LeNet_5_bs32.%j.out
#SBATCH --error=/home/sgflores/tesis/prof_nvidia_LeNet_5_bs32.%j.err

module purge
source /home/sgflores/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_nvidia
cd /home/sgflores/tesis

python3 profiling_2.py --batch_size 32 --arch LeNet-5

echo "[JOB prof_LeNet_5_bs32] Profiling terminado"
wc -l /home/sgflores/tesis/profiling_nvidia_LeNet_5_bs32.csv 2>/dev/null
