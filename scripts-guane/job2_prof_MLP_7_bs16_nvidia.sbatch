#!/bin/bash
#SBATCH --job-name=prof_MLP_7_bs16
#SBATCH --partition=gpu_titan
#SBATCH --nodelist=felix
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --mem=10000
#SBATCH --gres=gpu:1
#SBATCH --output=/home/afleguizamoc/tesis/prof_nvidia_MLP_7_bs16.%j.out
#SBATCH --error=/home/afleguizamoc/tesis/prof_nvidia_MLP_7_bs16.%j.err

module purge
source /home/afleguizamoc/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_nvidia
cd /home/afleguizamoc/tesis

python3 entrenamiento_mlp_nvidia.py --batch_size 16 --arch MLP-7
python3 profiling_mlp_nvidia.py --batch_size 16 --arch MLP-7

echo "[JOB prof_MLP_7_bs16] Profiling terminado"
wc -l /home/afleguizamoc/tesis/profiling_nvidia_MLP_7_bs16.csv 2>/dev/null
