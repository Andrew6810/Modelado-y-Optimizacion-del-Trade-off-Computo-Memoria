#!/bin/bash
#SBATCH --job-name=prof_MLP_7_bs16
#SBATCH --partition=amd
#SBATCH --nodelist=exadell
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=128
#SBATCH --mem=10000
#SBATCH --gres=gpu:1
#SBATCH --output=/home/afleguizamoc/tesis/prof_amd_MLP_7_bs16.%j.out
#SBATCH --error=/home/afleguizamoc/tesis/prof_amd_MLP_7_bs16.%j.err

module purge
source /home/afleguizamoc/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_amd
cd /home/afleguizamoc/tesis

python3 entrenamiento_mlp_amd.py --batch_size 16 --arch MLP-7
python3 profiling_mlp_amd.py --batch_size 16 --arch MLP-7

echo "[JOB prof_MLP_7_bs16] Profiling terminado"
wc -l /home/afleguizamoc/tesis/profiling_amd_MLP_7_bs16.csv 2>/dev/null