#!/bin/bash

#SBATCH --mem=20G
#SBATCH --time=01-00:00:00
#SBATCH --array=1-100
#SBATCH --job-name=verify
##SBATCH --constraint=avx512,2080ti
#SBATCH --constraint=avx512
#SBATCH --partition=gpu
#SBATCH --gpus=1

module load uri/main
module load R/4.2.1-foss-2022a
module load conda/latest
module load cuda/12.6
module load cudnn/8.9.7.29-12-cuda12.6
PYTHONPATH=""
source ~/.bashrc

#conda activate trans3
#conda activate bayes_trans_temp
conda activate bayes_trans_priors


echo "Hello from $(hostname)"

#module load cuda/12.4.1
#module load cudnn/8.9.7.29-12-cuda12.6
#conda activate trans_arm

#module load cuda/12.6
#module load cudnn/8.9.7.29-12-cuda12.6
#conda activate trans_gpu

#corr_var=0.1
#python python/verify_samplers.py $SLURM_ARRAY_TASK_ID $K $corr_var
#
python python/verify_samplers.py $SLURM_ARRAY_TASK_ID $K

#conda install -y numpy tqdm matplotlib IPython pandas conda-forge::rpy2
