#!/bin/bash

module load conda/latest
#conda activate trans3
conda activate bayes_trans_priors

#K=32
K=10

echo deleting old data in 5 seconds
sleep 5
rm sim_out/verify_$K

sbatch --export=K=$K verify.slurm

while [[ $(squeue -u yilinzhu_umass_edu | grep "verify" | wc -l) -gt 0 ]]; do
    echo Waiting...
    sleep 10
done
python python/plot_verify.py
echo done!
