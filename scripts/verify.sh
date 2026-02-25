#!/bin/bash

module load conda/latest
#conda activate trans3
conda activate bayes_trans_temp

#K=32
K=3

echo deleting old data in 5 seconds
sleep 5
rm -rf ../sim_out/verify_$K
sbatch --export=K=$K verify.slurm

while [[ $(squeue -u yilinzhu_umass_edu | grep "verify" | wc -l) -gt 0 ]]; do
    echo Waiting...
    sleep 10
done

echo All nodes finished. Plotting results...
cd ../code
python plot_verify.py
echo done!
