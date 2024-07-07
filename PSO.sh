#!/bin/bash

#SBATCH --job-name="PSO"
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=compute
#SBATCH --mem-per-cpu=1GB
#SBATCH --account=research-eemcs-me

module load 2023r1
module load python
module load py-numpy
module load py-pandas

python run_PSO.py >> log.txt