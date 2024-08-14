#!/bin/bash
#SBATCH -A r00664
#SBATCH -J mpg
#SBATCH -p general
#SBATCH -o mpg_%j.txt
#SBATCH -e mpg_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --time=10:00:00

time ../main $JB

