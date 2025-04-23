#!/bin/bash

#SBATCH --job-name=IM
#SBATCH --partition=small
#SBATCH --ntasks=16
#SBATCH --nodes=1
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

python ./im.py