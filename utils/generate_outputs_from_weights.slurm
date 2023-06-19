#!/bin/bash
#
#SBATCH -p a100	# partition (queue)
#SBATCH --nodelist=a100-004
#SBATCH -n 4			# number of cores
#SBATCH --job-name GenBigGAN
#SBATCH --mail-type=ALL
#SBATCH --mail-user=20397119@student.westernsydney.edu.au
#SBATCH --output %x.%N.%j.out	# STDOUT
#SBATCH --error %x.%N.%j.err	# STDERR

rm ./genned2/*
module load cuda/cuda-11.2 Python/Python3.9
python3 ./generate_outputs_from_weights.py 256 gen
