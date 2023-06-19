#!/bin/bash
#
#SBATCH -p a100	                    # partition (queue)
#SBATCH --nodelist=a100-002         # node to be allocated
#SBATCH -n 4			            # number of cores
#SBATCH --job-name BigGANFinal      # name of the job
#SBATCH --mail-type=ALL             # Email settings
#SBATCH --mail-user=<email>
#SBATCH --output %x.%N.%j.out	    # STDOUT
#SBATCH --error %x.%N.%j.err	    # STDERR

rm ./data/genned/*
module load cuda/cuda-11.2 Python/Python3.9
python3 ./BigGAN.py
