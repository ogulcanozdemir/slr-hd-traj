#!/bin/bash
# 
# CompecTA (c) 2018
#
# Example job submission script
#
# TODO:
#   - Set name of the job below changing "Slurm" value.
#   - Set the requested number of tasks (cpu cores) with --ntasks parameter.
#   - Set the required time limit for the job with --time parameter.
#   - Put this script and all the input file under the same directory.
#   - Set the required parameters, input and output file names below.
#   - If you do not want mail please remove the line that has --mail-type
#   - Put this script and all the input file under the same directory.
#   - Submit this file using:
#      sbatch slurm_example.sh

# -= Resources =-

#SBATCH -N 1          # nodes requested
#SBATCH --job-name=fhd_extract
#SBATCH --ntasks-per-node=1
#SBATCH --partition=main
#SBATCH --mem-per-cpu=20000
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --output=/raid/users/oozdemir/code/tm-shd-slr/scripts/results/output/out-%j.out
#SBATCH --error=/raid/users/oozdemir/code/tm-shd-slr/scripts/results/error/err-%j.err
#SBATCH --time=365-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user ogulcan.ozdemir@yahoo.com

module load cuda/9.0
module load cudnn/7.1.1/cuda-9.0

/raid/users/oozdemir/anaconda3/bin/python /raid/users/oozdemir/code/tm-shd-slr/experiments_fhd_tf.py $*
