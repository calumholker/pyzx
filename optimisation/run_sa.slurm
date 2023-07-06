#!/bin/bash
#SBATCH --job-name=sa
#SBATCH --output=sa_output_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=4G
#SBATCH --time=24:00:00

python3 -u simulated_annealing.py /home/calum/pyzx/circuits/benchmarking/Fast/before