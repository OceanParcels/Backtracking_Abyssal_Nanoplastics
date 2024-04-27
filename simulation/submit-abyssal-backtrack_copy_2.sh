#!/bin/bash -l
#
#SBATCH -J aby25           # the name of your job   
#SBATCH -p normal           # request normal partition, job takes > 1 hour (this line can also be left out because 'normal' is the default)  
#SBATCH -t 120:00:00         # time in hh:mm:ss you want to reserve for the job
#SBATCH -n 1               # the number of cores you want to use for the job, SLURM automatically determines how many nodes are needed
#SBATCH -o logs/hc13.%j.o  # the name of the file where the standard output will be written to. %j will be the jobid determined by SLURM
#SBATCH -e logs/hc13.%j.e  # the name of the file where potential errors will be written to. %j will be the jobid determined by SLURM
#SBATCH --mail-user=c.m.pierard@uu.nl
#SBATCH --mail-type=ALL

conda activate abyssal-nps

echo 'Running Backtracking Abyssal Nanoplastics simulation'
cd ${HOME}/3DModelling_SouthAtlantic/simulation

# first agument is the fragmentation timescale (int) second argument is the boolean for the fragmentation kernel

python3 backtrack_from_sampling_locations.py -ft 1000 -bm 1 -s 14
sleep 20
python3 backtrack_from_sampling_locations.py -ft 1000 -bm 1 -s 78
sleep 20
python3 backtrack_from_sampling_locations.py -ft 1000 -bm 1 -s 27
sleep 20
python3 backtrack_from_sampling_locations.py -ft 1000 -bm 1 -s 62
sleep 20
python3 backtrack_from_sampling_locations.py -ft 1000 -bm 1 -s 34

echo 'Finished computation.'
