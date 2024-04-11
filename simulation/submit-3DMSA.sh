#!/bin/bash -l
#
#SBATCH -J HC13_4           # the name of your job   
#SBATCH -p normal           # request normal partition, job takes > 1 hour (this line can also be left out because 'normal' is the default)  
#SBATCH -t 120:00:00         # time in hh:mm:ss you want to reserve for the job
#SBATCH -n 1               # the number of cores you want to use for the job, SLURM automatically determines how many nodes are needed
#SBATCH -o hc13_3.%j.o  # the name of the file where the standard output will be written to. %j will be the jobid determined by SLURM
#SBATCH -e hc13_3.%j.e  # the name of the file where potential errors will be written to. %j will be the jobid determined by SLURM
#SBATCH --mail-user=c.m.pierard@uu.nl
#SBATCH --mail-type=ALL

conda activate abyssal-nps

echo 'Running Backtracking Abyssal Nanoplastics simulation'
cd ${HOME}/3DModelling_SouthAtlantic/simulation

# first agument is the fragmentation timescale (int) second argument is the boolean for the fragmentation kernel

python3 backtrack_from_sampling_locations.py -ft 23169 -bm True &
sleep 15
python3 backtrack_from_sampling_locations.py -ft 19430 -bm True &
sleep 15
python3 backtrack_from_sampling_locations.py -ft 25170 -bm True &
sleep 15
python3 backtrack_from_sampling_locations.py -ft 23169 -bm False &
sleep 15
python3 backtrack_from_sampling_locations.py -ft 19430 -bm False &
sleep 15
python3 backtrack_from_sampling_locations.py -ft 25170 -bm False 

echo 'Finished computation.'
