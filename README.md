# Code for "Identifying the Origins of Nanoplastics in the Abyssal South Atlantic Using Backtracking Lagrangian Simulations with Fragmentation"

This repository contains code for the Lagrangian simulations and the analysis for the manuscript submitted to [Ocean and Coastal Research Journal](https://www.ocr-journal.org/).

## Getting Started

To get started with this project, you'll need to clone this repository to your local machine. You can do this by running the following command in your terminal:


`git clone https://github.com/OceanParcels/Backtracking_Abyssal_Nanoplastics.git`

To able to run the scripts we recommend creating a conda environment with the package versions required. You can create a new Conda environment from an environment.yml file using the conda env create command. Here's how you can do it:

`conda env create -f environment.yml`

## File structure
The script to run the parcels simulation it is found in `simulation/`. There you will see the following files:
1. [`backtrack_from_sampling_locations.py`](simulation/backtrack_from_sampling_locations.py): The parcels simulation script.
2. [`kernels_simple.py`](simulation/backtrack_from_sampling_locations.py): The kernels used in parcels simulation.
3. [`submit-3DMSA.sh`](simulation/submit-3DMSA.sh): sbatch script to submit simuation to supercomputer.

The scripts to run the analysis are located in `analysis/`, showing the following files:
4. ...
5. ...

## List of figures with the path to script of notebook where it was created
- Figure 1: [terminal_velocity[Range-of-validity].ipynb](analysis/terminal_velocity[Range-of-validity].ipynb)
- Figure 2: Not in repository. It was designed in specialized software.
- Figure 3: 
- Figure 4:
- Figure 5:
- Figure 6:

Supplementary material figures
- Figure S1 -  

## Data availability
The data will be made available with its own DOI, when the review process is over.
