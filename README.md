# Code for "Identifying the Origins of Nanoplastics in the Abyssal South Atlantic Using Backtracking Lagrangian Simulations with Fragmentation"

This repository contains code for the Lagrangian simulations and the analysis for the manuscript submitted to [Ocean and Coastal Research Journal](https://www.ocr-journal.org/).

## Getting Started

To reproduce the simulations and the analysis, you will need to clone this repository to your local machine. You can do this by running the following command in your terminal:


`git clone https://github.com/OceanParcels/Backtracking_Abyssal_Nanoplastics.git`

To able to run the scripts we recommend creating a conda environment with the package versions required. You can create a new Conda environment from an [environment.yml](environment.yml) file using:

`conda env create -f environment.yml`

## File Structure
List of most important scripts with a short description.

#### [Simulation](simulation/)

1. [`backtrack_from_sampling_locations.py`](simulation/backtrack_from_sampling_locations.py): _main simulation_ script.
2. [`kernels_simple.py`](simulation/backtrack_from_sampling_locations.py): the kernels used in parcels simulation.
3. [`backtrack_from_existing_file.py`](simulation/backtrack_from_existing_file.py): Parcels script to submit simulation from unfinished simulation due to TIMEOUTs.
4. [`submit-abyssal-backtrack.sh`](simulation/submit-abyssal-backtrack.sh): sbatch script to submit simulation to supercomputer.
5. [`stitch_together_output_files.ipynb`](simulation/stitch_together_output_files.ipynb): notebook to concatenate the output files of simulations that got interrupted.

#### [Analysis](analysis/)

The first script that you need to run to save the data used for plotting and the rest of the scriots are: `vertical_histograms.py` and `size_distribution.py`.
1. [`analysis_functions.py`](analysis\analysis_functions.py): functions used across analysis scripts.
2. [`Map_origin_particles_surface.ipynb`](analysis\Map_origin_particles_surface.ipynb): Figure 3 notebook.
2. [`size_distribution.py`](analysis/size_distribution.py): computes the size distributions at the surface and also plots the distribution and locations at the surface when they started to sink.
3. [`supplemnetary_material_plots.ipynb`](analysis/supplementary_material_plots.ipynb): Notebook to analyze the simulations which considered three maximum size classes $k$.
4. [`terminal_velocity[Range-of-validity].ipynb`](analysis/terminal_velocity[Range-of-validity].ipynb): Notebook analyzing the terminal velocity for different sizes of PET plastic particles in the ocean.
5. [`vertical_histograms.py`](analysis/vertical_histograms.py): Vertical transport analysis of the backwards-in-time trajectories. Also it shows the depth at which particles become nanoplastics.
6. [`vertical_Kz_profiles.py`](analysis/vertical_Kz_profiles.py): sample the Kz field and compute a simple climatology.

#### [Animations](animations/)

1. [`animation_3D.py`](animations/animation_3D.py): animates the trajectory of one random particle in 3D.
2. [`animation_flow_particles.py`](animations/animation_flow_particles.py): animates particles with the temperature filed backwards in time. 
3. [`animation_forward_MLD.py`](animations/animation_forward_MLD.py): animates the same but forward in time.

[Link to (some) animations.](https://cpierard.github.io/projects/backtrack-nps/)

## List of Figures
The figures are in [`article_figs`](article_figs/). The figures with their respective path to script or notebook where they were created are listed below.
- **Figure 1**: [`terminal_velocity[Range-of-validity].ipynb`](analysis/terminal_velocity[Range-of-validity].ipynb)
- **Figure 2**: Not in repository. It was made in specialized software.
- **Figure 3**: [`Map_origin_particles_surface.ipynb`](Map_origin_particles_surface.ipynb)
- **Figure 4**: [`size_distribution.py`](analysis/size_distribution.py)
- **Figure 5**: [`vertical_histograms.py`](analysis/vertical_histograms.py)
- **Figure 6**: [`size_distribution.py`](analysis/size_distribution.py)

**Supplementary Information Figures**
- **Figure S1**: [`supplementary_material_plots.ipynb`](analysis/supplementary_material_plots.ipynb)
- **Figure S2**: [`vertical_histograms.py`](analysis/vertical_histograms.py)

## Data Availability
The data will be made available with its own DOI, when the review process is over.
