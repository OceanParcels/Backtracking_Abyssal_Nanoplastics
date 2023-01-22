import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib

import cartopy as cart

import cartopy.crs as ccrs
import cmocean.cm as cmo
import seaborn as sns
from glob import glob
import cartopy.feature as cfeature
import pandas as pd

from scipy import stats
import matplotlib.cm as cm

import random
from tqdm import tqdm

sim = xr.open_zarr('/storage/shared/oceanparcels/output_data/data_Claudio/set_19/set19_100.zarr/')
sim = sim.compute()

observations = sim.dims['obs']
time = sim['time'][0,:]
mean_diameter = np.zeros(observations)
std_diameter = np.zeros(observations)
mean_SA = np.zeros(observations)
std_SA = np.zeros(observations)

for dt in range(observations):
    diameters  = sim['diameter'][:, dt]
    SA = 6*diameters**2
    mean_diameter[dt] = np.mean(diameters)
    std_diameter[dt] = np.std(diameters)
    mean_SA[dt] = np.mean(SA)
    std_SA[dt] = np.std(SA)
