#%% Importing necessary libraries
from glob import glob
import numpy as np
import xarray as xr
import itertools
import matplotlib.pyplot as plt
import cmocean.cm as cmo
from matplotlib.gridspec import GridSpec
from cartopy import geodesic
import cartopy.crs as ccrs
import shapely
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import draft_functions as funk

run_for_loop = False

# Define initial conditions
initial_depth = -5000  # int(sys.argv[1])  # 5 # 60 # 5179
lon_sample = 6.287  # 6.25
lat_sample = -32.171  # -32.171
origin = (lon_sample, lat_sample)
sim_time = 4484
# Set simulation time range
start_time = datetime.strptime('2019-01-20 12:00:00', '%Y-%m-%d %H:%M:%S')

datelist = pd.date_range(end=start_time, periods=sim_time+1)[::-1]
end_time = datelist[0]

# Define simulation fragmentation timescales
simulations = [10] + [i for i in range(100, 501, 100)]
simulations += [1000, 10000]
# Set depth bins for histograms
depth_bins = np.linspace(-5500, 0, 56)  # creates a 100m bins

# Create dictionary to store results of fragmentations into nanoparticles (NPs)
frag_into_NPs = {}
extra = ''

#%% Loop over simulations
if run_for_loop:
    for ft in simulations:
        print('Computing fragmentation timescale: ', ft)
        sim_dict = {}

        # Load the data from the simulation
        local_path = f'/storage/shared/oceanparcels/output_data/data_Claudio/hc13_2/hc13_{ft}.zarr'
        sim = xr.open_zarr(local_path)
        nano = sim.where(sim.radius < 1e-6/2, drop=False)
        
        # Find indices of the particles that are not NaN
        aux = np.isnan(nano['radius'].values)
        traj = nano.trajectory.values
        index_NP = len(nano.obs) - 1 - np.sum(aux, axis=1)
        
        sim_dict['particle_index'] = index_NP

        # Get depth, latitude, and longitude of NPs
        z = -nano['z'].values
        sim_dict['depths'] = z[(traj, index_NP)]

        latNP = nano['lat'].values
        lonNP = nano['lon'].values

        sim_dict['lat'] = latNP[(traj, index_NP)]
        sim_dict['lon'] = lonNP[(traj, index_NP)]

        # Compute displacement of NPs from a reference point (origin)
        xy_pos = (lonNP[(traj, index_NP)], latNP[(traj, index_NP)])
        sim_dict['displacement'] = funk.haversine(origin, xy_pos)

        # Compute histograms of particle counts for each depth bin over time
        zbins = len(depth_bins)-1
        hist_counts = np.zeros((zbins, sim_time))
        
        if ft == 10: 
            t_range = range(0, 1500, 1)
            
        else:
            t_range = range(0, sim_time, 1)
        
        
        for i, fr in enumerate(tqdm(t_range)):
            x = np.histogram(-nano['z'][:, fr].dropna('trajectory'), bins=depth_bins,
                            density=False)
            hist_counts[:, i] = x[0]

        # Compute total number of particles in each time step
        total_particles = np.sum(hist_counts, axis=0)
        sim_dict['counts'] = total_particles

        p_zt = np.ma.masked_equal(hist_counts, 0)/total_particles
        # creat a rolling average of h_masked

        # h_masked.rolling(time=10, center=True, ).mean()


        # compute the vertical information of h_masked
        I = np.log2(1/p_zt).data
        H = np.sum(p_zt.data*I, axis=0)

        sim_dict['vertical_distribution'] = p_zt
        sim_dict['vertical_information'] = I
        sim_dict['entropy'] = H

        frag_into_NPs[ft] = sim_dict

    np.save('../data/frag_into_NPs.npy', frag_into_NPs, allow_pickle=True)

#%% 
if not run_for_loop:
    frag_into_NPs = np.load('../data/frag_into_NPs.npy', allow_pickle=True)[()]

# %% create dataframe with the data of fragmentation into NPs 
if run_for_loop:
    df = pd.DataFrame(columns=['Particles', 'z median', 'z min', 'z max',
                          'T_s mean', 'T_s std', 'T_s median', 'T_s min', 'T_s max',
                          'X mean', 'X std', 'X median', 'X min', 'X max'])


    for ft in simulations:
        df.loc[ft] = [frag_into_NPs[ft]['particle_index'].size, 
                        np.nanmedian(frag_into_NPs[ft]['depths']),
                        np.nanmin(frag_into_NPs[ft]['depths']),
                        np.nanmax(frag_into_NPs[ft]['depths']),
                        np.nanmean(frag_into_NPs[ft]['particle_index']),
                        np.nanstd(frag_into_NPs[ft]['particle_index']),
                        np.nanmedian(frag_into_NPs[ft]['particle_index']),
                        np.nanmin(frag_into_NPs[ft]['particle_index']),
                        np.nanmax(frag_into_NPs[ft]['particle_index']),
                        np.nanmean(frag_into_NPs[ft]['displacement']),
                        np.nanstd(frag_into_NPs[ft]['displacement']),
                        np.nanmedian(frag_into_NPs[ft]['displacement']),
                        np.nanmin(frag_into_NPs[ft]['displacement']),
                        np.nanmax(frag_into_NPs[ft]['displacement'])]
    

    df.to_csv('../data/stats_frag_into_NPs.csv')
    df.to_latex('../data/frag_into_NPS_table.tex') # to print in latex format and save in a file

# %% Vertical distributions plots

x, y = np.meshgrid(datelist, depth_bins)

fig, ax = plt.subplots(ncols=1, nrows=len(simulations), figsize=(8, 8),
                       sharex=True, constrained_layout=True)

color_map = cmo.matter_r

for j, ft in enumerate(simulations):
    ax[j].set_facecolor('lightgrey')
    im = ax[j].pcolormesh(x, y, frag_into_NPs[ft]['vertical_distribution'],
                          cmap=color_map,
                          vmin=0, vmax=0.2)
    ax[j].text(18200, -1500, f'$\lambda_f$ = {ft} days', fontsize=8,
               ha='right')
    ax[j].set_yticks([-5500, -2500, 0])
    ax[j].grid()

ax[4].set_ylabel('Depth (m)')

fig.colorbar(im, ax=ax[-1], orientation='horizontal',
             extend='max', label='Depth Probability of Nanoplastics')

# ax[0].set_title('Nanoparticles (50-1000 $nm$) in the water column')
plt.show()
fig.savefig('../article_figs/vertical_distributionsNPs.png', dpi=300,
            facecolor=(1, 0, 0, 0))

# # %% Vertical Information plots

# x, y = np.meshgrid(datelist, depth_bins)

# fig, ax = plt.subplots(ncols=1, nrows=len(simulations), figsize=(8, 8),
#                        sharex=True, constrained_layout=True)

# color_map = cmo.algae

# for j, ft in enumerate(simulations):
#     ax[j].set_facecolor('lightgrey')
#     im = ax[j].pcolormesh(x, y, frag_into_NPs[ft]['vertical_information'].data,
#                           cmap=color_map)
#     ax[j].text(18200, -1500, f'$\lambda_f$ = {ft} days', fontsize=8,
#                ha='right')
#     ax[j].set_yticks([-5500, -2500, 0])
#     ax[j].grid()

# ax[4].set_ylabel('Depth (m)')
# fig.colorbar(im, ax=ax[-1], orientation='horizontal', label='Information (bits)')

# ax[0].set_title('Nanoparticles (50-1000 $nm$) in the water column')
# plt.show()
# fig.savefig('../figs/vertical_Information.png', dpi=300,
#             facecolor=(1, 0, 0, 0))

# #%% Entropy plots   

# fig = plt.figure(figsize=(6, 4))
# ax = fig.add_subplot(111)
# ax.grid(linestyle='--')
# ax.set_xlabel('Time (days)')
# ax.set_ylabel('Entropy (bits)')

# for ft in simulations[::-1]:
#     ax.plot(frag_into_NPs[ft]['entropy'], label=f'$\lambda_f$ = {ft} day')

# handles, labels = ax.get_legend_handles_labels()
# handles = handles[::-1]
# labels = labels[::-1]

# ax.legend(handles, labels)

# fig.savefig('../figs/entropy.png', dpi=300,
#             facecolor=(1, 0, 0, 0))

# %% Depth vs displacement Plot
# '-', '--', '-.', ':', 'None', ' ', '', 'solid', 'dashed', 'dashdot', 'dotted'
marker = itertools.cycle(('v', 'h', 'd', 'o', 'X', 'P', '^', 's'))

fig = plt.figure(figsize=(8, 8))
gs = GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4], wspace=0.01,
              hspace=0.01)

ax1 = fig.add_subplot(gs[0])

ax1.set_axis_off()

ax2 = fig.add_subplot(gs[1])
ax2.set_axis_off()

ax3 = fig.add_subplot(gs[2])
ax3.grid(linestyle='--')
ax3.set_xlabel('Total Displacement as Nanoparticles (km)')
ax3.set_ylabel('Depth of fragmentation into nanoparticles (m)')

ax4 = fig.add_subplot(gs[3])
ax4.set_axis_off()

alph = 0.7
for j, ft in enumerate(simulations[::-1]):

    ax1.hist(frag_into_NPs[ft]['displacement']/1e3, bins=30, alpha=alph)

    ax4.hist(frag_into_NPs[ft]['depths'], bins=30, alpha=alph,
             orientation="horizontal")

    ax3.scatter(frag_into_NPs[ft]['displacement']/1e3,
                frag_into_NPs[ft]['depths'],
                s=15,
                alpha=alph, label=f"$\lambda_f$ = {ft} days",
                marker=next(marker))


ax3.axhline(initial_depth, color='k', linestyle='--', label='Sampling depth')
# ax3.scatter(0, initial_depth,
#            label = 'Sampling Location', marker='*', s=50)

ax3.axhline(0, color='r', linestyle='--')
ax3.text(100, -160, 'Surface', color='r')

handles, labels = ax3.get_legend_handles_labels()
handles = handles[::-1]
labels = labels[::-1]

ax3.legend(handles, labels, shadow=False)

fig.savefig('../article_figs/depth_n_displacement.png', dpi=300,
            facecolor=(1, 0, 0, 0))


# %% ecdf surfacetime and size distribution of particles at the surface
fig, ax = plt.subplots(1, 3, figsize=(12, 3.5), tight_layout=True)

for j, ft in enumerate(simulations[::-1]):

    x, y = funk.ecdf(abs(frag_into_NPs[ft]['depths']), normalized=True,
                     invert=False)
    ax[0].plot(x, y, drawstyle='steps-post', label=f'$\lambda_f$ = {ft} days')
    
    x, y = funk.ecdf(frag_into_NPs[ft]['particle_index'], normalized=True,
                     invert=False)
    ax[1].plot(x, y, drawstyle='steps-post')
    
    x, y = funk.ecdf(frag_into_NPs[ft]['displacement']/1e3, normalized=True,
                     invert=False)
    ax[2].plot(x, y, drawstyle='steps-post', label=f'$\lambda_f$ = {ft} days')



ax[0].axvline(0, ls='--', label=r"Surface", color='r')
ax[0].axvline(-initial_depth, ls='--', label=r"Sampling Depth", color='k')

ax[1].axvline(sim_time, ls='--', label=r"Simulation time limit", color='red')

handles, labels = ax[0].get_legend_handles_labels()
handles = handles[::-1]
labels = labels[::-1]

ax[0].legend(handles, labels, fontsize=6, shadow=True, ncol=2,
             loc='upper left')
ax[1].legend(fontsize=6, shadow=True, loc='center right')

# ax[1].semilogx()
ax[0].set_xlabel('|z| (m)')
ax[0].set_ylabel(r'ECDF: $P(x \leq |z|)$')
ax[0].set_title('Depth of Fragmentation into NPs')

ax[2].set_xlabel(r'$X$ (km)')
ax[2].set_ylabel(r'ECDF: $P(x \leq X)$')
ax[2].set_title('Total Displacement as NPs')

ax[1].set_xlabel(r'$T_{NP}$ (days)')
ax[1].set_ylabel(r'ECDF: $P(x \leq T_{NP})$')
ax[1].set_title('Drift Time as NPs')

gridy = np.linspace(0, 1, 11)
ax[0].set_yticks(gridy)
ax[1].set_yticks(gridy)
ax[2].set_yticks(gridy)

ax[0].grid()
ax[1].grid()
ax[2].grid()

fig.savefig('../article_figs/ECDF_nanoparticles', dpi=300,
            facecolor=(1, 0, 0, 0))


# %% Maps of fragmenting location
marker = itertools.cycle(('v', 'h', 'd', 'o', 'X', 'P', '^', 's'))
fig, ax = funk.bathymetry_plot(alpha=0.1)

for j, ft in enumerate(simulations[::-1]):
    ax.scatter(frag_into_NPs[ft]['lon'], frag_into_NPs[ft]['lat'], zorder=2,
               s=20,
               label=f"$\lambda_f$ = {ft} days",
               marker=next(marker))

ax.scatter(origin[0], origin[1], zorder=5,
           label='Sampling Location', marker='*', s=80)

for r in range(7):
    circle_points = geodesic.Geodesic().circle(lon=origin[0], lat=origin[1],
                                               radius=r*1e6,
                                               n_samples=100,
                                               endpoint=False)
    geom = shapely.geometry.Polygon(circle_points)
    ax.add_geometries((geom,), crs=ccrs.PlateCarree(), facecolor='none',
                      edgecolor='black', linewidth=1., zorder=3, ls='--')

ax.set_title('Where do the particles fragment into Nanoparticles?')
handles, labels = ax.get_legend_handles_labels()
handles = handles[::-1]
labels = labels[::-1]

ax.legend(handles, labels, shadow=True)

fig.savefig('../article_figs/Map_location_fragmentation_into_NPs.png', dpi=300,
            facecolor=(1, 0, 0, 0))
