# %%
import numpy as np
import xarray as xr
import itertools
import matplotlib.pyplot as plt
from cartopy import geodesic
import cartopy.crs as ccrs
import shapely
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import draft_functions as funk
import cartopy.feature as cfeature

# %%
initial_depth = -5100  # int(sys.argv[1])  # 5 # 60 # 5179
lon_sample = 6.287  # 6.25
lat_sample = -32.171  # -32.171
origin = (lon_sample, lat_sample)

start_time = datetime.strptime('2019-01-20 12:00:00', '%Y-%m-%d %H:%M:%S')
sim_time = 4484
# create the date range in reverse from sampling time to the past
datelist = pd.date_range(end=start_time, periods=sim_time)[::-1]
end_time = datelist[0]

simulations = [10] + [i for i in range(100, 501, 100)]
simulations += [1000, 10000]

# %% Extracting the data from simulation outputs
surface_events = {}


for ft in tqdm(simulations):
    # print('Computing fragmentation timescale: ', ft)
    local_path = f'/storage/shared/oceanparcels/output_data/data_Claudio/hc13_3/hc13_{ft}.zarr'
    sim = xr.open_zarr(local_path)

    # loading fields as np arrays to do some nasty indexing
    radiuses = sim['radius'].values
    depths = sim['z'].values
    latitudes = sim['lat'].values
    longitudes = sim['lon'].values

    df_sim = {}  # dictiorany to store data per sim. Could be a pandas DF

    # Detecting the index and the particles that reach the surface
    k1, k2 = np.where(depths < 10)
    idx_particles, idx_k2 = np.unique(k1, return_index=True)
    surface_time = k2[idx_k2]

    df_sim['particle_index'] = idx_particles
    df_sim['surface_time'] = surface_time  # Days. equivalent to index in simus
    df_sim['radius'] = radiuses[idx_particles, surface_time]
    df_sim['depths'] = depths[idx_particles, surface_time]

    latz = latitudes[idx_particles, surface_time]
    lonz = longitudes[idx_particles, surface_time]
    xy_pos = (lonz, latz)

    df_sim['displacement'] = funk.haversine(origin, xy_pos)
    df_sim['lat'] = latz
    df_sim['lon'] = lonz

    surface_events[ft] = df_sim  # Storing the dictionaris in another dict


np.save('../data/surface_events.npy', surface_events)

# %% greate a pandas dataframe with the mean and std of the size distribution
# at the surface

df = pd.DataFrame(columns=['Particles', 'L median', 'L min', 'L max',
                           'T_s mean', 'T_s std', 'T_s median', 'T_s min', 'T_s max',
                            'X mean', 'X std', 'X median', 'X min', 'X max'])

for ft in simulations:
    df.loc[ft] = [len(surface_events[ft]['radius']),
                  np.median(surface_events[ft]['radius']),
                  surface_events[ft]['radius'].min(),
                  surface_events[ft]['radius'].max(),
                  surface_events[ft]['surface_time'].mean(),
                  surface_events[ft]['surface_time'].std(),
                  np.median(surface_events[ft]['surface_time']),
                  surface_events[ft]['surface_time'].min(),
                  surface_events[ft]['surface_time'].max(),
                  surface_events[ft]['displacement'].mean(),
                  surface_events[ft]['displacement'].std(),
                  np.median(surface_events[ft]['displacement']),
                  surface_events[ft]['displacement'].min(),
                  surface_events[ft]['displacement'].max()]

    

df.to_csv('../data/size_distribution_surface.csv')
df.to_latex('../article_figs/surface_events_numbers.tex') # to print in latex format and save in a file

# %% ecdf surfacetime and size distribution of particles at the surface

def forward(x):
    return x**(1/2)

def inverse(x):
    return x**2

fig, ax = plt.subplots(1, 3, figsize=(12, 3.5), tight_layout=True)

ax[0].axvline(1e-6, ls=':', color='black')
ax[0].axvline(1e-4, ls=':', label=r"Fragmentation limit", color='red')
ax[1].axvline(sim_time, ls=':', label=r"Simulation time limit", color='black')
ax[1].text(4300, 0.19, r'Time Limit', fontsize=6, color='k', rotation=-90)
ax[0].text(1e-6, 0.19, r"1 $\mu m$ Limit", fontsize=6, color='k', rotation=-90)

for j, ft in enumerate(simulations[::-1]):

    x, y = funk.ecdf(surface_events[ft]['radius'], normalized=True)
    ax[0].plot(x, y, drawstyle='steps-post')

    x, y = funk.ecdf(surface_events[ft]['surface_time'], normalized=True)
    ax[1].plot(x, y, drawstyle='steps-post')

    x, y = funk.ecdf(surface_events[ft]['displacement']/1e3, normalized=True)
    ax[2].plot(x, y, drawstyle='steps-post', label=f'$\lambda_f$ = {ft} days')    


ax[2].set_xscale('function', functions=(forward, inverse))

handles, labels = ax[2].get_legend_handles_labels()
handles = handles[::-1]
labels = labels[::-1]

ax[0].legend(fontsize=7, shadow=True)
ax[2].legend(handles, labels, fontsize=7, shadow=True, loc='center right')
ax[0].semilogx()
ax[0].set_xlabel('Surface Particle Radius, $R$ [m]')
ax[0].set_ylabel(r'ECDF: $P(x \leq R)$')
# ax[0].set_title('Particle Radius from Surface')

ax[1].set_xlabel(r'Surface Drift Time, $T_s$ [days]')
ax[1].set_ylabel(r'ECDF: $P(x \leq T_s)$')
# ax[1].set_title('Drift Time from Surface')

ax[2].set_xlabel(r'Displacement from Surface, $X$ [km]')
ax[2].set_ylabel(r'ECDF: $P(x \leq X)$')
# ax[2].set_title('Displacement from Surface')

gridy = np.linspace(0, 1, 11)
gridx = [500, 1000] + [i for i in range(2000, 10000, 2000)]

ax[0].set_yticks(gridy)
ax[1].set_yticks(gridy)
ax[2].set_yticks(gridy)
ax[2].set_xticks(gridx)

ax[0].grid()
ax[1].grid()
ax[2].grid()

ax[0].text(1e-3, 0, r'A', fontsize=12,
               ha='right')
ax[1].text(4400, 0, r'B', fontsize=12,
               ha='right')
ax[2].text(6800, 0, r'C', fontsize=12,
               ha='right')

fig.savefig('../article_figs/ECDF_surface', dpi=300,
            facecolor=(1, 0, 0, 0))


# %% Supporting map of the distributions
marker = itertools.cycle(('v', 'h', 'd', 'o', 'X', 'P', '^', 's'))

fig,ax = plt.subplots(figsize=(10,8),
                      subplot_kw={'projection': ccrs.PlateCarree()}, constrained_layout=True)

gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=1, color='black', alpha=0.3, linestyle=':')
gl.top_labels = False
gl.right_labels = False
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=0.5, color='black', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False

# ax.set_extent([0, 40,-60.916664, -20], crs=ccrs.PlateCarree())

# ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.LAND, zorder=1, color='black')
# ax.add_feature(cfeature.COASTLINE)

for j, ft in enumerate(simulations[::-1]):
    ax.scatter(surface_events[ft]['lon'], surface_events[ft]['lat'],
               zorder=2, s=20,
               label=f"$\lambda_f$ = {ft} days", marker=next(marker))

ax.scatter(origin[0], origin[1], zorder=5,
           label='Sampling Location', marker='*', s=100, edgecolors='black')

for r in range(1, 10):
    circle_points = geodesic.Geodesic().circle(lon=origin[0], lat=origin[1],
                                               radius=r*1e6,
                                               n_samples=360,
                                               endpoint=False)
    geom = shapely.geometry.Polygon(circle_points)
    ax.add_geometries((geom,), crs=ccrs.PlateCarree(), facecolor='none',
                      edgecolor='black', linewidth=1., zorder=3, ls='--',
                      label=f'{r} km')
    
handles, labels = ax.get_legend_handles_labels()
handles = handles[::-1]
labels = labels[::-1]

ax.text(3.3, -24.5, r"$1,000$ km", fontsize=5)
ax.text(-14.5, -30.5, r"$2,000$ km", fontsize=5, rotation=70)
ax.text(-25., -30.5, r"$3,000$ km", fontsize=5, rotation=70)
ax.text(-35.6, -30.5, r"$4,000$ km", fontsize=5, rotation=70)
ax.text(-46.6, -31.5, r"$5,000$ km", fontsize=5, rotation=70)
ax.text(-63.5, -48.5, r"$6,000$ km", fontsize=5, rotation=75)
ax.text(-79.3, -52.5, r"$7,000$ km", fontsize=5, rotation=62)
ax.text(-89.5, -45.3, r"$8,000$ km", fontsize=5, rotation=53)
ax.text(-89.5, -27.3, r"$9,000$ km", fontsize=5, rotation=55)
ax.set_extent([-97, 65, -63, 0], crs=ccrs.PlateCarree())

ax.legend(handles, labels, ncols=3, fontsize=9, shadow=True) 
fig.savefig('../article_figs/Map_location_surface.png', dpi=300,
            facecolor=(1, 0, 0, 0))

# %% Results ECDF
frag_into_NPs = np.load('../data/frag_into_NPs.npy', allow_pickle=True)[()]

fig, ax = plt.subplots(1, 2, figsize=(8, 3.5), tight_layout=True)

ax[0].axvline(initial_depth, ls=':', color='k')
ax[0].text(initial_depth - 200, 0.65, r'Sampling Depth', fontsize=6, color='k', rotation=-90)
ax[0].axvline(0, ls=':', color='k')
ax[0].text(0-200, 0.65, r'Surface', fontsize=6, color='k', rotation=-90)

ax[1].axvline(1e-6, ls=':', color='black')
ax[1].axvline(1e-4, ls=':', label=r"Fragmentation limit", color='red')
ax[1].text(1e-6, 0.08, r"1 $\mu m$ Limit", fontsize=6, color='k', rotation=-90)
ax[1].text(1.1e-4, 0.01, r"Fragmentation Limit", fontsize=6, color='r', rotation=-90)

for j, ft in enumerate(simulations[::-1]):

    x, y = funk.ecdf(frag_into_NPs[ft]['depths'], normalized=True,
                     invert=False)
    ax[0].plot(x, y, drawstyle='steps-post', label=f'$\lambda_f$ = {ft} days')
    
    x, y = funk.ecdf(frag_into_NPs[ft]['particle_index'], normalized=True,
                     invert=False)
    
    x, y = funk.ecdf(surface_events[ft]['radius'], normalized=True)
    ax[1].plot(x, y, drawstyle='steps-post')


handles, labels = ax[0].get_legend_handles_labels()
handles = handles[::-1]
labels = labels[::-1]

ax[0].legend(handles, labels, fontsize=7, shadow=True, ncol=2,
         loc='best')

ax[1].legend(handles, labels, fontsize=7, shadow=True, ncol=1,
         loc='best')

ax[0].set_xlabel('$R < 1\ \mu m$ Fragmentation Depth, $z$ [m]')
ax[0].set_ylabel(r'ECDF: $P(x \leq z)$')

ax[1].semilogx()
ax[1].set_xlabel('Surface Particles Radius, $R$ [m]')
ax[1].set_ylabel(r'ECDF: $P(x \leq R)$')

gridy = np.linspace(0, 1, 11)
gridx = [500, 1000] + [i for i in range(2000, 10000, 2000)]

ax[0].set_yticks(gridy)
ax[1].set_yticks(gridy)

ax[0].grid()
ax[1].grid()

# ax[0].text(-5500, 0.98, r'A', fontsize=12,
#                ha='right')
# ax[1].text(1e-7, 0.98, r'B', fontsize=12,
#                ha='right')

fig.savefig('../article_figs/ECDF_results2', dpi=300,
            facecolor=(1, 0, 0, 0))