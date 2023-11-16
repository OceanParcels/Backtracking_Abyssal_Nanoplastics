import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cmocean
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation, writers, PillowWriter, FFMpegWriter
from glob import glob
from datetime import datetime
from tqdm import tqdm
import matplotlib.ticker as mticker

# Get particle simulations and reverse time

pset = xr.open_zarr('/storage/shared/oceanparcels/output_data/data_Claudio/hc13_3/hc13_500.zarr')
pset = pset.compute()
pset = pset.where(pset['z'] > 10, drop=True)
pset = pset.reindex(obs = pset.obs[::-1])

# check start time and define it
idx = np.where(np.isnat(pset['time'][:,0].values) == False)[0][0]
aux = pset['time'][idx, 0].values
start_time = datetime.utcfromtimestamp(int(aux)/1e9)

# Mixed layer depth data
path_flow = '/storage/shared/oceanparcels/input_data/MOi/psy4v3r1/'
files = sorted(glob(path_flow + 'psy4v3r1-daily_2D_*.nc'))

end_time = datetime.strptime('2019-01-16 12:00:00', '%Y-%m-%d %H:%M:%S')
start_index = 0 
end_index = 0

for file in files:
    if file[-13:-3] == start_time.strftime('%Y-%m-%d'):
        start_index = files.index(file)
        
    if file[-13:-3] == end_time.strftime('%Y-%m-%d'):
        end_index = files.index(file)


files = files[start_index:end_index+1]

mesh_mask = xr.open_dataset('/storage/shared/oceanparcels/input_data/MOi/domain_ORCA0083-N006/coordinates.nc', decode_times=False)

indices = {'lat': range(600, 1350), 'lon': range(3100, 4000)}

fields = np.zeros((len(files), len(indices['lat']), len(indices['lon'])))
time = np.zeros(len(files),dtype='datetime64[ns]')

indices = {'lat': range(600, 1350+1), 'lon': range(3100, 4000+1)}

flow = xr.open_dataset(files[10])
lats = flow['nav_lat'][indices['lat'], indices['lon']].values
lons = flow['nav_lon'][indices['lat'], indices['lon']].values
lats -= lats[0,0] - lats[1,0]
lons -= lons[0,0] - lons[0,1]

indices = {'lat': range(600, 1350), 'lon': range(3100, 4000)}

bathy_moi = xr.load_dataset('/storage/shared/oceanparcels/input_data/MOi/domain_ORCA0083-N006/bathymetry_ORCA12_V3.3.nc')
landmask = bathy_moi['mask'][indices['lat'], indices['lon']]
masked_land = np.ma.masked_where(landmask==1, landmask)

for i, filename in enumerate(tqdm(files)):
    T = xr.open_dataset(filename)
    fields[i] = T['somxlavt'][:, indices['lat'], indices['lon']].values
    time[i] = T['time_counter'].values

# %% define layout animation

fig = plt.figure(figsize=(6,7))

gs = gridspec.GridSpec(2, 1, height_ratios=[6, 2])

t0 = 0

ax0 = plt.subplot(gs[0])
ax1 = plt.subplot(gs[1])
ax1.axis('off')
tracer = ax0.pcolormesh(lons, lats, -fields[t0], cmap='autumn', vmin=-100, vmax=0)

ax0.pcolormesh(lons, lats, masked_land, cmap='Greys_r', vmin=-1, vmax=1)

ax0.set_xlim([lons[0,0], lons[-1,-1]])
ax0.set_ylim([lats[0,0], lats[-1,-1]])
ax0.set_xlabel('Longitude')
ax0.set_ylabel('Latitude')
time_str = str(time[t0])[:10]
title = ax0.set_title(f'Date: {time_str}')

particles = ax0.scatter(pset['lon'][:,t0], pset['lat'][:,t0], s=1, c=-pset['z'][:,t0], 
                        cmap='winter', vmax=0, vmin=-2500, marker='o')

bar_ax0 = fig.add_axes([0.2, 0.25, 0.6, 0.02])
bar_ax1 = fig.add_axes([0.2, 0.15, 0.6, 0.02])

fig.colorbar(tracer, cax=bar_ax0, orientation='horizontal', 
             label='Mixed Layer Depth [m]', extend='min')
fig.colorbar(particles, cax=bar_ax1, orientation='horizontal', 
             label='Particle Depth [m]', extend='min')


def animate(i):
    time_str = str(time[i])[:10]
    title.set_text(f'Date: {time_str}')
    
    tracer.set_array(-fields[i].ravel())
    particles.set_offsets(np.c_[pset['lon'][:,i], pset['lat'][:,i]])
    z = -pset['z'][:,i].values
    particles.set_array(z.ravel())
    return tracer, particles, title


anim = FuncAnimation(fig, animate, frames=4480 , interval=100, blit=True, repeat=True)

writergif = PillowWriter(fps=30, codec="libx264")
anim.save(f'../article_figs/Forward_hc11_500_MLD.gif', writer=writergif)