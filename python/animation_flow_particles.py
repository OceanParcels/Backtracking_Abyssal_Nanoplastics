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

location = 'hc11'
if location == 'hc13':
    start_time = datetime.strptime('2019-01-20 12:00:00', '%Y-%m-%d %H:%M:%S')
    lat_obs = -32.171
    lon_obs = 6.287

elif location == 'hc11':
    start_time = datetime.strptime('2019-01-16 12:00:00', '%Y-%m-%d %H:%M:%S')
    lon_obs = -3.822
    lat_obs = -29.992

path_flow = '/storage/shared/oceanparcels/input_data/MOi/psy4v3r1/'
files = sorted(glob(path_flow + 'psy4v3r1-daily_T_*.nc'))

end_time = datetime.strptime('2006-10-11 12:00:00', '%Y-%m-%d %H:%M:%S')
start_index = 0 
end_index = 0

for file in files:
    if file[-13:-3] == start_time.strftime('%Y-%m-%d'):
        end_index = files.index(file)
        
    if file[-13:-3] == end_time.strftime('%Y-%m-%d'):
        start_index = files.index(file)
    
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
    fields[i] = T['votemper'][44, indices['lat'], indices['lon']].values
    time[i] = T['time_counter'].values

#reverting fields and time
fields = fields[::-1]
time = time[::-1]

pset = xr.open_zarr('/storage/shared/oceanparcels/output_data/data_Claudio/hc11/hc11_0.zarr')
pset = pset.compute()

fig = plt.figure(figsize=(6,7))

gs = gridspec.GridSpec(2, 1, height_ratios=[6, 2])

t0 = 2000

ax0 = plt.subplot(gs[0])
ax1 = plt.subplot(gs[1])
ax1.axis('off')
tracer = ax0.pcolormesh(lons, lats, fields[t0], cmap='cmo.thermal', vmin=-2, vmax=3)
ax0.pcolormesh(lons, lats, masked_land, cmap='Greys_r', vmin=-1, vmax=1)

ax0.set_xlim([lons[0,0], lons[-1,-1]])
ax0.set_ylim([lats[0,0], lats[-1,-1]])
ax0.set_xlabel('Longitude')
ax0.set_ylabel('Latitude')
time_str = str(time[t0])[:10]
title = ax0.set_title(f'Date: {time_str}')

particles = ax0.scatter(pset['lon'][:,t0], pset['lat'][:,t0], s=1, c= -pset['z'][:,t0], cmap='winter', vmax=-3594, vmin=-3600)

bar_ax0 = fig.add_axes([0.2, 0.25, 0.6, 0.02])
bar_ax1 = fig.add_axes([0.2, 0.15, 0.6, 0.02])
fig.colorbar(tracer, cax=bar_ax0, orientation='horizontal', label='Temperature [deg C] at -3597 m', extend='both')
fig.colorbar(particles, cax=bar_ax1, orientation='horizontal', 
             label='Particle Depth [m]', extend='both', 
             ticks=[-3600, -3597, -3594],  format=mticker.FixedFormatter(['< -3600', '-3597', '> -3594']))

def animate(i):
    time_str = str(time[i])[:10]
    title.set_text(f'Date: {time_str}')
    
    tracer.set_array(fields[i].ravel())
    particles.set_offsets(np.c_[pset['lon'][:,i], pset['lat'][:,i]])
    z = -pset['z'][:,i].values
    particles.set_array(z.ravel())
    return tracer, particles, title


anim = FuncAnimation(fig, animate, frames=4480 , interval=100, blit=True, repeat=True)

writergif = PillowWriter(fps=30, codec="libx264")
anim.save(f'../article_figs/{location}_T3600_inf.gif', writer=writergif)
