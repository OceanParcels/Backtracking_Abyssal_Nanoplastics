import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from glob import glob
from datetime import datetime
from tqdm import tqdm

location = 'HC11'
if location == 'HC13':
    start_time = datetime.strptime('2019-01-20 12:00:00', '%Y-%m-%d %H:%M:%S')
    lat_obs = -32.171
    lon_obs = 6.287

elif location == 'HC11':
    start_time = datetime.strptime('2019-01-16 12:00:00', '%Y-%m-%d %H:%M:%S')
    lon_obs = -3.822
    lat_obs = -29.992

path_flow = '/storage/shared/oceanparcels/input_data/MOi/psy4v3r1/'
files = sorted(glob(path_flow + 'psy4v3r1-daily_KZ_*.nc'))

end_time = datetime.strptime('2006-10-11 12:00:00', '%Y-%m-%d %H:%M:%S')
start_index = 0 
end_index = 0

for file in files:
    if file[-13:-3] == start_time.strftime('%Y-%m-%d'):
        end_index = files.index(file)
        
    if file[-13:-3] == end_time.strftime('%Y-%m-%d'):
        start_index = files.index(file)
    
files = files[start_index:end_index+1]

mesh_mask = xr.open_dataset(files[0])
z_depths = mesh_mask['depthw'].values

lon_rest = abs(mesh_mask['nav_lon'][0,:] - lon_obs)
lat_rest = abs(mesh_mask['nav_lat'][:,0] - lat_obs)
lon_index = np.where(lon_rest == lon_rest.min())[0][0]
lat_index = np.where(lat_rest == lat_rest.min())[0][0]

flow = xr.open_dataset(files[10])

d_deg = 5

indices = {'lat': range(lat_index-d_deg, lat_index+d_deg), 
           'lon': range(lon_index-d_deg, lon_index+d_deg)}
flow['votkeavt'][44, indices['lat'], indices['lon']].plot()

Kz = np.zeros((len(files), 50, len(indices['lat']), len(indices['lon'])))
# Kz_std = np.zeros((len(files), 50))

time = np.zeros(len(files),dtype='datetime64[ns]')

for i, filename in enumerate(tqdm(files)):
    T = xr.open_dataset(filename)
    Kz[i] = T['votkeavt'][:, indices['lat'], indices['lon']].values
    
    time[i] = T['time_counter'].values
    
Kz_mean = np.nanmean(Kz, axis=(0,2,3))
Kz_std = np.nanstd(Kz, axis=(0,2,3))
Kz_median = np.nanmedian(Kz, axis=(0,2,3))

np.save(f'../data/Kz_profile_{location}.npy', Kz)

fig, ax = plt.subplots(figsize=(3,4))
ax.plot(Kz_mean, -z_depths, color='k', label='mean')
ax.plot(Kz_std, -z_depths, label='std', ls=':', color='k')
ax.plot(Kz_median, -z_depths, label='median', ls='--', color='k')
ax.semilogx()
ax.legend(shadow=True, fancybox=True, fontsize=8)
ax.set_xlabel('$K_z$ [m$^2$/s]')
ax.set_ylabel('Depth [m]')
ax.set_title('HC13')
ax.set_xticks([1e-7,1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
fig.savefig(f'../article_figs/Kz_profile_{location}.png', dpi=300,
            facecolor=(1, 0, 0, 0))