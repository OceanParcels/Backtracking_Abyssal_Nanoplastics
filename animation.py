import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr
from scipy import interpolate
import matplotlib.cm as cm

sim = xr.load_dataset('../data/simulations/backtrack_loc0_column.nc')

frame = 0
n_day = 800

for n_day in range(0, 180, 10):

    fig = plt.figure()
    gs = fig.add_gridspec(3, 2,  width_ratios=(6.1, 3), height_ratios=(2, 3, 1),
                          left=0.1, right=0.9, bottom=0.1, top=0.9,
                          wspace=0.07, hspace=0.07)

    ax = plt.subplot(gs[1, 0], projection=ccrs.PlateCarree())
    ax.set_extent((-45, 46, -52, -16))
    ax.add_feature(cfeature.LAND, facecolor='#808080')
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=0.5, color='gray', alpha=0.5)
    gl.right_labels = False
    gl.top_labels = False

    ax_lon = plt.subplot(gs[0, 0], sharex=ax)
    ax_lon.grid()
    ax_lon.set_xticklabels([])
    ax_lon.set_ylabel('Depth (m)')
    ax_lat = plt.subplot(gs[1, 1], sharey=ax)
    ax_lat.grid()
    ax_lat.set_yticklabels([])
    ax_lat.set_xlabel('Depth (m)')
    ax_bar = plt.subplot(gs[2, 0])
    ax_bar.axis('off')

    ax_void = plt.subplot(gs[0, 1], sharey=ax)
    ax_void.axis('off')
    ax_lon.text(58, -3000, f'Particle age \n -{n_day} days')

    size_point = 0.7

    im = ax.scatter(sim['lon'][:, n_day], sim['lat'][:, n_day], c=-sim['z'][:, 0], s=size_point)
    ax_lon.scatter(sim['lon'][:, n_day], -sim['z'][:, n_day], c=-sim['z'][:, 0], s=size_point)
    ax_lat.scatter(-sim['z'][:, n_day], sim['lat'][:, n_day], c=-sim['z'][:, 0], s=size_point)

    bar_ax = fig.add_axes([0.12, 0.12, 0.52, 0.05])
    plt.colorbar(im, cax=bar_ax, orientation='horizontal', label='Final depth (m)')

    plt.savefig(f'../figs/anim01/frame{frame:03d}', dpi=200, facecolor=(1, 1, 1, 1))
    plt.close()
    frame += 1
