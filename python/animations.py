
Skip to content
Pull requests
Issues
Marketplace
Explore
@cpierard
OceanParcels /
BayesianAnalysis_SouthAtlantic
Public

Code
Issues
Pull requests
Actions
Projects
Wiki
Security

    Insights

BayesianAnalysis_SouthAtlantic/python/animations.py /
@cpierard
cpierard make a plot in a notebook and summit animation script
Latest commit e8805ef on 16 Jun 2021
History
1 contributor
98 lines (74 sloc) 3.31 KB
"""very slow but it will make the animation :|
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from datetime import timedelta as delta
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from matplotlib.lines import Line2D
import matplotlib.cm as cm
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation

series = 6
river_sources = np.load('../river_sources.npy', allow_pickle=True).item()
simulations = {}
particle_age = False

print('* Reading data')
for loc in river_sources.keys():
    simulations[loc] = xr.load_dataset(
        f'../data/simulations/sa-s{series:02d}/sa-s{series:02d}-{loc}.nc')

outputdt = delta(hours=24)
timerange = np.arange(np.nanmin(simulations['Congo']['time'].values),
                      np.nanmax(simulations['Congo']['time'].values) +
                      np.timedelta64(outputdt),
                      outputdt)

date_range = pd.date_range('2016-04-01', '2020-08-18')  # easier to print dates

###############################################################################
# Real time animations
###############################################################################
fig, ax = plt.subplots(figsize=(10, 5),
                       subplot_kw={'projection': ccrs.PlateCarree()})

gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=1, color='black', alpha=0.3, linestyle='--')
gl.top_labels = False
gl.right_labels = False

ax.set_extent([-73, 25, -55, 0], crs=ccrs.PlateCarree())

ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.BORDERS, linestyle=':')

x_colors = np.linspace(0, 1, 10)
colors = cm.get_cmap('tab10')(x_colors)
handles = []

for j, loc in enumerate(river_sources.keys()):
    time_id = np.where(simulations[loc]['time'] == timerange[0])
    scatter = ax.scatter(simulations[loc]['lon'].values[time_id],
                         simulations[loc]['lat'].values[time_id],
                         s=0.1, color=colors[j])
    handles.append(scatter)


legend_elements = [Line2D([0], [0], marker='o', color='w',
                          label=loc, markersize=7,
                          markerfacecolor=colors[j]) for j,
                   loc in enumerate(river_sources.keys())]

ax.legend(handles=legend_elements, loc='upper left', fontsize=8)

date_str = str(date_range[0].date())
title = ax.set_title(f'Date: {date_str}')


def animate_realtime(i):
    print(i)
    date_str = str(date_range[i].date())
    ax.set_title(f'Date: {date_str}')

    for j, loc in enumerate(river_sources.keys()):
        time_id = np.where(simulations[loc]['time'] == timerange[i*10])
        handles[j].set_offsets(np.c_[simulations[loc]['lon'].values[time_id],
                                     simulations[loc]['lat'].values[time_id]])


anim_realtime = FuncAnimation(fig, animate_realtime, frames=160)

# Set up formatting for the movie files
Writer = animation.writers['ffmpeg']
writer = Writer(fps=15, metadata=dict(artist='Claudio Pierard'),
                bitrate=3000)

print('* Writing animation')
anim_realtime.save(f'../animations/realtime_sa-s{series:02d}_3.mp4',
                   writer=writer)  # this sux

###############################################################################
# Particle age animations
###############################################################################

    © 2022 GitHub, Inc.

    Terms
    Privacy
    Security
    Status
    Docs
    Contact GitHub
    Pricing
    API
    Training
    Blog
    About

Loading complete