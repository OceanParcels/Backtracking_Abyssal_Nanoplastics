#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 15:24:39 2023

@author: claudio pierard
"""
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
import cmocean.cm as cmo
import seaborn as sns
# from matplotlib.gridspec import GridSpec

import cartopy
from cartopy import geodesic 
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import shapely


def ecdf(a, normalized=True, invert=False):
    x, counts = np.unique(a, return_counts=True)
    
    x = np.insert(x, 0, x[0])
    
    if invert:
        cusum = np.cumsum(counts[::-1])
        x = x[::-1]
        
    else:
        cusum = np.cumsum(counts)
        
    cusum = np.insert(cusum, 0, 0.)
    if normalized==False:
        return x, cusum
    else:
        return x, cusum/cusum[-1]


def filter_trajectories(data, condition):
    k, _ = np.where(condition)
    index = np.unique(k)

    data_relevant = data.where(data['trajectory'].isin(index), drop=True)

    return data_relevant


def haversine(coord1: object, coord2: object):

    # Coordinates in decimal degrees (e.g. 2.89078, 12.79797)
    lon1, lat1 = coord1
    lon2, lat2 = coord2

    R = 6371000  # radius of Earth in meters
    phi_1 = np.radians(lat1)
    phi_2 = np.radians(lat2)

    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    a = np.sin(delta_phi/2.0)**2 
    a += np.cos(phi_1)*np.cos(phi_2)*np.sin(delta_lambda/2.0)**2
    
    c = 2*np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    meters = R*c  # output distance in meters
    
    return meters


def ridge_plot(data, xlabel ='', title='', bins=128, h_space=-0.5, alpha=1, 
               figsize=(8,8), cmap='tab10'):
    """
        ridge_plot(data, xlabel, bins=128, h_space=-0.5, alpha=1, 
                   figsize=(8,6))
    Plots a comparison of kernel density estimates (KDE) for a diferent 
    groups of data.
    data : a dictionary with a 1D series per key/set (unlimited number of 
                                                      keys/sets).
    xlabel : the string that contains the label of the plot.
    title : the title for the plot.
    bins : the number of points to plot the computed KDE.
    h_space : the separation between distributions, it should be negative for 
    them to overlap.
    alpha : the transparency.
    figsize : the figure size.
    cmap : the colormap. Use the predefined Matplotlib colormaps.
    """
    nrows = len(data.keys())
    labels = list(data.keys())
    x_colors = np.linspace(0,1, nrows)
    colors = cm.get_cmap(cmap)(x_colors)
    fig, axes = plt.subplots(nrows,  sharex=True, figsize=figsize)
    min_glob = 999

    for i, key in enumerate(data.keys()):
        val_min = data[key][~np.isnan(data[key])].min()
        val_max = data[key][~np.isnan(data[key])].max()

        if val_min < min_glob:
            min_glob = val_min

        # x_values = np.linspace(val_min, val_max, bins)
        c = colors[i]
        axes[i].hist(data[key], bins=bins, alpha=alpha, color=c)
        # kernel = stats.gaussian_kde(data[key][~np.isnan(data[key])])
        # kde = kernel(x_values)
        # axes[i].plot(x_values, kde, color="#f0f0f0", lw=1)
        # axes[i].fill_between(x_values, kde, color=c, alpha=alpha)
        rect = axes[i].patch
        rect.set_alpha(0)
        axes[i].tick_params(left=False, labelleft=False)
        axes[0].set_title(title)

        if i == len(data.keys())-1:
            axes[i].tick_params(bottom=True, left=False, labelleft=False)
            spines = ["top","right","left"]
            #axes[i].set_ylim(-0.05,)
            axes[i].set_xlim(min_glob,)
            axes[i].set_xlabel(xlabel)

        else:
            axes[i].tick_params(bottom=False, left=False, labelleft=False)
            spines = ["top","right","left","bottom"]

        for s in spines:
            axes[i].spines[s].set_visible(False)

        depth_label = str(int(key))

    for j,l in enumerate(data.keys()):
        axes[j].text(min_glob, 0., labels[j], fontsize=8, ha="right")

    plt.subplots_adjust(hspace=h_space)
    return fig, axes


# ## Import data for bathymetry plots ###
shp_dict = {}
files = glob('../data/ne_10m_bathymetry_all/*.shp')
assert len(files) > 0
files.sort()
for f in files:
    depth = f.split('_')[-1].split('.')[0]
    # depth = '-' + f.split('_')[-1].split('.')[0]
    # depths.append(depth)
    nei = cartopy.io.shapereader.Reader(f)
    shp_dict[depth] = nei

depths_bathy = [d for d in shp_dict.keys()][::-1]
colors_bathy = sns.mpl_palette('cmo.ice_r', n_colors=8)
cmap_bathy = sns.mpl_palette('cmo.ice', n_colors=8, as_cmap=True)


def bathymetry_plot(figsize=(13, 7),alpha=1., ):

    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection=ccrs.PlateCarree())
    # ax.set_extent((-5, 20, -40, -25), crs=ccrs.PlateCarree())

    i = 0
    for depth in depths_bathy[:8]:
        ax.add_geometries(shp_dict[depth].geometries(),
                          crs=ccrs.PlateCarree(), color=colors_bathy[i], 
                          alpha=alpha)
        i += 1

    ax.add_feature(cartopy.feature.NaturalEarthFeature(category='physical',
                                                       name='land', 
                                                       scale='110m'),
                                                       color='black')

    gl = ax.gridlines(draw_labels=True)
    gl.right_labels = False
    gl.top_labels = False
    

    # Add custom colorbar
    axi = fig.add_axes([0.910, 0.35, 0.025, 0.3])
    # axi = fig.add_axes([0.8,0.2,0.025,0.6])
    norm = matplotlib.colors.Normalize(vmin=-6000, vmax=0)

    boundaries_bathy = (-np.array(depths_bathy[:8]).astype(int)).tolist()[::-1]
    ticks_bathy = -np.array(depths_bathy).astype(int)
    matplotlib.colorbar.ColorbarBase(ax=axi, cmap=cmap_bathy, norm=norm,
                                            boundaries=boundaries_bathy,
                                            ticks=ticks_bathy,
                                            spacing='proportional',
                                            extend='neither',
                                            label='Depth (m)')

    return fig, ax


def bathymetry_subplots(nrows=2,ncols=1, figsize=(13, 7),alpha=1., ):

    fig, ax = plt.subplots(nrows=2,ncols=1,
                            subplot_kw={'projection': ccrs.PlateCarree()},
                            figsize=figsize, sharey=True)
    
    for k in range(nrows*ncols):
        i = 0
        for depth in depths_bathy[:8]:
            ax[k].add_geometries(shp_dict[depth].geometries(),
                              crs=ccrs.PlateCarree(), color=colors_bathy[i], 
                              alpha=alpha)
            i += 1
    
        ax[k].add_feature(cartopy.feature.NaturalEarthFeature(category='physical',
                                                           name='land', 
                                                           scale='110m'),
                                                          color='black')
    
        


    # Add custom colorbar
    axi = fig.add_axes([0.910, 0.35, 0.025, 0.3])
    # axi = fig.add_axes([0.8,0.2,0.025,0.6])
    norm = matplotlib.colors.Normalize(vmin=-6000, vmax=0)

    boundaries_bathy = (-np.array(depths_bathy[:8]).astype(int)).tolist()[::-1]
    ticks_bathy = -np.array(depths_bathy).astype(int)
    matplotlib.colorbar.ColorbarBase(ax=axi, cmap=cmap_bathy, norm=norm,
                                            boundaries=boundaries_bathy,
                                            ticks=ticks_bathy,
                                            spacing='proportional',
                                            extend='neither',
                                            label='Depth (m)')

    return fig, ax
