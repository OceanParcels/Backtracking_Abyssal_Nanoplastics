import numpy as np

def stuck_particles_mask(dataset):
    """
    Function that masks the particles that are stuck on the seafloor from xarray
    datasets.
    
    dataset: and xarray dataset.
    """
    time = dataset.dims['obs']
    diff_lon = np.roll(dataset['lon'], axis=1, shift=1) - dataset['lon']
    diff_lat = np.roll(dataset['lat'], axis=1, shift=1) - dataset['lat']
    diff_z = np.roll(dataset['z'], axis=1, shift=1) - dataset['z']
    
    mask = (diff_lon + diff_lat + diff_z)
    mask = ~(mask == 0)
    new_dataset = dataset.where(mask == True)
    
    return new_dataset