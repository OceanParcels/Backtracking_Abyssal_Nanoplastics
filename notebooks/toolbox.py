import numpy as np
import random
import string

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


def average_parcels_output(array, window=30, normalized=True):
    nx, nt = array.shape

    new_t_dim = nt//window
    averaged = np.zeros((nx, new_t_dim, ))
    time_array = np.array(range(1, new_t_dim))

    for t in range(0, new_t_dim):
        index_slice = slice((t)*window, (t+1)*window)
        mean_aux = np.mean(array[:, index_slice], axis=1)
        averaged[t] = mean_aux

    return averaged, time_array*window


CHARACTERS = (
    string.ascii_uppercase
    + string.digits
)

def generate_unique_key():
    return ''.join(random.sample(CHARACTERS, 6))