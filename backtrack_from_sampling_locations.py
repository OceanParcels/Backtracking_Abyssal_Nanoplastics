"""
python3 backtrack_from_sampling_locations.py frag_ timescale
"""
# %%
from glob import glob
import numpy as np
from parcels import FieldSet, ParticleSet
from parcels import ErrorCode, Field
from parcels.application_kernels.TEOSseawaterdensity import PolyTEOS10_bsq
from datetime import timedelta
from datetime import datetime
import kernels_simple
import sys
from tqdm import tqdm
import xarray as xr

###############################################################################
# %%Setting up all parameters for simulation
###############################################################################

# Control Panel for Kernels
Test_run = True
frag_timescale = int(sys.argv[1])

# Initial conditions
initial_depth = 5100 #int(sys.argv[1])  # 5 # 60 # 5179
lon_sample = 6.287 #6.25
lat_sample = -32.171 #-32.171
start_time = datetime.strptime('2020-01-30 12:00:00', '%Y-%m-%d %H:%M:%S')

# Particle Size and Density
particle_diameter = 5e-08  # meters
initial_particle_density = 1380  # PET kg/m3

data_path = '/storage/shared/oceanparcels/input_data/MOi/psy4v3r1/'
###############################################################################
# %%
###############################################################################
if Test_run:
    # Number of particles and simulation time
    n_points = 100
    sim_time = 12  # days backwards
    file_range = range(19, 20)
    output_path = '/storage/shared/oceanparcels/output_data/' + \
                    f'data_Claudio/tests/3d-b.zarr'
    
    wfiles = sorted(glob(data_path+'psy4v3r1-daily_W_2020-01-*.nc'))
    chunking_express = 1
else:
    # Number of particles and simulation time
    n_points = 10000
    sim_time = 4855 #10*365  # days backwards
    file_range = range(6, 21)
    output_path = '/storage/shared/oceanparcels/output_data/' + \
        f'data_Claudio/set_19_errata/set19_e_{frag_timescale}.zarr'
    chunking_express = 8

    wfiles = []
    for i in tqdm(file_range):
        wfiles = wfiles + sorted(glob(data_path + f'psy4v3r1-daily_W_20{i:02d}*.nc'))

###############################################################################
# %%Reading files #
###############################################################################
vfiles = [f.replace('_W_', '_V_') for f in wfiles]
ufiles = [f.replace('_W_', '_U_') for f in wfiles]
tfiles = [f.replace('_W_', '_T_') for f in wfiles]
sfiles = [f.replace('_W_', '_S_') for f in wfiles]
twoDfiles = [f.replace('_W_', '_2D_') for f in wfiles]
KZfiles = [f.replace('_W_', '_KZ_') for f in wfiles]


mesh_mask = '/storage/shared/oceanparcels/input_data/MOi/' + \
            'domain_ORCA0083-N006/coordinates.nc'
bathy_file = '/storage/shared/oceanparcels/input_data/MOi/' + \
    'domain_ORCA0083-N006/bathymetry_ORCA12_V3.3.nc'

filenames = {'U': {'lon': mesh_mask,
                   'lat': mesh_mask,
                   'depth': wfiles[0],
                   'data': ufiles},
             'V': {'lon': mesh_mask,
                   'lat': mesh_mask,
                   'depth': wfiles[0],
                   'data': vfiles},
             'W': {'lon': mesh_mask,
                   'lat': mesh_mask,
                   'depth': wfiles[0],
                   'data': wfiles},
            'cons_temperature': {'lon': mesh_mask,
                                 'lat': mesh_mask,
                                 'depth': wfiles[0],
                                 'data': tfiles},
            'abs_salinity': {'lon': mesh_mask,
                             'lat': mesh_mask,
                             'depth': wfiles[0],
                             'data': sfiles},
            'mld': {'lon': mesh_mask,
                    'lat': mesh_mask,
                    'depth': twoDfiles[0],
                    'data': twoDfiles},
            'Kz': {'lon': mesh_mask,
                   'lat': mesh_mask,
                   'depth': wfiles[0],
                   'data': KZfiles}}     

variables = {'U': 'vozocrtx',
             'V': 'vomecrty',
             'W': 'vovecrtz',
             'cons_temperature': 'votemper',
            'abs_salinity': 'vosaline',
            'mld': 'somxlavt',
            'Kz': 'votkeavt'}

dimensions = {'U': {'lon': 'glamf',
                    'lat': 'gphif',
                    'depth': 'depthw',
                    'time': 'time_counter'},
              'V': {'lon': 'glamf',
                    'lat': 'gphif',
                    'depth': 'depthw',
                    'time': 'time_counter'},
              'W': {'lon': 'glamf',
                    'lat': 'gphif',
                    'depth': 'depthw',
                    'time': 'time_counter'},
               'cons_temperature': {'lon': 'glamf',
                                  'lat': 'gphif',
                                  'depth': 'depthw',
                                  'time': 'time_counter'},

                 'abs_salinity': {'lon': 'glamf',
                              'lat': 'gphif',
                              'depth': 'depthw',
                              'time': 'time_counter'},
                 'mld': {'lon': 'glamf',
                            'lat': 'gphif',
                            'depth': 'deptht',
                            'time': 'time_counter'},
                 'Kz': {'lon': 'glamf',
                    'lat': 'gphif',
                    'depth': 'depthw',
                    'time': 'time_counter'}}


###############################################################################
# %%Fieldset #
###############################################################################

indices = {'lat': range(500, 1700), 'lon': range(2600, 4321)}
# indices = {'lat': range(0, 1700), 'lon': range(2000, 4321)}

fieldset = FieldSet.from_nemo(filenames, variables, dimensions,
                              allow_time_extrapolation=False,
                              indices=indices,
                              chunksize=False)



bathy = xr.load_dataset(bathy_file)
fieldset.add_field(Field('bathymetry', bathy['Bathymetry'].values,
                         lon=bathy['nav_lon'].values,
                         lat=bathy['nav_lat'].values,
                         mesh='spherical'))

fieldset.add_constant('fragmentation_timescale', frag_timescale) 
###############################################################################
# %%Particle Set #
###############################################################################

np.random.seed(0)
lon_cluster = [lon_sample]*n_points + np.random.normal(loc=0, scale=0.01, size=n_points)
lat_cluster = [lat_sample]*n_points + np.random.normal(loc=0, scale=0.01, size=n_points)
lon_cluster = np.array(lon_cluster) 
lat_cluster = np.array(lat_cluster)

depth_cluster = np.ones(n_points)*initial_depth  # meters
date_cluster = [start_time]*n_points
initial_diameters = np.zeros_like(lon_cluster) + particle_diameter + (1 - np.random.random(len(lon_cluster)))*1e-9
initial_densities = np.zeros_like(lon_cluster) + initial_particle_density

pset = ParticleSet.from_list(fieldset=fieldset, pclass=kernels_simple.PlasticParticle,
                             lon=lon_cluster,
                             lat=lat_cluster,
                             depth=depth_cluster,
                             time=date_cluster,
                            diameter=initial_diameters,
                            particle_density=initial_densities)

###############################################################################
# %%Kernels #
###############################################################################
# Sampling first timestep
sample_kernel = pset.Kernel(kernels_simple.SampleField)
pset.execute(sample_kernel, dt=0)
pset.execute(pset.Kernel(PolyTEOS10_bsq), dt=0)
sinking_kernel = pset.Kernel(kernels_simple.SinkingVelocity)
pset.execute(sinking_kernel, dt=0)

# Loading kernels
kernels = sample_kernel + pset.Kernel(PolyTEOS10_bsq)
kernels += pset.Kernel(kernels_simple.AdvectionRK4_3D)
kernels += pset.Kernel(kernels_simple.BrownianMotion2D)
kernels += pset.Kernel(kernels_simple.VerticalRandomWalk)
kernels += pset.Kernel(kernels_simple.Fragmentation)
kernels += sinking_kernel
kernels += pset.Kernel(kernels_simple.reflectiveBC)
kernels += pset.Kernel(kernels_simple.In_MixedLayer)

print('Kernels loaded')

# Output file
output_file = pset.ParticleFile(name=output_path,
                                outputdt=timedelta(hours=24),
                               chunks=(n_points, chunking_express))

pset.execute(kernels,
             output_file=output_file,
             runtime=timedelta(days=sim_time),
             dt=-timedelta(hours=1),
             recovery={ErrorCode.ErrorOutOfBounds: kernels_simple.delete_particle})

output_file.close()
