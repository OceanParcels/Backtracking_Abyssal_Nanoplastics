"""
To run do:
      python3 backtrack_from_sampling_locations.py _frag_timescale_ _Frag_on_

frag_timescale: a number
Frag_on: True or False
"""
# %%
from glob import glob
import numpy as np
from parcels import FieldSet, ParticleSet
from parcels import ErrorCode, Field
from parcels.application_kernels.TEOSseawaterdensity import PolyTEOS10_bsq
from datetime import timedelta
from datetime import datetime
import kernels_simple as kernels_simple
import sys
from tqdm import tqdm
import xarray as xr

###############################################################################
# %%Setting up all parameters for simulation
###############################################################################

# Control Panel for Kernels
Test_run = False
frag_timescale = int(sys.argv[1])
Frag_on = sys.argv[2]

# Initial conditions
# HC13 depth: 5000 m
# HC11 depth: 4835 m
initial_depth = 4835

# HC13 lat: -32.171, lon: 6.287
# HC11 lat: -29.992, lon: -3.822
lon_sample = -3.822
lat_sample = -29.992

#HC13 date: '2019-01-20 12:00:00'
#HC11 date: '2019-01-16 12:00:00'

start_time = datetime.strptime('2019-01-16 12:00:00', '%Y-%m-%d %H:%M:%S')

# Particle Size and Density
initial_particle_density = 1380  # PET & PVC kg/m3

###############################################################################
# %%
###############################################################################
data_path = '/storage/shared/oceanparcels/input_data/MOi/psy4v3r1/'
wfiles = sorted(glob(data_path + f'psy4v3r1-daily_W_*.nc'))

if Test_run:
    # Number of particles and simulation time
    n_points = 100
    sim_time = 60  # days backwards
    output_path = '/storage/shared/oceanparcels/output_data/' + \
                    f'data_Claudio/tests/optimum.zarr'
    
    wfiles = sorted(glob(data_path+'psy4v3r1-daily_W_2018-11-*.nc'))
    wfiles += sorted(glob(data_path+'psy4v3r1-daily_W_2018-12-*.nc'))
    wfiles += sorted(glob(data_path+'psy4v3r1-daily_W_2019-01-*.nc'))
    chunking_express = 12
    end_time = datetime.strptime('2018-11-20 12:00:00', '%Y-%m-%d %H:%M:%S')
    
else:
    # Number of particles and simulation time
    n_points = 8192 #2^13
    sim_time = 4484
    # From 11 October 2006 to and including 20 January 2019 (forward).
    # Result: 4485 days or 12 years, 3 months, 10 days including the end date.
    end_time = datetime.strptime('2006-10-11 12:00:00', '%Y-%m-%d %H:%M:%S')
    
    file_range = range(6, 21)
    output_path = '/storage/shared/oceanparcels/output_data/' + \
        f'data_Claudio/hc11/hc11_{frag_timescale}.zarr'
    chunking_express = 500

# Loading the only the files that we need.
# indexes are inverted because the start date is in the future.
# it's a backwards in time simulation
start_index = 0 
end_index = 0

for file in wfiles:
    if file[-13:-3] == start_time.strftime('%Y-%m-%d'):
        end_index = wfiles.index(file)
        
    if file[-13:-3] == end_time.strftime('%Y-%m-%d'):
        start_index = wfiles.index(file)
    
wfiles = wfiles[start_index:end_index+1]

###############################################################################
# %%Reading files #
###############################################################################
vfiles = [f.replace('_W_', '_V_') for f in wfiles]
ufiles = [f.replace('_W_', '_U_') for f in wfiles]
tfiles = [f.replace('_W_', '_T_') for f in wfiles]
sfiles = [f.replace('_W_', '_S_') for f in wfiles]
KZfiles = [f.replace('_W_', '_KZ_') for f in wfiles]


mesh_mask = '/storage/shared/oceanparcels/input_data/MOi/' + \
            'domain_ORCA0083-N006/coordinates.nc'

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
            'Kz': {'lon': mesh_mask,
                   'lat': mesh_mask,
                   'depth': wfiles[0],
                   'data': KZfiles}}     

variables = {'U': 'vozocrtx',
             'V': 'vomecrty',
             'W': 'vovecrtz',
             'cons_temperature': 'votemper',
             'abs_salinity': 'vosaline',
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
              'Kz': {'lon': 'glamf',
                    'lat': 'gphif',
                    'depth': 'depthw',
                    'time': 'time_counter'}}


###############################################################################
# %%Fieldset #
###############################################################################

# indices = {'lat': range(0, 1700), 'lon': range(200, 4321)}
indices = {'lat': range(200, 1700), 'lon': range(2300, 4321)} # whole domain for frag timescale < 400
# indices = {'lat': range(0, 1700)} # whole domain for frag timescale >= 400

fieldset = FieldSet.from_nemo(filenames, variables, dimensions,
                              allow_time_extrapolation=False,
                              indices=indices,
                              chunksize=False)


zdepth_file = '/nethome/6525954/depth_zgrid_ORCA12_V3.3.nc' 
zdepth = xr.load_dataset(zdepth_file)
fieldset.add_field(Field('depth_zgrid', zdepth['depth_zgrid'].values,
                    lon=zdepth['nav_lon'].values,
                    lat=zdepth['nav_lat'].values,
                    mesh='spherical', interp_method="nearest"))


coastal_file = '/nethome/6525954/coastal_distance_ORCA12_V3.3.nc' 
coastal = xr.load_dataset(coastal_file)
fieldset.add_field(Field('Distance', coastal['dis_var'].values,
                    lon=coastal['lon'].values,
                    lat=coastal['lat'].values,
                    mesh='spherical'))

# stokes_einstein eq. T= 4degC, and R =1e-8 m
K_h = 1.56e-6 # m^2/s. molecular diffusion. 

fieldset.add_field(Field('Kh_zonal', np.zeros_like(zdepth['depth_zgrid'].values) + K_h,
                         lon=zdepth['nav_lon'].values,
                         lat=zdepth['nav_lat'].values,
                         mesh='spherical'))

fieldset.add_field(Field('Kh_meridional', np.zeros_like(zdepth['depth_zgrid'].values) + K_h,
                         lon=zdepth['nav_lon'].values,
                         lat=zdepth['nav_lat'].values,
                         mesh='spherical'))

fieldset.add_constant('fragmentation_timescale', frag_timescale)

###############################################################################
# %%Particle Set #
###############################################################################

np.random.seed(42)
lon_cluster = [lon_sample]*n_points + np.random.normal(loc=0, scale=0.01, size=n_points)
lat_cluster = [lat_sample]*n_points + np.random.normal(loc=0, scale=0.01, size=n_points)
lon_cluster = np.array(lon_cluster) 
lat_cluster = np.array(lat_cluster)

depth_cluster = np.ones(n_points)*initial_depth  # meters
date_cluster = [start_time]*n_points
initial_radius = np.zeros_like(lon_cluster) + np.random.uniform(5e-9, 5e-7, n_points)
initial_densities = np.zeros_like(lon_cluster) + initial_particle_density

pset = ParticleSet.from_list(fieldset=fieldset, pclass=kernels_simple.PlasticParticle,
                             lon=lon_cluster,
                             lat=lat_cluster,
                             depth=depth_cluster,
                             time=date_cluster,
                             radius=initial_radius,
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
kernels += sinking_kernel
kernels += pset.Kernel(kernels_simple.VerticalRandomWalk)
kernels += pset.Kernel(kernels_simple.BrownianMotion2D)

if Frag_on == 'True':
      print('Fragmentation is on')
      kernels += pset.Kernel(kernels_simple.Fragmentation)

kernels += pset.Kernel(kernels_simple.periodicBC)
kernels += pset.Kernel(kernels_simple.reflectiveBC_bottom)
kernels += pset.Kernel(kernels_simple.reflectiveBC)

print('Kernels loaded')

# Output file
output_file = pset.ParticleFile(name=output_path,
                                outputdt=timedelta(hours=1),
                               chunks=(n_points, chunking_express))

pset.execute(kernels,
             output_file=output_file,
             runtime=timedelta(days=sim_time),
             dt=-timedelta(hours=1),
             recovery={ErrorCode.ErrorOutOfBounds: kernels_simple.delete_particle})

output_file.close()
