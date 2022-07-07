"""
python3 backtrack_from_sampling_locations.py 5173 dif v_s frag frag_timescale
"""

from glob import glob
import numpy as np
from parcels import FieldSet, ParticleSet, JITParticle
from parcels import ErrorCode, AdvectionRK4_3D, Variable, Field
from parcels.application_kernels.TEOSseawaterdensity import PolyTEOS10_bsq
from datetime import timedelta
from datetime import datetime
import xarray as xr
import local_kernels
import sys
import toolbox  # homemade module with useful functions
import os
import pandas as pd

###############################################################################
# Setting up all parameters for simulation     #
###############################################################################

# Control Panel for Kernels
bio_ON = False
Test_run = True

# Reading arguments

if str(sys.argv[2]) == "dif":
    diffusion = True
    print('diffusion')
else:
    diffusion = False

if str(sys.argv[3]) == "v_s":
    sinking_v = True
    print('v_s')
else:
    sinking_v = False

if str(sys.argv[4]) == "frag":
    fragmentation = True
    print('fragmentation')
else:
    fragmentation = False


frag_timescale = int(sys.argv[5])
frag_mode = 1/2
# Initial condition
initial_depth = int(sys.argv[1])  # 5 # 60 # 5179
lon_sample = 6.287
lat_sample = -32.171
start_time = datetime.strptime('2019-12-30 12:00:00', '%Y-%m-%d %H:%M:%S')

ID = toolbox.generate_unique_key()
submission_date = datetime.now()

###############################################################################
# #
###############################################################################

if Test_run:
    # Number of particles and simulation time
    n_points = 1000
    sim_time = 10  # days backwards
    file_range = range(19, 20)
    output_path = '/storage/shared/oceanparcels/output_data/' + \
        f'data_Claudio/tests/{ID}.zarr'

else:
    # Number of particles and simulation time
    n_points = 10000
    sim_time = 10*365  # days backwards
    file_range = range(7, 20)
    output_path = '/storage/shared/oceanparcels/output_data/' + \
        f'data_Claudio/set_10/{ID}.zarr'

# Particle Size and Density
particle_diameter = 5e-08  # meters
particle_density = 1380  # PET kg/m3
# initial_volume = 4/3*np.pi*particle_radius**3

###############################################################################
# Simulations Log #
###############################################################################
log_file = 'log_simulationsV2.csv'
# log_run = toolbox.log_params()
log_run = {'ID': [ID],
           'test_run': [Test_run],
           'date': [submission_date],
           'depth': [initial_depth],
           'lon': [lon_sample],
           'lat': [lat_sample],
           'start_time': [start_time],
           'sim_time': [sim_time],
           'diameter': [particle_diameter],
           'density': [particle_density],
           'diffusion': [diffusion],
           'sinking_vel': [sinking_v],
           'fragmentation': [fragmentation],
           'frag_timescale': [frag_timescale],
           'frag_mode': [frag_mode],
           'bio_fields': [bio_ON]}

log_run = pd.DataFrame(log_run)

if os.path.exists(log_file):
    log = pd.read_csv(log_file, index_col=0)
else:
    log = pd.DataFrame()
    
log = pd.concat([log, log_run], axis=0)
log.to_csv(log_file)

###############################################################################
# Reading files #
###############################################################################
data_path = '/storage/shared/oceanparcels/input_data/MOi/psy4v3r1/'

ufiles = []
vfiles = []
wfiles = []
tfiles = []
sfiles = []
twoDfiles = []
KZfiles = []

for i in file_range:
    ufiles = ufiles + sorted(glob(data_path + f'psy4v3r1-daily_U_20{i:02d}*.nc'))
    vfiles = vfiles + sorted(glob(data_path + f'psy4v3r1-daily_V_20{i:02d}*.nc'))
    wfiles = wfiles + sorted(glob(data_path + f'psy4v3r1-daily_W_20{i:02d}*.nc'))
    tfiles = tfiles + sorted(glob(data_path + f'psy4v3r1-daily_T_20{i:02d}*.nc'))
    sfiles = sfiles + sorted(glob(data_path + f'psy4v3r1-daily_S_20{i:02d}*.nc'))
    twoDfiles = twoDfiles + sorted(glob(data_path +
                                        f'psy4v3r1-daily_2D_20{i:02d}*.nc'))
    KZfiles = KZfiles + sorted(glob(data_path +
                                    f'psy4v3r1-daily_KZ_20{i:02d}*.nc'))

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
                   'data': wfiles}}

filenames['cons_temperature'] = {'lon': mesh_mask,
                                 'lat': mesh_mask,
                                 'depth': wfiles[0],
                                 'data': tfiles}
filenames['abs_salinity'] = {'lon': mesh_mask,
                             'lat': mesh_mask,
                             'depth': wfiles[0],
                             'data': sfiles}

filenames['mld'] = {'lon': mesh_mask,
                    'lat': mesh_mask,
                    'depth': twoDfiles[0],
                    'data': twoDfiles}

filenames['Kz'] = {'lon': mesh_mask,
                   'lat': mesh_mask,
                   'depth': wfiles[0],
                   'data': KZfiles}

variables = {'U': 'vozocrtx',
             'V': 'vomecrty',
             'W': 'vovecrtz'}

variables['cons_temperature'] = 'votemper'
variables['abs_salinity'] = 'vosaline'
variables['mld'] = 'somxlavt'
variables['Kz'] = 'votkeavt'

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
                    'time': 'time_counter'}}

dimensions['cons_temperature'] = {'lon': 'glamf',
                                  'lat': 'gphif',
                                  'depth': 'depthw',
                                  'time': 'time_counter'}

dimensions['abs_salinity'] = {'lon': 'glamf',
                              'lat': 'gphif',
                              'depth': 'depthw',
                              'time': 'time_counter'}

dimensions['mld'] = {'lon': 'glamf',
                            'lat': 'gphif',
                            'depth': 'deptht',
                            'time': 'time_counter'}

dimensions['Kz'] = {'lon': 'glamf',
                    'lat': 'gphif',
                    'depth': 'depthw',
                    'time': 'time_counter'}


if bio_ON:
    bio_data_path = '/storage/shared/oceanparcels/input_data/MOi/biomer4v2r1/'
    phfiles = sorted(glob(bio_data_path + 'biomer4v2r1-weekly_ph_2019*.nc'))
    mesh_mask_bio = '/storage/shared/oceanparcels/input_data/MOi/' + \
                    'domain_ORCA025-N006/coordinates.nc'
    filenames_bio = {'ph': {'lon': mesh_mask_bio,
                            'lat': mesh_mask_bio,
                            'depth': wfiles[0],
                            'data': phfiles}}

    variables_bio = {'ph': 'ph'}

    dimensions_bio = {'ph': {'lon': 'glamf',
                             'lat': 'gphif',
                             'depth': 'depthw',
                             'time': 'time_counter'}}

###############################################################################
# Fieldset #
###############################################################################
# if initial_depth == 5:
#     min_ind, max_ind = 0, 33 # Also these
# elif initial_depth == 60:
#     min_ind, max_ind = 0, 33 # Need to check these depths
# elif initial_depth == 5179:
#     min_ind, max_ind = 34, 49
# else:
#     raise ValueError('Depth indices have not been setup.')

# indices = {'lat': range(500, 1800),
#            'lon': range(0, 4322),
#            'deptht': range(min_ind, max_ind)}  # after domain expansion
#  {'deptht': range(min_ind, max_ind)}

indices = {'lat': range(750, 1300), 'lon': range(2900, 4000)}  # before domain expansion

fieldset = FieldSet.from_nemo(filenames, variables, dimensions,
                              allow_time_extrapolation=False,
                              indices=indices,
                              chunksize=False)

print('Fieldset loaded')

# Load biofiles
if bio_ON:
    bio_fieldset = FieldSet.from_nemo(filenames_bio, variables_bio,
                                      dimensions_bio)
    fieldset.add_field(bio_fieldset.ph)

fieldset.add_constant('viscosity', 1e-6)
fieldset.add_constant('particle_density', particle_density)
bathy = xr.load_dataset(bathy_file)

fieldset.add_field(Field('bathymetry', bathy['Bathymetry'].values,
                         lon=bathy['nav_lon'].values,
                         lat=bathy['nav_lat'].values,
                         mesh='spherical'))

###############################################################################
# Particle Set #
###############################################################################


class PlasticParticle(JITParticle):
    cons_temperature = Variable('cons_temperature', dtype=np.float32,
                                initial=0)
    abs_salinity = Variable('abs_salinity', dtype=np.float32,
                            initial=0)
    mld = Variable('mld', dtype=np.float32, initial=0)
    surface = Variable('surface', dtype=np.int32, initial=0)
    Kz = Variable('Kz', dtype=np.float32, initial=0)
    diameter = Variable('diameter', dtype=np.float32, initial=particle_diameter)
    seafloor = Variable('seafloor', dtype=np.float32, initial=0)
    density = Variable('density', dtype=np.float32, initial=0)
    v_s = Variable('v_s', dtype=np.float32, initial=0)
    w = Variable('w', dtype=np.float32, initial=0)

np.random.seed(0)
lon_cluster = [lon_sample]*n_points
lat_cluster = [lat_sample]*n_points
lon_cluster = np.array(lon_cluster) # +(np.random.random(len(lon_cluster))-0.5)/24
lat_cluster = np.array(lat_cluster) # +(np.random.random(len(lat_cluster))-0.5)/24

depth_cluster = np.ones(n_points)*initial_depth  # meters
date_cluster = [start_time]*n_points

print('--------------')

pset = ParticleSet.from_list(fieldset=fieldset, pclass=PlasticParticle,
                             lon=lon_cluster,
                             lat=lat_cluster,
                             depth=depth_cluster,
                             time=date_cluster)


print('Particle Set Created')

###############################################################################
# Kernels #
###############################################################################
# Sampling first timestep
sample_kernel = pset.Kernel(local_kernels.SampleField)
pset.execute(sample_kernel, dt=0)
pset.execute(pset.Kernel(PolyTEOS10_bsq))

# Loading kernels
kernels = pset.Kernel(AdvectionRK4_3D) + sample_kernel + pset.Kernel(PolyTEOS10_bsq)
kernels += pset.Kernel(local_kernels.periodicBC)
kernels += pset.Kernel(local_kernels.reflectiveBC)
kernels += pset.Kernel(local_kernels.ML_freeze)

if sinking_v:
    print('v_s')
    kernels += pset.Kernel(local_kernels.SinkingVelocity)

if diffusion:
    print('Vertical diffusion')
    kernels += pset.Kernel(local_kernels.VerticalRandomWalk)

if fragmentation:
    print('fragmentation')
    fieldset.add_constant('fragmentation_mode', frag_mode)
    fieldset.add_constant('fragmentation_timescale', frag_timescale)  # days
    kernels += pset.Kernel(local_kernels.fragmentation)

print('Kernels loaded')

# Output file
output_file = pset.ParticleFile(name=output_path,
                                outputdt=timedelta(hours=24))

pset.execute(kernels,
             output_file=output_file,
             runtime=timedelta(days=sim_time),
             dt=-timedelta(hours=1),
             recovery={ErrorCode.ErrorOutOfBounds: local_kernels.delete_particle})

output_file.close()
