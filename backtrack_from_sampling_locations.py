"""
python3 backtrack_from_sampling_locations.py frag_ timescale
"""

from glob import glob
import numpy as np
from parcels import FieldSet, ParticleSet, JITParticle
from parcels import ErrorCode, AdvectionRK4_3D, Variable, Field
from parcels.application_kernels.TEOSseawaterdensity import PolyTEOS10_bsq
from datetime import timedelta
from datetime import datetime
import kernels_simple as local_kernels
import sys

###############################################################################
# Setting up all parameters for simulation
###############################################################################

# Control Panel for Kernels
Test_run = False
same_initial_cond = False
frag_timescale = int(sys.argv[1])

# Initial conditions
initial_depth = 5100 #int(sys.argv[1])  # 5 # 60 # 5179
lon_sample = 6.287 #6.25
lat_sample = -32.171 #-32.171
start_time = datetime.strptime('2020-01-30 12:00:00', '%Y-%m-%d %H:%M:%S')

# Particle Size and Density
particle_diameter = 5e-08  # meters
initial_particle_density = 1380  # PET kg/m3

###############################################################################
# #
###############################################################################
if Test_run:
    # Number of particles and simulation time
    n_points = 100
    sim_time = 10  # days backwards
    file_range = range(19, 21)
    output_path = '/storage/shared/oceanparcels/output_data/' + \
        f'data_Claudio/tests/a16.zarr'

else:
    # Number of particles and simulation time
    n_points = 10000
    sim_time = 4855 #10*365  # days backwards
    file_range = range(6, 21)
    output_path = '/storage/shared/oceanparcels/output_data/' + \
        f'data_Claudio/set_19/set19_{frag_timescale}.zarr'


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

for i in tqdm(file_range):
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
# Fieldset #
###############################################################################

# indices = {'lat': range(500, 1700), 'lon': range(2600, 4321)}
indices = {'lat': range(0, 1700), 'lon': range(2000, 4321)}

fieldset = FieldSet.from_nemo(filenames, variables, dimensions,
                              allow_time_extrapolation=False,
                              indices=indices,
                              chunksize=False)



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
    
    in_mld = Variable('in_mld', dtype=np.float32, initial=1)
    
    Kz = Variable('Kz', dtype=np.float32, initial=0)
    seafloor = Variable('seafloor', dtype=np.float32, initial=0)
    density = Variable('density', dtype=np.float32, initial=0)
    v_s = Variable('v_s', dtype=np.float32, initial=0)
    u = Variable('u', dtype=np.float32, initial=0)
    v = Variable('v', dtype=np.float32, initial=0)
    w = Variable('w', dtype=np.float32, initial=0)
    w_k = Variable('w_k', dtype=np.float32, initial=0)
    diameter = Variable('diameter', dtype=np.float64, initial=0)
    particle_density = Variable('particle_density', dtype=np.float32,
                            initial=initial_particle_density)


np.random.seed(0)
lon_cluster = [lon_sample]*n_points + np.random.normal(loc=0, scale=0.01, size=n_points)
lat_cluster = [lat_sample]*n_points + np.random.normal(loc=0, scale=0.01, size=n_points)
lon_cluster = np.array(lon_cluster) 
lat_cluster = np.array(lat_cluster)
depth_cluster = np.ones(n_points)*initial_depth  # meters
date_cluster = [start_time]*n_points
initial_diameters = np.zeros_like(lon_cluster) + particle_diameter + (1 - np.random.random(len(lon_cluster)))*1e-9

pset = ParticleSet.from_list(fieldset=fieldset, pclass=PlasticParticle,
                             lon=lon_cluster,
                             lat=lat_cluster,
                             depth=depth_cluster,
                             time=date_cluster,
                            diameter=initial_diameters)


###############################################################################
# Kernels #
###############################################################################
# Sampling first timestep
sample_kernel = pset.Kernel(local_kernels.SampleField)
pset.execute(sample_kernel, dt=0)
pset.execute(pset.Kernel(PolyTEOS10_bsq), dt=0)

# Loading kernels
kernels = sample_kernel + pset.Kernel(PolyTEOS10_bsq) 
kernels += pset.Kernel(local_kernels.AdvectionRK4_3D)
kernels += pset.Kernel(local_kernels.VerticalRandomWalk)

fieldset.add_constant('fragmentation_timescale', frag_timescale)  # days
kernels += pset.Kernel(local_kernels.Fragmentation16)

kernels += pset.Kernel(local_kernels.SinkingVelocity)
kernels += pset.Kernel(local_kernels.reflectiveBC)
kernels += pset.Kernel(local_kernels.periodicBC)
kernels += pset.Kernel(local_kernels.In_MixedLayer)


print('Kernels loaded')

# Output file
output_file = pset.ParticleFile(name=output_path,
                                outputdt=timedelta(hours=24),
                               chunks=(n_points, 10))

pset.execute(kernels,
             output_file=output_file,
             runtime=timedelta(days=sim_time),
             dt=-timedelta(hours=1),
             recovery={ErrorCode.ErrorOutOfBounds: local_kernels.delete_particle})

output_file.close()
