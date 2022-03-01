from glob import glob
import numpy as np
from parcels import FieldSet, ParticleSet, JITParticle
from parcels import ErrorCode, AdvectionRK4_3D, Variable, Field
from parcels.application_kernels.TEOSseawaterdensity import PolyTEOS10_bsq
from datetime import timedelta
from datetime import datetime
import xarray as xr
import kernels
import sys

# Kernels
bio_ON = False
diffusion = False # this kernel has not been added yet
sinking_v = False

# Particle Size and Density
particle_size = 1e-6  # meters
particle_density = 1380  # kg/m3

# Number of particles and simulation time
n_points = 50000
sim_time = 10*365  # days backwards

# Initial condition
initial_depth = int(sys.argv[1])  # 5 # 60 # 5179
start_time = datetime.strptime('2019-12-30 12:00:00', '%Y-%m-%d %H:%M:%S')

# Lorenz - MOi
data_path = '/storage/shared/oceanparcels/input_data/MOi/psy4v3r1/'
output_path = '/storage/shared/oceanparcels/output_data/' + \
    f'data_Claudio/backtrack_SA/SA_{initial_depth}m_t{sim_time}_diff-{diffusion}.nc'
#       'periodic_boundaries_test.nc'

print(f'SA_{initial_depth}m_t{sim_time}_diff-{diffusion}')
ufiles = []
vfiles = []
wfiles = []
tfiles = []
sfiles = []
twoDfiles = []

for i in range(7, 20):
    ufiles = ufiles + sorted(glob(data_path + f'psy4v3r1-daily_U_20{i:02d}*.nc'))
    vfiles = vfiles + sorted(glob(data_path + f'psy4v3r1-daily_V_20{i:02d}*.nc'))
    wfiles = wfiles + sorted(glob(data_path + f'psy4v3r1-daily_W_20{i:02d}*.nc'))
    tfiles = tfiles + sorted(glob(data_path + f'psy4v3r1-daily_T_20{i:02d}*.nc'))
    sfiles = sfiles + sorted(glob(data_path + f'psy4v3r1-daily_S_20{i:02d}*.nc'))
    twoDfiles = twoDfiles + sorted(glob(data_path +
                                        f'psy4v3r1-daily_2D_20{i:02d}*.nc'))

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


variables = {'U': 'vozocrtx',
             'V': 'vomecrty',
             'W': 'vovecrtz'}

variables['cons_temperature'] = 'votemper'
variables['abs_salinity'] = 'vosaline'
variables['mld'] = 'somxlavt'

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

if initial_depth == 5:
    min_ind, max_ind = 0, 33 # Also these
elif initial_depth == 60:
    min_ind, max_ind = 0, 33 # Need to check these depths
elif initial_depth == 5179:
    min_ind, max_ind = 34, 49
else:
    raise ValueError('Depth indices have not been setup.') 
# indices = {'lat': range(750, 1300), 'lon': range(2900, 4000)}  # before domain expansion
indices = {'lat': range(500, 1800),
           'lon': range(0, 4322),
           'deptht': range(min_ind, max_ind)}  # after domain expansion 
#  {'deptht': range(min_ind, max_ind)}
fieldset = FieldSet.from_nemo(filenames, variables, dimensions,
                              allow_time_extrapolation=False,
                              indices=indices,
                             chunksize=False)
#                               indices=indices) # I comment this for long runs

print('Fieldset loaded')
if bio_ON:
    bio_fieldset = FieldSet.from_nemo(filenames_bio, variables_bio,
                                      dimensions_bio)
    fieldset.add_field(bio_fieldset.ph)

# fieldset.add_constant('grow_rate', 1e-6)
# fieldset.add_constant('g', -9.81)
fieldset.add_constant('viscosity', 1e-6)
fieldset.add_constant('particle_density', particle_density)
bathy = xr.load_dataset(bathy_file)

fieldset.add_field(Field('bathymetry', bathy['Bathymetry'].values,
                         lon=bathy['nav_lon'].values,
                         lat=bathy['nav_lat'].values,
                         mesh='spherical'))


class PlasticParticle(JITParticle):
    cons_temperature = Variable('cons_temperature', dtype=np.float32,
                                initial=0)
    abs_salinity = Variable('abs_salinity', dtype=np.float32,
                            initial=0)
    mld = Variable('mld', dtype=np.float32, initial=0)
    alpha = Variable('alpha', dtype=np.float32, initial=particle_size)
    density = Variable('density', dtype=np.float32, initial=1035)
    v_s = Variable('v_s', dtype=np.float32, initial=0)

#     beta = Variable('beta', dtype=np.float32, initial=0)
#     tau_p = v_s = Variable('tau_p', dtype=np.float32, initial=0)
#     if bio_ON:
#         ph = Variable('ph', dtype=np.float32, initial=0)


lon_cluster = [6.287]*n_points
lat_cluster = [-32.171]*n_points
lon_cluster = np.array(lon_cluster)+(np.random.random(len(lon_cluster))-0.5)/24
lat_cluster = np.array(lat_cluster)+(np.random.random(len(lat_cluster))-0.5)/24

depth_cluster = np.ones(n_points)*initial_depth  # meters
date_cluster = [start_time]*n_points

print('--------------')

pset = ParticleSet.from_list(fieldset=fieldset, pclass=PlasticParticle,
                             lon=lon_cluster,
                             lat=lat_cluster,
                             depth=depth_cluster,
                             time=date_cluster)


print('Particle Set Created')

def delete_particle(particle, fieldset, time):
    particle.delete()


def SampleField(particle, fielset, time):
    particle.cons_temperature = fieldset.cons_temperature[time, particle.depth,
                                                          particle.lat,
                                                          particle.lon]
    particle.abs_salinity = fieldset.abs_salinity[time, particle.depth,
                                                  particle.lat, particle.lon]
    particle.mld = fieldset.mld[time, particle.depth,
                                particle.lat, particle.lon]


def periodicBC(particle, fieldset, time):
    if particle.lon <= -180.:
        particle.lon += 360.
    elif particle.lon >= 180.:
        particle.lon -= 360.


def SinkingVelocity(particle, fieldset, time):
    rho_p = fieldset.particle_density
    rho_f = particle.density
    nu = fieldset.viscosity
    alpha = particle.alpha
    g = 9.81
    dt = particle.dt
    beta = 3*rho_f/(2*rho_p + rho_f)
    tau_p = alpha*alpha/(3*beta*nu)
    tolerance = 10

    seafloor = fieldset.bathymetry[time, particle.depth,
                                   particle.lat, particle.lon]

    if (particle.depth - 10) < seafloor and (particle.depth + 10) > 0:
        v_s = (1 - beta)*g*tau_p
    else:
        v_s = 0

    particle.v_s = v_s
    particle.depth = particle.depth + v_s*dt

    
#Sampling first timestep
sample_kernel = pset.Kernel(SampleField)
pset.execute(sample_kernel, dt=0)

# Loading kernels
kernels = pset.Kernel(AdvectionRK4_3D) + sample_kernel + pset.Kernel(PolyTEOS10_bsq)

if sinking_v:
    kernels += pset.Kernel(SinkingVelocity)
    
if diffusion:
    print('pending')
    
kernels += pset.Kernel(periodicBC)

print('Kernels loaded')

# Output file
output_file = pset.ParticleFile(name=output_path,
                                outputdt=timedelta(hours=24))

pset.execute(kernels,
             output_file=output_file,
             runtime=timedelta(days=sim_time),
             dt=-timedelta(hours=1),
             recovery={ErrorCode.ErrorOutOfBounds: delete_particle})

output_file.close()
