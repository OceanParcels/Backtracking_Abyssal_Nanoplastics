from glob import glob
import numpy as np
from parcels import FieldSet, ParticleSet, ScipyParticle, JITParticle
from parcels import ErrorCode, Variable, Field
from parcels.application_kernels.TEOSseawaterdensity import PolyTEOS10_bsq
from datetime import timedelta
from datetime import datetime
import kernels
import xarray as xr
import sys

# bio_ON = False

RKx = sys.argv[1]  # 'RK4' or 'RK1'
Advection_ = True

print(RKx)

n_points = 1000
sim_time = 25  # days backwards
# particle_size = 1e-6  # meters
particle_size = np.linspace(1e-6, 1e-4, n_points)
particle_density = 1380  # kg/m3
initial_depth = 1  # 5 # 60 # 5179
start_time = datetime.strptime('2018-01-01 12:00:00', '%Y-%m-%d %H:%M:%S')
series = 2

# Lorenz - MOi fields
data_path = '/storage/shared/oceanparcels/input_data/MOi/psy4v3r1/'
output_path = '/storage/shared/oceanparcels/output_data/' + \
    f'data_Claudio/{RKx}_{initial_depth}m_t{sim_time}_rho{particle_density}_advec.nc'

print(f'SA_{initial_depth}m_s{series:02d}.nc')
ufiles = []
vfiles = []
wfiles = []
tfiles = []
sfiles = []
twoDfiles = []
k_zfiles = []

for i in range(8, 9):
    ufiles = ufiles + sorted(glob(data_path + f'psy4v3r1-daily_U_201{i}*.nc'))
    vfiles = vfiles + sorted(glob(data_path + f'psy4v3r1-daily_V_201{i}*.nc'))
    wfiles = wfiles + sorted(glob(data_path + f'psy4v3r1-daily_W_201{i}*.nc'))
    tfiles = tfiles + sorted(glob(data_path + f'psy4v3r1-daily_T_201{i}*.nc'))
    sfiles = sfiles + sorted(glob(data_path + f'psy4v3r1-daily_S_201{i}*.nc'))
    twoDfiles = twoDfiles + sorted(glob(data_path +
                                        f'psy4v3r1-daily_2D_201{i}*.nc'))
    k_zfiles = k_zfiles + sorted(glob(data_path +
                                      f'psy4v3r1-daily_KZ_201{i}*.nc'))


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

filenames['K_z'] = {'lon': mesh_mask,
                    'lat': mesh_mask,
                    'depth': k_zfiles[0],
                    'data': k_zfiles}

variables = {'U': 'vozocrtx',
             'V': 'vomecrty',
             'W': 'vovecrtz'}

variables['cons_temperature'] = 'votemper'
variables['abs_salinity'] = 'vosaline'
variables['mld'] = 'somxlavt'
variables['K_z'] = 'votkeavt'

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

dimensions['K_z'] = {'lon': 'glamf',
                            'lat': 'gphif',
                            'depth': 'depthw',
                            'time': 'time_counter'}


indices = {'lat': range(750, 1300), 'lon': range(2900, 4000)}

fieldset = FieldSet.from_nemo(filenames, variables, dimensions,
                              allow_time_extrapolation=False,
                              indices=indices)

print('Fieldset loaded')

# fieldset.add_constant('grow_rate', 1e-6)
# fieldset.add_constant('g', -9.81)
fieldset.add_constant('viscosity', 1e-6)
fieldset.add_constant('particle_density', particle_density)
bathy = xr.load_dataset(bathy_file)

fieldset.add_field(Field('bathymetry', bathy['Bathymetry'].values,
                         lon=bathy['nav_lon'].values,
                         lat=bathy['nav_lat'].values,
                         mesh='spherical'))


if RKx == 'RK4':
    class PlasticParticle(ScipyParticle):
        cons_temperature = Variable('cons_temperature', dtype=np.float32,
                                    initial=0)
        abs_salinity = Variable('abs_salinity', dtype=np.float32,
                                initial=0)
        mld = Variable('mld', dtype=np.float32, initial=0)
        alpha = Variable('alpha', dtype=np.float32, initial=particle_size)
        density = Variable('density', dtype=np.float32, initial=1035)
        v_s = Variable('v_s', dtype=np.float32, initial=0)
        alpha = Variable('alpha', dtype=np.float32, initial=particle_size)
        w = Variable('w', dtype=np.float32, initial=0)
        k_z = Variable('k_z', dtype=np.float32, initial=0)


elif RKx == 'RK1':
    class PlasticParticle(JITParticle):
        cons_temperature = Variable('cons_temperature', dtype=np.float32,
                                    initial=0)
        abs_salinity = Variable('abs_salinity', dtype=np.float32,
                                initial=0)
        mld = Variable('mld', dtype=np.float32, initial=0)
        alpha = Variable('alpha', dtype=np.float32, initial=particle_size)
        density = Variable('density', dtype=np.float32, initial=1035)
        v_s = Variable('v_s', dtype=np.float32, initial=0)
        alpha = Variable('alpha', dtype=np.float32, initial=particle_size)
        w = Variable('w', dtype=np.float32, initial=0)
        k_z = Variable('k_z', dtype=np.float32, initial=0)


lon_cluster = [6.287]*n_points
lat_cluster = [-32.171]*n_points

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


if RKx == 'RK4':
    v_s_kernel = pset.Kernel(kernels.SinkingVelocity_RK4)
elif RKx == 'RK1':
    v_s_kernel = pset.Kernel(kernels.SinkingVelocity)

    
if Advection_:
    kernels = pset.Kernel(kernels.AdvectionRK4_1D) + \
            pset.Kernel(kernels.SampleField) + \
            pset.Kernel(PolyTEOS10_bsq) + v_s_kernel
else:
    kernels = pset.Kernel(kernels.SampleField) + \
        pset.Kernel(PolyTEOS10_bsq) + v_s_kernel

print('Kernels loaded')

# Output file
output_file = pset.ParticleFile(name=output_path,
                                outputdt=timedelta(hours=12))

pset.execute(kernels,
             output_file=output_file,
             runtime=timedelta(days=sim_time),
             dt=timedelta(hours=1),
             recovery={ErrorCode.ErrorOutOfBounds: delete_particle})

output_file.close()
