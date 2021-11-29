from glob import glob
import numpy as np
from parcels import FieldSet, ParticleSet, JITParticle
from parcels import ErrorCode, AdvectionRK4_3D, Variable
from parcels.application_kernels.TEOSseawaterdensity import PolyTEOS10_bsq
from datetime import timedelta
from datetime import datetime

bio_ON = False
n_points = 10
sim_time = 10 # days backwards
particle_size = 1e-6 # meters
particle_density = 1380 # kg/m3
initial_depth = 5179 # 5 # 60 # 5179
start_time = datetime.strptime('2019-12-30 12:00:00', '%Y-%m-%d %H:%M:%S')

if bio_ON:
    bio_data_path = '/storage/shared/oceanparcels/input_data/MOi/biomer4v2r1/'
    phfiles = sorted(glob(bio_data_path + 'biomer4v2r1-weekly_ph_2019*.nc'))
    mesh_mask_bio = '/storage/shared/oceanparcels/input_data/MOi/domain_ORCA025-N006/coordinates.nc'
    filenames_bio = {'ph': {'lon': mesh_mask_bio, 
                              'lat': mesh_mask_bio, 
                              'depth': wfiles[0], 
                              'data': phfiles}}
    
    variables_bio = {'ph': 'ph'}

    dimensions_bio = {'ph': {'lon': 'glamf', 
                               'lat': 'gphif', 
                               'depth': 'depthw', 
                               'time': 'time_counter'}}
    
# Lorenz - MOi fields
data_path = '/storage/shared/oceanparcels/input_data/MOi/2019/'
output_path = '/storage/shared/oceanparcels/output_data/' + \
    'data_Claudio/SA_loc01_5179m.nc'

ufiles = []
vfiles = []
wfiles = []
tfiles = []
sfiles = [] 
twoDfiles = []

for i in range(9, 10):
    ufiles = ufiles + sorted(glob(data_path + f'psy4v3r1-daily_U_201{i}*.nc'))
    vfiles = vfiles + sorted(glob(data_path + f'psy4v3r1-daily_V_201{i}*.nc'))
    wfiles = wfiles + sorted(glob(data_path + f'psy4v3r1-daily_W_201{i}*.nc'))
    tfiles = tfiles + sorted(glob(data_path + f'psy4v3r1-daily_T_201{i}*.nc'))    
    sfiles = sfiles + sorted(glob(data_path + f'psy4v3r1-daily_S_201{i}*.nc'))
    twoDfiles = twoDfiles + sorted(glob(data_path + f'psy4v3r1-daily_2D_201{i}*.nc'))

mesh_mask = '/storage/shared/oceanparcels/input_data/MOi/domain_ORCA0083-N006/coordinates.nc'
bathy_file = '/storage/shared/oceanparcels/input_data/MOi/domain_ORCA0083-N006/bathymetry_ORCA12_V3.3.nc'

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

filenames['temperature'] = {'lon': mesh_mask, 
                                 'lat': mesh_mask, 
                                 'depth': wfiles[0], 
                                 'data': tfiles}
filenames['salinity'] = {'lon': mesh_mask, 
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

indices = {'lat': range(750, 1300), 'lon': range(2900, 4000)}

fieldset = FieldSet.from_nemo(filenames, variables, dimensions,
                              allow_time_extrapolation=False,
                              indices=indices)

if bio_ON:
    bio_fieldset = FieldSet.from_nemo(filenames_bio, variables_bio, dimensions_bio)
    fieldset.add_field(bio_fieldset.ph)
    
# fieldset.add_constant('grow_rate', 1e-6)
# fieldset.add_constant('g', -9.81)
fieldset.add_constant('viscosity', 1e-6)
fieldset.add_constant('particle_density', particle_density)


class PlasticParticle(JITParticle):
    cons_temperature = Variable('cons_temperature', dtype=np.float32, initial=fieldset.cons_temperature)
    abs_salinity = Variable('abs_salinity', dtype=np.float32, initial=fieldset.abs_salinity)
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

depth_cluster = np.ones(n_points)*initial_depth #meters
date_cluster = [start_time]*n_points

pset = ParticleSet.from_list(fieldset=fieldset, pclass=PlasticParticle,
                             lon=lon_cluster,
                             lat=lat_cluster,
                             depth=depth_cluster,
                             time=date_cluster)


def delete_particle(particle, fieldset, time):
    particle.delete()
    

def SampleField(particle, fielset, time):
    particle.cons_temperature = fieldset.cons_temperature[time, particle.depth, 
                                               particle.lat, particle.lon]
    particle.abs_salinity = fieldset.abs_salinity[time, particle.depth, 
                                               particle.lat, particle.lon]
    particle.mld = fieldset.mld[time, particle.depth, 
                                               particle.lat, particle.lon]
#     particle.ph = fieldset.ph[time, particle.depth, 
#                                                particle.lat, particle.lon]

def SinkingVelocity(particle, fieldset, time):
    rho_p = fieldset.particle_density ##
    rho_f = particle.density
    nu = fieldset.viscosity
    alpha = particle.alpha
    g = 9.81
    dt = particle.dt
    beta = 3*rho_f/(2*rho_p + rho_f)
    tau_p = alpha*alpha/(3*beta*nu) ## alpha*alpha
    
    seafloor = fieldset.bathymetry[time, particle.depth, particle.lat, particle.lon]
    
    if particle.depth < seafloor and particle.depth > 0:
        v_s = (1 - beta)*g*tau_p
    else:
        v_s = 0
        
    particle.v_s = v_s
    particle.depth = particle.depth + v_s*dt


kernels = pset.Kernel(AdvectionRK4_3D) + pset.Kernel(SampleField) + pset.Kernel(PolyTEOS10_bsq) + pset.Kernel(SinkingVelocity)

# Output file
output_file = pset.ParticleFile(name=output_path,
                                outputdt=timedelta(hours=24))

pset.execute(kernels,
             output_file=output_file,
             runtime=timedelta(days=sim_time),
             dt=-timedelta(hours=1),
             recovery={ErrorCode.ErrorOutOfBounds: delete_particle})

output_file.close()
