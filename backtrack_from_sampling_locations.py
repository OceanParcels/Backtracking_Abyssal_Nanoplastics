from glob import glob
import numpy as np
from parcels import FieldSet, ParticleSet, JITParticle
from parcels import ErrorCode, AdvectionRK4_3D, Variable
from parcels.application_kernels.TEOSseawaterdensity import PolyTEOS10_bsq
from datetime import timedelta
from datetime import datetime


n_points = 10000
bio_ON = False

sim_time = 360 #days backwards

# Lorenz - MOi fields
data_path = '/storage/shared/oceanparcels/input_data/MOi/2019/'

output_path = '/storage/shared/oceanparcels/output_data/' + \
    'data_Claudio/backtrack_samplinglocation5170-600.nc'

if bio_ON:
    bio_data_path = '/storage/shared/oceanparcels/input_data/MOi/biomer4v2r1/'
    phfiles = sorted(glob(bio_data_path + 'biomer4v2r1-weekly_ph_201*.nc'))
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

ufiles = sorted(glob(data_path + 'psy4v3r1-daily_U_2019*.nc'))
vfiles = sorted(glob(data_path + 'psy4v3r1-daily_V_2019*.nc'))
wfiles = sorted(glob(data_path + 'psy4v3r1-daily_W_2019*.nc'))
tfiles = sorted(glob(data_path + 'psy4v3r1-daily_T_2019*.nc'))    
sfiles = sorted(glob(data_path + 'psy4v3r1-daily_S_2019*.nc'))
twoDfiles = sorted(glob(data_path + 'psy4v3r1-daily_2D_2019*.nc'))

mesh_mask = '/storage/shared/oceanparcels/input_data/MOi/domain_ORCA0083-N006/coordinates.nc'

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

start_time = datetime.strptime('2019-12-30 12:00:00', '%Y-%m-%d %H:%M:%S')
# psy4v3r1-daily_2D_2019-01-01.nc

variables = {'U': 'vozocrtx',
             'V': 'vomecrty',
             'W': 'vovecrtz'}

variables['temperature'] = 'votemper'
variables['salinity'] = 'vosaline'
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

dimensions['temperature'] = {'lon': 'glamf', 
                                  'lat': 'gphif',
                                  'depth': 'depthw', 
                                  'time': 'time_counter'}

dimensions['salinity'] = {'lon': 'glamf', 
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
    
fieldset.add_constant('grow_rate', 1e-6)
fieldset.add_constant('g', 9.81)
fieldset.add_constant('beta', 0.88)
fieldset.add_constant('viscosity', 1e-6)

class PlasticParticle(JITParticle):
    temperature = Variable('temperature', dtype=np.float32, initial=0)
    salinity = Variable('salinity', dtype=np.float32, initial=0)
    mld = Variable('mld', dtype=np.float32, initial=0)
    alpha = Variable('alpha', dtype=np.float32, initial=1e-5)
#     ph = Variable('ph', dtype=np.float32, initial=0)
    
lon_cluster = [6.287]*n_points
lat_cluster = [-32.171]*n_points
lon_cluster = np.array(lon_cluster)+(np.random.random(len(lon_cluster))-0.5)/24
lat_cluster = np.array(lat_cluster)+(np.random.random(len(lat_cluster))-0.5)/24

depth_cluster = np.ones(n_points)*5170 #meters
date_cluster = [start_time]*n_points

pset = ParticleSet.from_list(fieldset=fieldset, pclass=PlasticParticle,
                             lon=lon_cluster,
                             lat=lat_cluster,
                             depth=depth_cluster,
                             time=date_cluster)

def delete_particle(particle, fieldset, time):
    particle.delete()
    
def GrowParticle(particle, fieldset, time):
    particle.alpha = particle.alpha - fieldset.grow_rate*particle.dt
    
def SampleField(particle, fielset, time):
    particle.temperature = fieldset.temperature[time, particle.depth, 
                                               particle.lat, particle.lon]
    particle.salinity = fieldset.salinity[time, particle.depth, 
                                               particle.lat, particle.lon]
#     particle.ph = fieldset.ph[time, particle.depth, 
#                                                particle.lat, particle.lon]
    particle.mld = fieldset.mld[time, particle.depth, 
                                               particle.lat, particle.lon]
    
kernels = pset.Kernel(AdvectionRK4_3D) + pset.Kernel(SampleField)

# Output file
output_file = pset.ParticleFile(name=output_path,
                                outputdt=timedelta(hours=24))

pset.execute(kernels,
             output_file=output_file,
             runtime=timedelta(days=sim_time),
             dt=-timedelta(hours=1),
             recovery={ErrorCode.ErrorOutOfBounds: delete_particle})

output_file.close()
