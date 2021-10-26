from glob import glob
import numpy as np
from parcels import FieldSet, ParticleSet, JITParticle
from parcels import ErrorCode, AdvectionRK4_3D

from datetime import timedelta
# from os import path
from datetime import datetime
# from parcels import ParcelsRandom
# import math

# local path - NEMO
# data_path = 'data/NEMO/'
# output_path = 'data/test3.nc'

# Gemini - NEMO fieldsets
data_path = '/data/oceanparcels/input_data/NEMO-MEDUSA/ORCA0083-N006/'
output_path = '/scratch/cpierard/backtrack_loc0_column.nc'

# Lorenz - MOi fields
# data_path = '/storage/shared/oceanparcels/input_data/MOi/2019/'
# output_data = '/storage/shared/oceanparcels/output_data/' + \
#     'data_Claudio/backtrack_loc0_column.nc'

# Local path - MOi
data_path = 'data/MOi/'
output_data = 'data/test-moi-1.nc'

ufiles = sorted(glob(data_path+'psy4v3r1-daily_U*.nc'))
vfiles = sorted(glob(data_path+'psy4v3r1-daily_V*.nc'))
wfiles = sorted(glob(data_path+'psy4v3r1-daily_W*.nc'))
mesh_mask = data_path + 'domain/coordinates.nc'

print(ufiles, vfiles, wfiles)

# # removes the weird files that don't have date.
# ufiles2 = []
# vfiles2 = []
# wfiles2 = []
# for i in range(len(ufiles)):
#     if len(ufiles[i].split('_')[2]) > 11:
#         ufiles2.append(ufiles[i])
#
# for i in range(len(vfiles)):
#     if len(vfiles[i].split('_')[2]) > 11:
#         vfiles2.append(vfiles[i])
#
# for i in range(len(wfiles)):
#     if len(wfiles[i].split('_')[2]) > 11:
#         wfiles2.append(wfiles[i])
# filenames = {'U': {'lon': mesh_mask,
#                    'lat': mesh_mask,
#                    'depth': wfiles2[0],
#                    'data': ufiles2},
#              'V': {'lon': mesh_mask,
#                    'lat': mesh_mask,
#                    'depth': wfiles2[0],
#                    'data': vfiles2},
#              'W': {'lon': mesh_mask,
#                    'lat': mesh_mask,
#                    'depth': wfiles2[0],
#                    'data': wfiles2}}

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

n_points = 10000
# start_time = datetime.strptime('2007-08-22 12:00:00', '%Y-%m-%d %H:%M:%S')

# start_time = datetime.strptime('2010-12-20 12:00:00', '%Y-%m-%d %H:%M:%S')
start_time = datetime.strptime('2019-12-02 12:00:00', '%Y-%m-%d %H:%M:%S')
# psy4v3r1-daily_2D_2019-01-01.nc

variables = {'U': 'vozocrtx',
             'V': 'vomecrty',
             'W': 'vovecrtz'}

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

indices = {'lat': range(750, 1300), 'lon': range(2900, 4000)}
# indices = {'lat': range(500, 1400), 'lon': range(2500, 3800)}
fieldset = FieldSet.from_nemo(filenames, variables, dimensions,
                              allow_time_extrapolation=False)
# indices=indices)

# indices=indices)
lon_cluster = [6.287]*n_points
lat_cluster = [-32.171]*n_points
# depth_cluster = [5000]*n_points  # closest level to -5000m
depth_cluster = np.linspace(1, 5000, n_points)

date_cluster = [start_time]*n_points

# for i in range(n_points):
#     random_date = start_time + timedelta(days=np.random.randint(0, 365),
#                                          hours=np.random.randint(0, 23))
#     date_cluster[i] = random_date

# lon_cluster = np.array(lon_cluster)+(np.random.random(len(lon_cluster))-0.5)/12
# lat_cluster = np.array(lat_cluster)+(np.random.random(len(lat_cluster))-0.5)/12

pset = ParticleSet.from_list(fieldset=fieldset, pclass=JITParticle,
                             lon=lon_cluster,
                             lat=lat_cluster,
                             depth=depth_cluster,
                             time=date_cluster)


def delete_particle(particle, fieldset, time):
    particle.delete()


kernels = pset.Kernel(AdvectionRK4_3D)

# Output file
output_file = pset.ParticleFile(name=output_path,
                                outputdt=timedelta(hours=24))

pset.execute(kernels,
             output_file=output_file,
             runtime=timedelta(days=1),
             dt=-timedelta(hours=1),
             recovery={ErrorCode.ErrorOutOfBounds: delete_particle})

output_file.close()
