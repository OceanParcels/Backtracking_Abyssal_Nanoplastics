from parcels import FieldSet, ParticleSet, JITParticle
from parcels import ErrorCode, AdvectionRK4_3D
from glob import glob
import numpy as np
from datetime import timedelta
# from os import path
from datetime import datetime
# from parcels import ParcelsRandom
# import math


# data_path = 'data/NEMO/'
# output_path = 'data/test2.nc'
data_path = '/data/oceanparcels/input_data/NEMO-MEDUSA/ORCA0083-N006/'
output_path = '/scratch/cpierard/backtrack_loc0.nc'

ufiles = sorted(glob(data_path+'ORCA*U.nc'))
vfiles = sorted(glob(data_path+'ORCA*V.nc'))
wfiles = sorted(glob(data_path+'ORCA*W.nc'))
mesh_mask = data_path + 'coordinates.nc'

n_points = 10000
# start_time = datetime.strptime('2007-08-22 12:00:00', '%Y-%m-%d %H:%M:%S')

start_time = datetime.strptime('2010-12-20 12:00:00',
# '%Y-%m-%d %H:%M:%S')

filenames={'U': {'lon': mesh_mask,
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

variables={'U': 'uo',
             'V': 'vo',
             'W': 'wo'}
dimensions={'U': {'lon': 'glamf',
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

indices={'lat': range(500, 1400), 'lon': range(2500, 3800)}
fieldset=FieldSet.from_nemo(filenames, variables, dimensions,
                              allow_time_extrapolation=True,
                              indices=indices)


lon_cluster=[-6.287]*n_points
lat_cluster=[-32.171]*n_points
depth_cluster=[70]*n_points  # closest level to -5000m
date_cluster=[start_time]*n_points

# for i in range(n_points):
#     random_date = start_time + timedelta(days=np.random.randint(0, 365),
#                                          hours=np.random.randint(0, 23))
#     date_cluster[i] = random_date

lon_cluster=np.array(lon_cluster)+(np.random.random(len(lon_cluster))-0.5)/12
lat_cluster=np.array(lat_cluster)+(np.random.random(len(lat_cluster))-0.5)/12

pset=ParticleSet.from_list(fieldset=fieldset, pclass=JITParticle,
                             lon=lon_cluster,
                             lat=lat_cluster,
                             depth=depth_cluster,
                             time=date_cluster)


def delete_particle(particle, fieldset, time):
    particle.delete()


kernels=pset.Kernel(AdvectionRK4_3D)

# Output file
output_file=pset.ParticleFile(name=output_path,
                                outputdt=timedelta(hours=24))

pset.execute(kernels,
             output_file=output_file,
             runtime=timedelta(days=365),
             dt=-timedelta(hours=1),
             recovery={ErrorCode.ErrorOutOfBounds: delete_particle})

output_file.close()
