from parcels import FieldSet, ParticleSet, JITParticle, AdvectionRK4_3D
from parcels import ErrorCode, Field
from glob import glob
import numpy as np
from datetime import timedelta
from os import path
from datetime import datetime
from parcels import ParcelsRandom
import math


data_path = 'data/NEMO/'
output_path = 'data/test1.nc'
# data_path = '/data/oceanparcels/input_data/NEMO-MEDUSA/ORCA0083-N006/'
# output_path = '/scratch/cpierard/test_1.nc'

ufiles = sorted(glob(data_path+'ORCA*U.nc'))
vfiles = sorted(glob(data_path+'ORCA*V.nc'))
wfiles = sorted(glob(data_path+'ORCA*W.nc'))
mesh_mask = data_path + 'coordinates.nc'

n_points = 10
start_time = datetime.strptime('2007-08-22 12:00:00', '%Y-%m-%d %H:%M:%S')
K_bar = 10
# start_time = datetime.strptime('2010-12-20 12:00:00',
# '%Y-%m-%d %H:%M:%S')


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

variables = {'U': 'uo',
             'V': 'vo',
             'W': 'wo'}
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

indices = {'lat': range(500, 1400), 'lon': range(2500, 3800)}
fieldset = FieldSet.from_nemo(filenames, variables, dimensions,
                              allow_time_extrapolation=True,
                              indices=indices)

###############################################################################
# Adding the horizontal diffusion                                             #
###############################################################################
size2D = (fieldset.U.grid.ydim, fieldset.U.grid.xdim, fieldset.U.grid.zdim)
K_h = K_bar * np.ones(size2D)

fieldset.add_field(Field('Kh_zonal', data=K_h,
                         lon=fieldset.U.grid.lon, lat=fieldset.U.grid.lat,
                         depth=fieldset.U.grid.depth, mesh='spherical'))
fieldset.add_field(Field('Kh_meridional', data=K_h,
                         lon=fieldset.U.grid.lon, lat=fieldset.U.grid.lat,
                         depth=fieldset.U.grid.depth, mesh='spherical'))
fieldset.add_field(Field('Kh_vertical', data=K_h,
                         lon=fieldset.U.grid.lon, lat=fieldset.U.grid.lat,
                         depth=fieldset.U.grid.depth, mesh='spherical'))


lon_cluster = [-6.287]*n_points
lat_cluster = [-32.171]*n_points
depth_cluster = [70]*n_points  # closest level to -5000m
date_cluster = [start_time]*n_points
# for i in range(n_points):
#     random_date = start_time + timedelta(days=np.random.randint(0, 365),
#                                          hours=np.random.randint(0, 23))
#     date_cluster[i] = random_date

# lon_cluster = np.array(lon_cluster)+(np.random.random(len(lon_cluster))-0.5)/24

pset = ParticleSet.from_list(fieldset=fieldset, pclass=JITParticle,
                             lon=lon_cluster,
                             lat=lat_cluster,
                             depth=depth_cluster,
                             time=date_cluster)


def delete_particle(particle, fieldset, time):
    particle.delete()


def BrownianMotion3D(particle, fieldset, time):
    """Kernel for simple Brownian particle diffusion in zonal, meridional and
    vertical direction. Assumes that fieldset has fields Kh_zonal and
    Kh_meridional we don't want particles to jump on land and thereby beach"""
    if particle.beach == 0:
        dWx = ParcelsRandom.normalvariate(0, math.sqrt(math.fabs(particle.dt)))
        dWy = ParcelsRandom.normalvariate(0, math.sqrt(math.fabs(particle.dt)))
        dWz = ParcelsRandom.normalvariate(0, math.sqrt(math.fabs(particle.dt)))

        bx = math.sqrt(2 * fieldset.Kh_zonal[particle])
        by = math.sqrt(2 * fieldset.Kh_meridional[particle])
        bz = math.sqrt(2 * fieldset.Kh_vertical[particle])

        particle.lon += bx * dWx
        particle.lat += by * dWy
        particle.lat += bz * dWz


kernels = pset.Kernel(AdvectionRK4_3D) + pset.Kernel(BrownianMotion3D)

# Output file
output_file = pset.ParticleFile(name=output_path,
                                outputdt=timedelta(hours=1))

pset.execute(kernels,
             output_file=output_file,
             runtime=timedelta(days=8),
             dt=-timedelta(hours=1),
             recovery={ErrorCode.ErrorOutOfBounds: delete_particle})

output_file.close()
