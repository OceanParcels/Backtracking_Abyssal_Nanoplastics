from parcels import FieldSet, ParticleSet, JITParticle, AdvectionRK4_3D
from glob import glob
import numpy as np
from datetime import timedelta
from os import path

data_path = 'data/NEMO/'
output_path = 'data/test.nc'
ufiles = sorted(glob(data_path+'ORCA*U.nc'))
vfiles = sorted(glob(data_path+'ORCA*V.nc'))
wfiles = sorted(glob(data_path+'ORCA*W.nc'))
mesh_mask = data_path + 'coordinates.nc'

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

fieldset = FieldSet.from_nemo(filenames, variables, dimensions)

pset = ParticleSet.from_line(fieldset=fieldset, pclass=JITParticle,
                             size=10,
                             start=(1.9, 52.5),
                             finish=(3.4, 51.6),
                             depth=1)

kernels = pset.Kernel(AdvectionRK4_3D)

# Output file
output_file = pset.ParticleFile(name=output_path,
                                outputdt=timedelta(hours=6))

pset.execute(kernels,
             output_file=output_file,
             runtime=timedelta(days=1),
             dt=timedelta(hours=6))

output_file.close()
