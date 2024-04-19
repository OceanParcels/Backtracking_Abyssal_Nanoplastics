"""
To run do:
      python3 backtrack_from_sampling_locations.py -ft 10000 -bm 1

-ft : Fragmentation timescale in days (int)
-bm : Brownian motion on or off (boolean)
"""
# %%
from glob import glob
import numpy as np
from parcels import FieldSet, ParticleSet
from parcels import ErrorCode, Field
from parcels.application_kernels.TEOSseawaterdensity import PolyTEOS10_bsq
from datetime import timedelta
from datetime import datetime
import kernels_simple as kernels_simple
import sys
from tqdm import tqdm
import xarray as xr

from argparse import ArgumentParser

###############################################################################
# %%Setting up all parameters for simulation
###############################################################################

# Control Panel for Kernels
Test_run = False

arguments = ArgumentParser()
arguments.add_argument('-ft', '--frag_timescale', type=int, default=23000, help='Fragmentation timescale (days)')
arguments.add_argument('-bm', '--brownian_motion', type=int, help='Brownian motion on (1) or off (0)')

args = arguments.parse_args()

frag_timescale = args.frag_timescale
Brownian_on = args.brownian_motion

# You need to save the last two steps in a file and load it here, because the function from_particlefile does not 
# support backtracking
existing_file_name = '/storage/shared/oceanparcels/output_data/data_Claudio/abyssal_nps_outputs/hc13_25000_BM_False_lasttwo.zarr'

# insert this manually. It's only used for loading the filedsets. Leave a month as buffer
start_time = datetime.strptime('2012-11-21 12:00:00', '%Y-%m-%d %H:%M:%S')

# Particle Size and Density
initial_particle_density = 1380  # PET & PVC kg/m3

###############################################################################
# %%
###############################################################################
data_path = '/storage2/shared/oceanparcels/input_data/MOi/psy4v3r1/'
wfiles = sorted(glob(data_path + f'psy4v3r1-daily_W_*.nc'))

if Test_run:
    # Number of particles and simulation time
#     n_points = 100
    sim_time = 20  # days backwards
    output_path = '/storage/shared/oceanparcels/output_data/' + \
                    f'data_Claudio/tests/restart23000file.zarr'
    
    wfiles = sorted(glob(data_path+'psy4v3r1-daily_W_2011-*.nc'))
    chunking_express = 1
    end_time = datetime.strptime('2011-02-01 12:00:00', '%Y-%m-%d %H:%M:%S')
    
else:
    # Number of particles and simulation time
    n_points = 8192 #2^13
    sim_time = 4484 # 4484
    # From 11 October 2006 to and including 20 January 2019 (forward).
    # Result: 4485 days or 12 years, 3 months, 10 days including the end date.
    end_time = datetime.strptime('2007-01-01 12:00:00', '%Y-%m-%d %H:%M:%S')
    
    file_range = range(6, 21)
    output_path = '/storage/shared/oceanparcels/output_data/' + \
        f'data_Claudio/abyssal_nps_outputs/hc13_{frag_timescale}_BM_{Brownian_on}_part2.zarr'
    chunking_express = 500


print(output_path, Brownian_on)

# Loading the only the files that we need.
# indexes are inverted because the start date is in the future.
# it's a backwards in time simulation
start_index = 0 
end_index = 0

for file in wfiles:
    if file[-13:-3] == start_time.strftime('%Y-%m-%d'):
        end_index = wfiles.index(file)
        
    if file[-13:-3] == end_time.strftime('%Y-%m-%d'):
        start_index = wfiles.index(file)
    
wfiles = wfiles[start_index:end_index+1]

###############################################################################
# %%Reading files #
###############################################################################
vfiles = [f.replace('_W_', '_V_') for f in wfiles]
ufiles = [f.replace('_W_', '_U_') for f in wfiles]
tfiles = [f.replace('_W_', '_T_') for f in wfiles]
sfiles = [f.replace('_W_', '_S_') for f in wfiles]
KZfiles = [f.replace('_W_', '_KZ_') for f in wfiles]
twoDfiles = [f.replace('_W_', '_2D_') for f in wfiles]

mesh_mask = '/storage/shared/oceanparcels/input_data/MOi/' + \
            'domain_ORCA0083-N006/Old/coordinates.nc'

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
# %%Fieldset #
###############################################################################

# indices = {'lat': range(0, 1700), 'lon': range(200, 4321)}
# indices = {'lat': range(200, 1700), 'lon': range(2300, 4321)} # domain for frag timescale < 400
indices = {'lat': range(200, 1700)} # whole domain for frag timescale >= 400

fieldset = FieldSet.from_nemo(filenames, variables, dimensions,
                              allow_time_extrapolation=False,
                              indices=indices,
                              chunksize=False)

fieldset.Kz.interp_method = 'linear'

zdepth_file = '/nethome/6525954/depth_zgrid_ORCA12_V3.3.nc' 
zdepth = xr.load_dataset(zdepth_file)
fieldset.add_field(Field('depth_zgrid', zdepth['depth_zgrid'].values,
                    lon=zdepth['nav_lon'].values,
                    lat=zdepth['nav_lat'].values,
                    mesh='spherical', interp_method="nearest"))


coastal_file = '/nethome/6525954/coastal_distance_ORCA12_V3.3.nc' 
coastal = xr.load_dataset(coastal_file)
fieldset.add_field(Field('Distance', coastal['dis_var'].values,
                    lon=coastal['lon'].values,
                    lat=coastal['lat'].values,
                    mesh='spherical'))

fieldset.add_constant('fragmentation_timescale', frag_timescale)

if Brownian_on == 1:
      # stokes_einstein eq. T= 4degC, and R =1e-8 m
      K_h = 1.56e-6 # m^2/s. molecular diffusion. 

      fieldset.add_field(Field('Kh_zonal', np.zeros_like(zdepth['depth_zgrid'].values) + K_h,
                              lon=zdepth['nav_lon'].values,
                              lat=zdepth['nav_lat'].values,
                              mesh='spherical'))

      fieldset.add_field(Field('Kh_meridional', np.zeros_like(zdepth['depth_zgrid'].values) + K_h,
                              lon=zdepth['nav_lon'].values,
                              lat=zdepth['nav_lat'].values,
                              mesh='spherical'))

###############################################################################
# %%Particle Set #
###############################################################################
# Initialise the ParticleSet from a zarr ParticleFile. This creates a new 
# ParticleSet based on locations of all particles written in a zarr ParticleFile 
# at a certain time. Particle IDs are preserved if restart=True

# start_time = datetime.strptime('2011-11-01 12:00:00', '%Y-%m-%d %H:%M:%S')
pset = ParticleSet.from_particlefile(fieldset=fieldset, pclass=kernels_simple.PlasticParticle,
                             filename=existing_file_name)
                        #      restarttime=start_time)

###############################################################################
# %%Kernels #
###############################################################################
sample_kernel = pset.Kernel(kernels_simple.SampleField)
sinking_kernel = pset.Kernel(kernels_simple.SinkingVelocity)

kernels = sample_kernel + pset.Kernel(PolyTEOS10_bsq)
kernels += pset.Kernel(kernels_simple.AdvectionRK4_3D)
kernels += sinking_kernel
kernels += pset.Kernel(kernels_simple.VerticalRandomWalk)

if Brownian_on == 1:
      print('Brownian motion ON')
      kernels += pset.Kernel(kernels_simple.BrownianMotion2D)

kernels += pset.Kernel(kernels_simple.Fragmentation)
kernels += pset.Kernel(kernels_simple.periodicBC)
kernels += pset.Kernel(kernels_simple.reflectiveBC_bottom)
kernels += pset.Kernel(kernels_simple.reflectiveBC)

print('Kernels loaded')

# Output file
output_file = pset.ParticleFile(name=output_path,
                                outputdt=timedelta(days=1))

pset.execute(kernels,
             output_file=output_file,
             runtime=timedelta(days=sim_time),
             dt=-timedelta(hours=1),
             recovery={ErrorCode.ErrorOutOfBounds: kernels_simple.delete_particle})

output_file.close()

# %%
