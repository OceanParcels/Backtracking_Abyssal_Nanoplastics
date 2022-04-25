import random
import string

CHARACTERS = (
    string.ascii_uppercase
    + string.digits
)

def generate_unique_key():
    return ''.join(random.sample(CHARACTERS, 6))


def log_params():
    log = {'ID': [ID],
       'test_run': [Test_run],
       'date': [submission_date],
       'depth': [initial_depth],
       'lon': [lon_sample],
       'lat': [lat_sample],
       'start_time': [start_time],
       'radius': [particle_radius],
       'density': [particle_density],
       'diffusion': [diffusion],
       'sinking_vel': [sinking_v],
       'fragmentation': [fragmentation],
       'frag_timescale': [frag_timescale],
       'frag_mode': [frag_mode],
        'bio_fields': [bio_ON]}
    return log
