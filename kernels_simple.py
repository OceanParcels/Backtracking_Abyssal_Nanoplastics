from parcels import ParcelsRandom, Variable, JITParticle
import math
import numpy as np

# NOTES: 
# particle.depth > 0. If negative, it must be above the surface.
# particle.mld > 0. the mixed layer depth at the lat and lon of the particle.


class PlasticParticle(JITParticle):
    cons_temperature = Variable('cons_temperature', dtype=np.float32,
                                initial=0)
    abs_salinity = Variable('abs_salinity', dtype=np.float32,
                            initial=0)
    mld = Variable('mld', dtype=np.float32, initial=0)
    seafloor = Variable('seafloor', dtype=np.float32, initial=0)
    density = Variable('density', dtype=np.float32, initial=0)
    v_s = Variable('v_s', dtype=np.float32, initial=0)
    u = Variable('u', dtype=np.float32, initial=0)
    v = Variable('v', dtype=np.float32, initial=0)
    w = Variable('w', dtype=np.float32, initial=0)
    w_k = Variable('w_k', dtype=np.float32, initial=0)
    radius = Variable('radius', dtype=np.float64, initial=0)
    particle_density = Variable('particle_density', dtype=np.float32,
                            initial=0)
    beta = Variable('beta',  dtype=np.float32, initial=0)
    distance = Variable('distance', dtype= np.float32, initial=0)
    surface = Variable('surface', dtype=np.float32, initial=0)
    bottom = Variable('bottom', dtype=np.float32, initial=0)

def delete_particle(particle, fieldset, time):
    particle.delete()


def SampleField(particle, fieldset, time):
    particle.cons_temperature = fieldset.cons_temperature[time, particle.depth,
                                                          particle.lat,
                                                          particle.lon]
    particle.abs_salinity = fieldset.abs_salinity[time, particle.depth,
                                                  particle.lat, particle.lon]
    particle.mld = fieldset.mld[time, particle.depth,
                                particle.lat, particle.lon]
    particle.seafloor = fieldset.bathymetry[time, particle.depth,
                                   particle.lat, particle.lon]
    particle.distance = fieldset.Distance[time, particle.depth, 
                                          particle.lat, particle.lon]
    

def Initialize_particle_depth(particle, fieldset, time):
    """
    Due to the the interpolation of the bathymetry field to determine
    the particle position from the seafloor. The seafloor depth varies
    from particle to particle. Therefore, the particle depth is set to
    the seafloor depth at the beginning of the simulation plus 10 m.
    """
    floor = fieldset.bathymetry[time, particle.depth,
                                   particle.lat, particle.lon]
    particle.depth = floor - fieldset.distance_from_seafloor


def AdvectionRK4_3D(particle, fieldset, time):
    """Advection of particles using fourth-order Runge-Kutta integration including vertical velocity.

    Function needs to be converted to Kernel object before execution"""
    
    (u1, v1, w1) = fieldset.UVW[time, particle.depth, particle.lat, particle.lon]
    particle.u, particle.v, particle.w = u1, v1, w1
    lon1 = particle.lon + u1*.5*particle.dt
    lat1 = particle.lat + v1*.5*particle.dt
    dep1 = particle.depth + w1*.5*particle.dt
    (u2, v2, w2) = fieldset.UVW[time + .5 * particle.dt, dep1, lat1, lon1, particle]
    lon2 = particle.lon + u2*.5*particle.dt
    lat2 = particle.lat + v2*.5*particle.dt
    dep2 = particle.depth + w2*.5*particle.dt
    (u3, v3, w3) = fieldset.UVW[time + .5 * particle.dt, dep2, lat2, lon2, particle]
    lon3 = particle.lon + u3*particle.dt
    lat3 = particle.lat + v3*particle.dt
    dep3 = particle.depth + w3*particle.dt
    (u4, v4, w4) = fieldset.UVW[time + particle.dt, dep3, lat3, lon3, particle]
    
    if particle.depth > particle.seafloor:
        particle.lon += 0
        particle.lat += 0
        particle.depth += 0
        
    elif particle.depth < 10:
        particle.lon += 0
        particle.lat += 0
        particle.depth += 0
        
    else:
        particle.lon += (u1 + 2*u2 + 2*u3 + u4) / 6. * particle.dt
        particle.lat += (v1 + 2*v2 + 2*v3 + v4) / 6. * particle.dt
        particle.depth += (w1 + 2*w2 + 2*w3 + w4) / 6. * particle.dt


def VerticalRandomWalk(particle, fieldset, time):
    """Kz from fieldset is in m2/s no need for convertion.
        Implementing 1D random walk in vertical direction based on 
        Ross and Sharples (2004) https://doi.org/10.4319/lom.2004.2.289
        
        Needs to run first sinking velocity kernel to calculate sinking velocity v_s
    """
    
    d_z = 1 #metters. delta z
    k_z = fieldset.Kz[time, particle.depth,
                              particle.lat, particle.lon]
    
    kz_dz = fieldset.Kz[time, particle.depth + d_z,
                              particle.lat, particle.lon]
    
    Kz_deterministic = (k_z + kz_dz)/d_z * particle.dt # gradient of Kz in z direction
    
    Kz_random = ParcelsRandom.uniform(-1., 1.) * math.sqrt(math.fabs(particle.dt) * 6 * k_z)
    
    Kz_movement = particle.v_s*particle.dt # the movements is the sinking velocity
    
    vertical_diffusion = Kz_deterministic + Kz_random
    
    particle.w_k = vertical_diffusion/particle.dt
    
    if particle.depth > particle.seafloor:
        # if particle gets below seafloor diffusion not added
        particle.depth += 0
        
    elif particle.depth < 10:
        # if particle gets above 10m diffusion not added
        particle.depth += 0
        
    else:
        particle.depth = particle.depth + vertical_diffusion
        
        
def BrownianMotion2D(particle, fieldset, time):
    """Kernel for simple Brownian particle diffusion in zonal and meridional
    direction. Assumes that fieldset has fields Kh_zonal and Kh_meridional.
    This two fields are constant and have the same values of Kh, which is 
    defined in the main script. The Kh=1.5e-6m/s2, same as Sinking velocity kernel."""
    
    dWx = ParcelsRandom.normalvariate(0, math.sqrt(math.fabs(particle.dt)))
    dWy = ParcelsRandom.normalvariate(0, math.sqrt(math.fabs(particle.dt)))

    bx = math.sqrt(2 * fieldset.Kh_zonal[time, particle.depth, 
                                          particle.lat, particle.lon])
    by = math.sqrt(2 * fieldset.Kh_meridional[time, particle.depth, 
                                          particle.lat, particle.lon])
    
    if particle.depth < 10:
        particle.lon += 0
        particle.lat += 0
        
    elif particle.depth > particle.seafloor:
        particle.lon += 0
        particle.lat += 0
        
    else:
        particle.lon += bx * dWx
        particle.lat += by * dWy


# def Fragmentation16(particle, fieldset, time):
#     N_total = 170
#     #if particle.depth > particle.mld and particle.radius < 1e-3:
#     if particle.radius < 1e-3:
        
#         # the dt is negative in the backward simulation, but normaly the 
#         # exponet should be negative. 
#         fragmentation_prob = math.exp(particle.dt/(fieldset.fragmentation_timescale*86400.))

#         if ParcelsRandom.random(0., 1.) > fragmentation_prob:
#             nummer = ParcelsRandom.random(0., 1.)
#             plim3 = 128/N_total
#             plim2 = plim3 + 32/N_total 
#             plim1 = plim2 + 8/N_total 
#             plim0 = plim1 + 2/N_total
            
#             if nummer <= plim3:
#                 frag_mode = 16
            
#             elif (plim3 < nummer) and (nummer <= plim2):
#                 frag_mode = 8

#             elif (plim2 < nummer) and (nummer <= plim1):
#                 frag_mode = 4

#             else:
#                 frag_mode = 2

#             particle.radius = particle.radius*frag_mode # division for reverse
            
#     else:
#         particle.radius = particle.radius
            

def Fragmentation(particle, fieldset, time):
    N_total = 42.5
    
    if particle.radius < 1e-3:
        
        # the dt is negative in the backward simulation, but normaly the 
        # exponet should be negative. 
        fragmentation_prob = math.exp(particle.dt/(fieldset.fragmentation_timescale*86400.))

        if ParcelsRandom.random(0., 1.) > fragmentation_prob:
            nummer = ParcelsRandom.random(0., 1.)

            plim3 = 32/N_total
            plim2 = plim3 + 8/N_total 
            plim1 = plim2 + 2/N_total
            plim0 = plim1 + 0.5/N_total
            
            if nummer <= plim3:
                particle.radius = 8*particle.radius
            
            elif (plim3 < nummer) and (nummer <= plim2):
                particle.radius = 4*particle.radius

            elif (plim2 < nummer) and (nummer <= plim1):
                particle.radius = 2*particle.radius

            else:
                particle.radius = 1.259921*particle.radius
            
    else:
        particle.radius += 0
        

def SinkingVelocity(particle, fieldset, time):
    """
    Sinking velocity kernel based on Stokes law. 
    This definition is equivalent to monroy 2017 but more convininet
    because of how we define beta. 
    """
    
    rho_p = particle.particle_density # Extract particle density
    rho_f = particle.density # Extract fluid density
    
    beta = rho_p/rho_f # denitiy ratio
    particle.beta = beta
    
    viscosity = 1.5e-6 # m2/s
    
    radius = particle.radius/2 # radius of the particle
 
    v_s = (beta - 1)*9.81*2*radius**2/(9*viscosity)
    particle.v_s = v_s

    if particle.depth < 10:
        particle.depth += 0
        
    elif particle.depth > particle.seafloor:
        particle.depth += 0
        
    else:
        particle.depth = particle.depth + v_s*particle.dt
    

def periodicBC(particle, fieldset, time):
    if particle.lon <= -180.:
        particle.lon += 360.
        
    elif particle.lon >= 180.:
        particle.lon -= 360.


def reflectiveBC(particle, fieldset, time):
    if particle.depth < 0:
        particle.depth = 10
        
    else:
        particle.depth += 0

        
def At_Surface(particle, fieldset, time):
    if particle.depth < 10:
        particle.surface = 1.
        
    else:
        particle.surface = 0.
        
        
def At_Seafloor(particle, fieldset, time):
    if particle.depth > particle.seafloor:
        particle.bottom = 1.
        
    else:
        particle.bottom = 0.