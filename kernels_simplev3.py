from parcels import ParcelsRandom, Variable, JITParticle
import math
import numpy as np

# NOTES:
# particle.depth > 0. If negative, it must be above the surface.
# careful with the dt. It is negative because of the backward integration.

class PlasticParticle(JITParticle):
    """
    Particle class definition with additional variables
    """
    cons_temperature = Variable('cons_temperature', dtype=np.float32,
                                initial=0)
    abs_salinity = Variable('abs_salinity', dtype=np.float32,
                            initial=0)

    density = Variable('density', dtype=np.float32, initial=0)
    kz = Variable('kz', dtype=np.float32, initial=0)
    
    # dynamic variables
    u = Variable('u', dtype=np.float32, initial=0)
    v = Variable('v', dtype=np.float32, initial=0)
    w = Variable('w', dtype=np.float32, initial=0)
    v_s = Variable('v_s', dtype=np.float32, initial=0)
    
    # vertical displacement variables 
    z_kz = Variable('z_kz', dtype=np.float32, initial=0)

    radius = Variable('radius', dtype=np.float64, initial=0)
    particle_density = Variable('particle_density', dtype=np.float32,
                            initial=0)
    
    distance = Variable('distance', dtype= np.float32, initial=0)
    
    bottom = Variable('bottom', dtype=np.float32, initial=0)
    
    # amount of times particle touches the bottom
    floored = Variable('floored', dtype=np.float32, initial=0)
    
    #number of times particles fragments
    frag_events = Variable('frag_events', dtype=np.float32, initial=0)
    

def delete_particle(particle, fieldset, time):
    particle.delete()


def SampleField(particle, fieldset, time):
    """
    Sample the fieldset at the particle location and store it in the
    particle variable.
    """
    particle.cons_temperature = fieldset.cons_temperature[time, particle.depth,
                                                          particle.lat,
                                                          particle.lon]
    particle.abs_salinity = fieldset.abs_salinity[time, particle.depth,
                                                  particle.lat, particle.lon]
    particle.distance = fieldset.Distance[time, particle.depth, 
                                          particle.lat, particle.lon]
    
    particle.bottom = fieldset.depth_zgrid[time, particle.depth, 
                                          particle.lat, particle.lon]


def AdvectionRK4_3D(particle, fieldset, time):
    """
    Advection of particles using fourth-order Runge-Kutta integration 
    including vertical velocity.
    Function needs to be converted to Kernel object before execution
    """
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
    
    if particle.depth < 10:
        particle.lon += 0
        particle.lat += 0
        particle.depth += 0
        
    else:
        particle.lon += (u1 + 2*u2 + 2*u3 + u4) / 6. * particle.dt
        particle.lat += (v1 + 2*v2 + 2*v3 + v4) / 6. * particle.dt
        particle.depth += (w1 + 2*w2 + 2*w3 + w4) / 6. * particle.dt


def SinkingVelocity(particle, fieldset, time):
    """
    Sinking velocity kernel based on Stokes law. 
    This definition is equivalent to monroy 2017 but more convininet
    because of how we define beta. 
    """
    
    rho_p = particle.particle_density # Extract particle density
    rho_f = particle.density # Extract fluid density
    
    beta = rho_p/rho_f # denitiy ratio

    viscosity = 1.5e-6 # m2/s kinematic viscosity
 
    v_s = (beta - 1)*9.81*2*particle.radius**2/(9*viscosity)
    particle.v_s = v_s


def VerticalRandomWalk(particle, fieldset, time):
    """Kz from fieldset is in m2/s no need for convertion.
        Implementing 1D random walk in vertical direction based on 
        Ross and Sharples (2004) https://doi.org/10.4319/lom.2004.2.289
        
        Needs to run first sinking velocity kernel to calculate sinking velocity v_s
    """
    
    d_z = 1 #metters. delta z
    k_z = fieldset.Kz[time, particle.depth,
                              particle.lat, particle.lon]
    particle.kz = k_z
     
    kz_dz = fieldset.Kz[time, particle.depth - d_z,
                              particle.lat, particle.lon]

    
    Kz_deterministic = (k_z - kz_dz)/d_z * math.fabs(particle.dt) # gradient of Kz in z direction
    
    Kz_random = ParcelsRandom.uniform(-1., 1.) * math.sqrt(math.fabs(particle.dt) * 6 * k_z)
    
    Kz_movement = particle.v_s*particle.dt # dt < 0!
    
    particle.z_kz = Kz_deterministic + Kz_random
    
    vertical_diffusion = Kz_deterministic + Kz_random + Kz_movement
    
    if particle.depth > particle.bottom:
        # if particle gets below seafloor diffusion not added
        particle.depth += Kz_movement
        
    elif particle.depth < 10:
        # if particle gets above 10m diffusion not added
        particle.depth += 0
        
    else:
        particle.depth += vertical_diffusion
        # particle.depth += Kz_movement
        
        
def BrownianMotion2D(particle, fieldset, time):
    """
    Kernel for simple Brownian particle diffusion in zonal and meridional
    direction. Assumes that fieldset has fields Kh_zonal and Kh_meridional.
    This two fields are constant and have the same values of Kh, which is 
    defined in the main script. The Kh=1.5e-6m/s2, same as Sinking velocity kernel.
    """
    
    dWx = ParcelsRandom.normalvariate(0, math.sqrt(math.fabs(particle.dt)))
    dWy = ParcelsRandom.normalvariate(0, math.sqrt(math.fabs(particle.dt)))
    
    bx = math.sqrt(2 * fieldset.Kh_zonal[time, particle.depth, 
                                          particle.lat, particle.lon])
    by = math.sqrt(2 * fieldset.Kh_meridional[time, particle.depth, 
                                          particle.lat, particle.lon])
    
    if particle.depth < 10:
        particle.lon += 0
        particle.lat += 0
        
    elif particle.depth > particle.bottom:
        particle.lon += 0
        particle.lat += 0
        
    else:
        particle.lon += bx * dWx
        particle.lat += by * dWy


def Fragmentation(particle, fieldset, time):
    """
    Kernel for de-fragmentation of particles with three size classes.
    If random number is larger than the probability of fragmentation
    there is a fragmentation event and the particle radius changes
    according to the fragmentation distribution
    
    
    """
    N_total = 10.5 # total number of particles in fragmentation event
    
    if particle.radius < 1e-4:
        
        # the dt is negative in the backward simulation, but normaly the 
        # exponet should be negative. 
        fragmentation_prob = math.exp(particle.dt/(fieldset.fragmentation_timescale*86400.))

      
        if ParcelsRandom.random(0., 1.) > fragmentation_prob:
            particle.frag_events += 1.
            nummer = ParcelsRandom.random(0., 1.)

            plim2 = 8/N_total 
            plim1 = plim2 + 2/N_total
            
            if nummer <= plim2:
                particle.radius = 4*particle.radius

            elif (nummer > plim2) and (nummer <= plim1):
                particle.radius = 2*particle.radius

            else:
                particle.radius = 1.259921*particle.radius
            
    else:
        particle.radius += 0
    

def periodicBC(particle, fieldset, time):
    """
    Kernel for periodic boundary conditions in zonal direction.
    """
    if particle.lon <= -180.:
        particle.lon += 360.
        
    elif particle.lon >= 180.:
        particle.lon -= 360.


def reflectiveBC(particle, fieldset, time):
    """
    Kernel for reflective boundary conditions at the surface to avoid
    particles overshooting the surface.
    """
    if particle.depth < 0:
        particle.depth = 9.
        
    else:
        particle.depth += 0


def reflectiveBC_bottom(particle, fieldset, time):
    """
    Kernel for reflective boundary conditions at the bottom to avoid
    particles getting stuck in the bottom.
    """
    if particle.depth > particle.bottom:
        particle.depth = particle.bottom - 10.
        particle.floored += 1
        
    else:
        particle.depth += 0

