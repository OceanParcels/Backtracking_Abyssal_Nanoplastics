from parcels import ParcelsRandom, Variable, JITParticle
import math
import numpy as np

# NOTES: 
# particle.depth > 0. If negative, it must be above the surface.
# particle.mld > 0. the mixed layer depth at the lat and lon of the particle.
# The Kernels are listed in the order in which they are run.


class PlasticParticle(JITParticle):
    cons_temperature = Variable('cons_temperature', dtype=np.float32,
                                initial=0)
    abs_salinity = Variable('abs_salinity', dtype=np.float32,
                            initial=0)
    mld = Variable('mld', dtype=np.float32, initial=0)
    in_mld = Variable('in_mld', dtype=np.float32, initial=0)
    Kz = Variable('Kz', dtype=np.float32, initial=0)
    seafloor = Variable('seafloor', dtype=np.float32, initial=0)
    density = Variable('density', dtype=np.float32, initial=0)
    v_s = Variable('v_s', dtype=np.float32, initial=0)
    u = Variable('u', dtype=np.float32, initial=0)
    v = Variable('v', dtype=np.float32, initial=0)
    w = Variable('w', dtype=np.float32, initial=0)
    w_k = Variable('w_k', dtype=np.float32, initial=0)
    diameter = Variable('diameter', dtype=np.float64, initial=0)
    particle_density = Variable('particle_density', dtype=np.float32,
                            initial=0)
    beta = Variable('beta',  dtype=np.float32, initial=0)
    tau_p = Variable('tau_p',  dtype=np.float32, initial=0)


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
    particle.Kz = fieldset.Kz[time, particle.depth,
                              particle.lat, particle.lon]
    particle.seafloor = fieldset.bathymetry[time, particle.depth,
                                   particle.lat, particle.lon]
    particle.u, particle.v, particle.w = fieldset.UVW[time, particle.depth,
                              particle.lat, particle.lon]


def AdvectionRK4_3D(particle, fieldset, time):
    """Advection of particles using fourth-order Runge-Kutta integration including vertical velocity.

    Function needs to be converted to Kernel object before execution"""
    # if particle.depth > particle.mld:
    (u1, v1, w1) = fieldset.UVW[particle]
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
    particle.lon += (u1 + 2*u2 + 2*u3 + u4) / 6. * particle.dt
    particle.lat += (v1 + 2*v2 + 2*v3 + v4) / 6. * particle.dt
    particle.depth += (w1 + 2*w2 + 2*w3 + w4) / 6. * particle.dt


def AdvectionRK4_1D(particle, fieldset, time):
    """Advection of particles using fourth-order Runge-Kutta integration including vertical velocity.

    Function needs to be converted to Kernel object before execution"""
    # if particle.depth > particle.mld:
    (u1, v1, w1) = fieldset.UVW[particle]
    lon = particle.lon
    lat = particle.lat
    dep1 = particle.depth + w1*.5*particle.dt
    (u2, v2, w2) = fieldset.UVW[time + .5 * particle.dt, dep1, lat, lon, particle]
    dep2 = particle.depth + w2*.5*particle.dt
    (u3, v3, w3) = fieldset.UVW[time + .5 * particle.dt, dep2, lat, lon, particle]
    dep3 = particle.depth + w3*particle.dt
    (u4, v4, w4) = fieldset.UVW[time + particle.dt, dep3, lat, lon, particle]
    particle.depth += (w1 + 2*w2 + 2*w3 + w4) / 6. * particle.dt


def VerticalRandomWalk(particle, fieldset, time):
    """Kz is in m2/s no need for convertion"""
#     if particle.depth < particle.mld:
    if particle.depth > 10:
        dWz = ParcelsRandom.normalvariate(0, math.sqrt(math.fabs(particle.dt)))
        b = math.sqrt(2 * particle.Kz)

        seafloor = particle.seafloor

        if (particle.depth - 10) < seafloor and (particle.depth) > particle.mld:
            particle.depth += b * dWz
            
        particle.w_k = dWz/particle.dt
        
    else:
        particle.w_k = 0
        
        
def BrownianMotion2D(particle, fieldset, time):
    """Kernel for simple Brownian particle diffusion in zonal and meridional
    direction. Assumes that fieldset has fields Kh_zonal and Kh_meridional
    we don't want particles to jump on land and thereby beach"""
    k = 10
    
    dWx = ParcelsRandom.normalvariate(0, math.sqrt(math.fabs(particle.dt)))
    dWy = ParcelsRandom.normalvariate(0, math.sqrt(math.fabs(particle.dt)))

    bx = math.sqrt(2 * k)
    by = math.sqrt(2 * k)

    particle.lon += bx * dWx
    particle.lat += by * dWy


def Fragmentation16(particle, fieldset, time):
    N_total = 170
    #if particle.depth > particle.mld and particle.diameter < 1e-3:
    if particle.diameter < 1e-3:
        
        # the dt is negative in the backward simulation, but normaly the 
        # exponet should be negative. 
        fragmentation_prob = math.exp(particle.dt/(fieldset.fragmentation_timescale*86400.))

        if ParcelsRandom.random(0., 1.) > fragmentation_prob:
            nummer = ParcelsRandom.random(0., 1.)
            plim3 = 128/N_total
            plim2 = plim3 + 32/N_total 
            plim1 = plim2 + 8/N_total 
            plim0 = plim1 + 2/N_total
            
            if nummer <= plim3:
                frag_mode = 16
            
            elif (plim3 < nummer) and (nummer <= plim2):
                frag_mode = 8

            elif (plim2 < nummer) and (nummer <= plim1):
                frag_mode = 4

            else:
                frag_mode = 2

            particle.diameter = particle.diameter*frag_mode # division for reverse
            
    else:
        particle.diameter = particle.diameter
            

def Fragmentation(particle, fieldset, time):
    N_total = 42 
    
    if particle.diameter < 1e-3:
        
        # the dt is negative in the backward simulation, but normaly the 
        # exponet should be negative. 
        fragmentation_prob = math.exp(particle.dt/(fieldset.fragmentation_timescale*86400.))

        if ParcelsRandom.random(0., 1.) > fragmentation_prob:
            nummer = ParcelsRandom.random(0., 1.)

            plim2 = 32/N_total
            plim1 = plim2 + 8/N_total 
            plim0 = plim1 + 2/N_total
            
            if nummer <= plim2:
                frag_mode = 8
            
            elif (plim2 < nummer) and (nummer <= plim1):
                frag_mode = 4

            else:
                frag_mode = 2

            particle.diameter = particle.diameter*frag_mode # division for reverse
    else:
        particle.diameter = particle.diameter
        

def SinkingVelocity(particle, fieldset, time):
    rho_p = particle.particle_density
    rho_f = particle.density
    beta = 3*particle.density/(2*particle.particle_density + particle.density)
    particle.beta = beta
    viscosity = 1.5e-6 
    tau_p = particle.diameter*particle.diameter/(3*beta*viscosity)
    particle.tau_p = tau_p
    v_s = (1 - beta)*9.81*tau_p
    particle.v_s = v_s

    if particle.depth > 10:
        particle.depth = particle.depth + v_s*particle.dt
    else:
        particle.v_s = 0
    

def periodicBC(particle, fieldset, time):
    if particle.lon <= -180.:
        particle.lon += 360.
    elif particle.lon >= 180.:
        particle.lon -= 360.


def reflectiveBC(particle, fieldset, time):
    if particle.depth < 0:
        particle.depth = 10
    else:
        particle.depth = particle.depth


def In_MixedLayer(particle, fieldset, time):
    if particle.depth > particle.mld:
        particle.in_mld = 0.
    else:
        particle.in_mld = 1.