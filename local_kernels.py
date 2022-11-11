from parcels import ParcelsRandom
import math
import numpy as np

# NOTES: 
# particle.depth > 0. If negative, it must be above the surface.
# particle.mld > 0. the mixed layer depth at the lat and lon of the particle.
# The Kernels are listed in the order in which they are run.

def delete_particle(particle, fieldset, time):
    particle.delete()


def SampleField(particle, fielset, time):
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
    if particle.in_motion == 1:
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


def VerticalRandomWalk(particle, fieldset, time):
    """Kz is in m2/s no need for convertion"""
#     if particle.depth < particle.mld:
    if particle.in_motion == 1 and particle.depth < particle.mld:
        dWz = ParcelsRandom.normalvariate(0, math.sqrt(math.fabs(particle.dt)))
        b = math.sqrt(2 * particle.Kz)

        seafloor = particle.seafloor

        if (particle.depth - 10) < seafloor and (particle.depth) > particle.mld:
            particle.depth += b * dWz
            
        particle.w_k = dWz/particle.dt
        
        
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


def Fragmentation(particle, fieldset, time):
    
    #if particle.depth > particle.mld and particle.diameter < 1e-3:
    if particle.in_motion == 1 and particle.diameter < 1e-3:
        
        # the dt is negative in the backward simulation, but normaly the 
        # exponet should be negative. 
        fragmentation_prob = math.exp(particle.dt/(fieldset.fragmentation_timescale*86400.))

        if ParcelsRandom.random(0., 1.) > fragmentation_prob:
            nummer = ParcelsRandom.random(0., 1.)
            plim0 = 32/42 #8./14.5
            plim1 = 40/42 #12./14.5
            plim2 = 2/42 #14./14.5
            
            if nummer <= plim0:
                frag_mode = 8

            elif (plim0 < nummer) and (nummer <= plim1):
                frag_mode = 4

            else:
                frag_mode = 2

            particle.diameter = particle.diameter*frag_mode # division for reverse
            

def SinkingVelocity(particle, fieldset, time):
    if particle.depth > particle.mld:
        rho_p = particle.particle_density
        rho_f = particle.density
        beta = 3*particle.density/(2*particle.particle_density + particle.density)
        tau_p = particle.diameter*particle.diameter/(12*beta*1e-6)
        v_s = (1 - beta)*9.81*tau_p
        particle.v_s = v_s
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
        particle.depth = particle.mld
        
#     if particle.seafloor/particle.depth < 1:
#         particle.depth = particle.seafloor - 10


def stuck_Seafloor(particle, fieldset, time):
    if particle.seafloor/particle.depth <= 1:
        particle.in_motion = 0
    else:
        particle.in_motion = 1


def In_MixedLayer(particle, fieldset, time):
    if particle.depth < particle.mld:
        particle.in_motion = 0
    else:
        particle.in_motion = 1