from parcels import ParcelsRandom
import math


def SinkingVelocity(particle, fieldset, time):
    rho_p = fieldset.particle_density
    rho_f = particle.density
    nu = fieldset.viscosity
    alpha = particle.radius
    g = 9.81
    dt = particle.dt
    beta = 3*rho_f/(2*rho_p + rho_f)
    tau_p = alpha*alpha/(3*beta*nu)
    tolerance = 10

    seafloor = fieldset.bathymetry[time, particle.depth,
                                   particle.lat, particle.lon]

    if (particle.depth - 10) < seafloor and (particle.depth) > particle.mld:
        v_s = (1 - beta)*g*tau_p
    else:
        v_s = 0

    particle.v_s = v_s
    particle.depth = particle.depth + v_s*dt


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

    seafloor = fieldset.bathymetry[time, particle.depth,
                                   particle.lat, particle.lon]
#     particle.w = fieldset.W[time, particle.depth,
#                             particle.lat, particle.lon]
#     particle.k_z = fieldset.K_z[time, particle.depth,
#                                 particle.lat, particle.lon]


def AdvectionRK4_1D(particle, fieldset, time):
    """
    Advection of particles using fourth-order Runge-Kutta integration including
    vertical velocity.Function needs to be converted to Kernel object before
    execution.
    """
    (w1) = fieldset.W[time, particle.depth,
                      particle.lat, particle.lon]
    depth_1 = particle.depth + w1 * .5 * particle.dt

    (w2) = fieldset.W[time + .5 * particle.dt, depth_1,
                      particle.lat, particle.lon]
    depth_2 = particle.depth + w2 * .5 * particle.dt

    (w3) = fieldset.W[time + .5 * particle.dt, depth_2,
                      particle.lat, particle.lon]
    depth_3 = particle.depth + w3 * particle.dt

    (w4) = fieldset.W[time + particle.dt, depth_3,
                      particle.lat, particle.lon]

    particle.depth += (w1 + 2 * w2 + 2 * w3 + w4) / 6. * particle.dt


def delete_particle(particle, fieldset, time):
    particle.delete()


def reflectiveBC(particle, fieldset, time):
    if particle.depth < 0:
        particle.depth = math.fabs(particle.depth)


def periodicBC(particle, fieldset, time):
    if particle.lon <= -180.:
        particle.lon += 360.
    elif particle.lon >= 180.:
        particle.lon -= 360.


def BrownianMotion3D(particle, fieldset, time):
    """Kernel for simple Brownian particle diffusion in zonal and meridional
    direction. Assumes that fieldset has fields Kh_zonal and Kh_meridional
    we don't want particles to jump on land and thereby beach"""

    dWx = ParcelsRandom.normalvariate(0, math.sqrt(math.fabs(particle.dt)))
    dWy = ParcelsRandom.normalvariate(0, math.sqrt(math.fabs(particle.dt)))
    dWz = ParcelsRandom.normalvariate(0, math.sqrt(math.fabs(particle.dt)))

    b = math.sqrt(2 * fieldset.diffusion)

    particle.lon += b * dWx
    particle.lat += b * dWy
    particle.depth += b * dWz


def VerticalRandomWalk(particle, fieldset, time):
    """Kz is in m2/s no need for convertion"""

    dWz = ParcelsRandom.normalvariate(0, math.sqrt(math.fabs(particle.dt)))
    b = math.sqrt(2 * particle.Kz)

    seafloor = fieldset.bathymetry[time, particle.depth,
                                   particle.lat, particle.lon]

    if (particle.depth - 50) < seafloor and (particle.depth) > particle.mld:
        particle.depth += b * dWz


def fragmentation(particle, fieldset, time):
    if particle.radius < 5e-3:
        fragmentation_prob = math.exp(-1/(fieldset.fragmentation_timescale*24))

        if ParcelsRandom.random(0., 1.) > fragmentation_prob:
            particle.volume = particle.volume/fieldset.fragmentation_mode
            particle.radius = (3*particle.volume/(4*math.pi))**(1./3.)


def SinkingVelocity_RK4(particle, fieldset, time):
    def polyTEOS10_bsq(Z, SA, CT):
        Z = - Z  # particle.depth  # note: use negative depths!
        SAu = 40 * 35.16504 / 35
        CTu = 40
        Zu = 1e4
        deltaS = 32
        R000 = 8.0189615746e+02
        R100 = 8.6672408165e+02
        R200 = -1.7864682637e+03
        R300 = 2.0375295546e+03
        R400 = -1.2849161071e+03
        R500 = 4.3227585684e+02
        R600 = -6.0579916612e+01
        R010 = 2.6010145068e+01
        R110 = -6.5281885265e+01
        R210 = 8.1770425108e+01
        R310 = -5.6888046321e+01
        R410 = 1.7681814114e+01
        R510 = -1.9193502195e+00
        R020 = -3.7074170417e+01
        R120 = 6.1548258127e+01
        R220 = -6.0362551501e+01
        R320 = 2.9130021253e+01
        R420 = -5.4723692739e+00
        R030 = 2.1661789529e+01
        R130 = -3.3449108469e+01
        R230 = 1.9717078466e+01
        R330 = -3.1742946532e+00
        R040 = -8.3627885467e+00
        R140 = 1.1311538584e+01
        R240 = -5.3563304045e+00
        R050 = 5.4048723791e-01
        R150 = 4.8169980163e-01
        R060 = -1.9083568888e-01
        R001 = 1.9681925209e+01
        R101 = -4.2549998214e+01
        R201 = 5.0774768218e+01
        R301 = -3.0938076334e+01
        R401 = 6.6051753097e+00
        R011 = -1.3336301113e+01
        R111 = -4.4870114575e+00
        R211 = 5.0042598061e+00
        R311 = -6.5399043664e-01
        R021 = 6.7080479603e+00
        R121 = 3.5063081279e+00
        R221 = -1.8795372996e+00
        R031 = -2.4649669534e+00
        R131 = -5.5077101279e-01
        R041 = 5.5927935970e-01
        R002 = 2.0660924175e+00
        R102 = -4.9527603989e+00
        R202 = 2.5019633244e+00
        R012 = 2.0564311499e+00
        R112 = -2.1311365518e-01
        R022 = -1.2419983026e+00
        R003 = -2.3342758797e-02
        R103 = -1.8507636718e-02
        R013 = 3.7969820455e-01
        ss = math.sqrt((SA + deltaS) / SAu)
        tt = CT / CTu
        zz = -Z / Zu
        rz3 = R013 * tt + R103 * ss + R003
        rz2 = (R022 * tt + R112 * ss + R012) * tt + (R202 * ss + R102) * ss + R002
        rz1 = (((R041 * tt + R131 * ss + R031) * tt + (R221 * ss + R121) * ss + R021) * tt + ((R311 * ss + R211)
                                                                                              * ss + R111) * ss + R011) * tt + (((R401 * ss + R301) * ss + R201) * ss + R101) * ss + R001
        rz0 = (((((R060 * tt + R150 * ss + R050) * tt + (R240 * ss + R140) * ss + R040) * tt + ((R330 * ss + R230) * ss + R130) * ss + R030) * tt + (((R420 * ss + R320) * ss + R220) * ss + R120)
                * ss + R020) * tt + ((((R510 * ss + R410) * ss + R310) * ss + R210) * ss + R110) * ss + R010) * tt + (((((R600 * ss + R500) * ss + R400) * ss + R300) * ss + R200) * ss + R100) * ss + R000
        density = ((rz3 * zz + rz2) * zz + rz1) * zz + rz0

        return density

    g = 9.81
    dt = particle.dt
    rho_p = fieldset.particle_density

    nu = fieldset.viscosity
    alpha = particle.alpha
    seafloor = fieldset.bathymetry[time, particle.depth,
                                   particle.lat, particle.lon]

    if particle.depth < seafloor and particle.depth > 0:
        T = fieldset.cons_temperature[time, particle.depth,
                                      particle.lat, particle.lon]
        S = fieldset.abs_salinity[time, particle.depth,
                                  particle.lat, particle.lon]
        rho_f = polyTEOS10_bsq(particle.depth, S, T)
        beta = 3*rho_f/(2*rho_p + rho_f)
        tau_p = alpha*alpha/(3*beta*nu)
        v_s_1 = (1 - beta)*g*tau_p
        depth_1 = particle.depth + v_s_1*dt*0.5
        T1 = fieldset.cons_temperature[time + .5*dt, depth_1,
                                       particle.lat, particle.lon]
        S1 = fieldset.abs_salinity[time + .5*dt, depth_1,
                                   particle.lat, particle.lon]
        # --
        rho_f = polyTEOS10_bsq(depth_1, S1, T1)
        beta = 3*rho_f/(2*rho_p + rho_f)
        tau_p = alpha*alpha/(3*beta*nu)  # alpha*alpha
        v_s_2 = (1 - beta)*g*tau_p
        depth_2 = particle.depth + v_s_2*dt*0.5
        T2 = fieldset.cons_temperature[time + .5*dt, depth_2,
                                       particle.lat, particle.lon]
        S2 = fieldset.abs_salinity[time + .5*dt, depth_2,
                                   particle.lat, particle.lon]
        # --
        rho_f = polyTEOS10_bsq(depth_2, S2, T2)
        beta = 3*rho_f/(2*rho_p + rho_f)
        tau_p = alpha*alpha/(3*beta*nu)  # alpha*alpha
        v_s_3 = (1 - beta)*g*tau_p
        depth_3 = particle.depth + v_s_3*dt
        T3 = fieldset.cons_temperature[time + dt, depth_3,
                                       particle.lat, particle.lon]
        S3 = fieldset.abs_salinity[time + dt, depth_3,
                                   particle.lat, particle.lon]
        # --
        rho_f = polyTEOS10_bsq(depth_3, S3, T3)
        beta = 3*rho_f/(2*rho_p + rho_f)
        tau_p = alpha*alpha/(3*beta*nu)  # alpha*alpha
        v_s_4 = (1 - beta)*g*tau_p

        v_s = (v_s_1 + 2 * v_s_2 + 2 * v_s_3 + v_s_4)/6.

    else:
        v_s = 0

    particle.depth = particle.depth + v_s*dt
    particle.v_s = v_s
