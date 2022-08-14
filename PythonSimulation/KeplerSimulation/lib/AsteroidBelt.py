import numpy as np
import matplotlib.pyplot as plt
import time as time_pkg

from astropy import units as u

from poliastro.core.elements import rv2coe, coe2rv_many

from numba import njit as jit, prange

from scipy.stats import expon, uniform

from lib.GPUKeplerSimulation import propagate, average_scaler, M_to_E, E_to_nu, nu_to_E, E_to_M
from lib.DisplaySimulation import displaySimulation


@jit(parallel=True)
def rv2coe_many(k, r, v, tol=1e-8):
    n = k.shape[0]
    p = np.zeros(n)
    ecc = np.zeros(n)
    inc = np.zeros(n)
    raan = np.zeros(n)
    argp = np.zeros(n)
    nu = np.zeros(n)
    for i in prange(n):
        p[i], ecc[i], inc[i], raan[i], argp[i], nu[i] = rv2coe(
            k[i], r[i], v[i])

    return p, ecc, inc, raan, argp, nu


class AsteroidBelt:
    size = 0
    micro = 1*u.km**3/u.s**2  # V(r) = - m*micro/r
    a = np.array([])*u.km
    ecc = np.array([])*u.one
    inc = np.array([])*u.rad
    Omega = np.array([])*u.rad
    w = np.array([])*u.rad
    M = np.array([])*u.rad

    def __init__(self, _size, _micro, _a, _ecc, _inc, _Omega, _w, _M):
        self.size = _size
        self.micro = _micro
        if np.any(np.isnan(_a)):
            print("Warning: semi-major axis contains NaN values.")
        self.a = _a
        if np.any(np.isnan(_ecc)):
            print("Warning: eccentricity contains NaN values.")
        self.ecc = _ecc
        if np.any(np.isnan(_inc)):
            print("Warning: Incline contains NaN values.")
        self.inc = _inc
        if np.any(np.isnan(_Omega)):
            print("Warning: Omega contains NaN values.")
        self.Omega = _Omega
        if np.any(np.isnan(_w)):
            print("Warning: lower case omega contains NaN values.")
        self.w = _w
        if np.any(np.isnan(_M)):
            print("Warning: mean anomaly contains NaN values.")
        self.M = _M

    def setMeanAnamoly(self, _M):
        self.M = _M

    def convert_belt_to_pos_vel(self):
        nu = np.squeeze(E_to_nu(M_to_E(np.array([self.M.to(u.rad).value]).T,
                                       self.ecc.value), self.ecc.value, numpyOutput=True))

        size = len(nu)
        positions = np.zeros((size, 3))
        velocities = np.zeros((size, 3))
        k = np.full(size, self.micro)

        positions, velocities = coe2rv_many(
            k, self.a*(1-np.square(self.ecc)), self.ecc, self.inc, self.Omega, self.w, nu*u.rad)

        positions = positions*u.km
        velocities = velocities*u.km/u.s

        return k, positions, velocities

    def convert_pos_vel_to_belt(self, k, positions, velocities):
        p, self.ecc, self.inc, self.Omega, self.w, nu = rv2coe_many(
            k, positions, velocities)
        p, self.ecc, self.inc, self.Omega, self.w, nu = p*u.km, self.ecc * \
            u.one, self.inc*u.rad, self.Omega*u.rad, self.w*u.rad, nu*u.rad
        self.a = p/(1 - np.square(self.ecc))
        self.M = np.squeeze(E_to_M(nu_to_E(np.array(
            [nu.to("rad").value]).T, self.ecc.value), self.ecc.value, numpyOutput=True))*u.rad

    def radial_kick(self, kickStrength):
        k, positions, velocities_at_kick = self.convert_belt_to_pos_vel()

        old_radius = np.linalg.norm(positions, axis=1)
        unit_r = np.einsum("ik,i->ik", positions, 1/old_radius)
        old_r_0 = self.a*(1 - np.square(self.ecc))
        newPosition = positions + kickStrength*unit_r
        radialVelocity = np.einsum(
            "ik,ik,ij->ij", velocities_at_kick, unit_r, unit_r)
        oldThetaVelocity = velocities_at_kick - radialVelocity
        newThetaVelocity = np.einsum(
            "i,ik->ik", old_radius/(old_radius + kickStrength), oldThetaVelocity)
        newVelocity = radialVelocity + newThetaVelocity

        self.convert_pos_vel_to_belt(k, newPosition, newVelocity)
        # new_r_0 = self.a*(1 - np.square(self.ecc))
        # if old_r_0 - new_r_0 > 10**-12*u.km:
        #     print("L_2 before and after do not match: ", old_r_0, "\t", new_r_0)

    def radial_quadrapole_kick(self, quad_strength, radius_offset):
        k, positions, velocities_at_kick = self.convert_belt_to_pos_vel()

        old_radius = np.linalg.norm(positions, axis=1)
        unit_r = np.einsum("ik,i->ik", positions, 1/old_radius)
        newVelocity = velocities_at_kick + \
            np.einsum("i,ik->ik", quad_strength *
                      (old_radius - radius_offset), unit_r)

        self.convert_pos_vel_to_belt(k, positions, newVelocity)

    #         if (np.linalg.norm(orb[i].r) > old_radius):
    #             print("The new radius is greater than the original.", np.linalg.norm(orb[i].r), old_radius)
    #             print(orb[i].r, old_radius)
    #             print("Radius should be",np.linalg.norm(newPosition) )


def createAstroidBeltFromActionAngleCoords(size, micro, m, J_1, J_2, J_3, M, w, Omega):
    a = np.square(J_1/m)/micro

    ecc = np.sqrt(1 - np.square(J_2/J_1)) * u.one
    inc = np.arccos(J_3/J_2)
    return AsteroidBelt(size, micro, a, ecc, inc, Omega, w, M)


def generateNearSHODistribution(size, micro, m, r_0, epsilon):
    # Near SHO Distribution
    # The equibrium distance r_0 = a*(1-e^2)
    # expanding the hamiltonian to the second order in r:
    # H = p^2/2m + m*micro/(a^3(1-e)^3)*(r-r_0)^2 + m*micro/(2r_0) - micro*m/r_0
    # This implies that H_SHO ~ p^2/2m + m*micro/(a^3(1-e^2)^3)*(r-r_0)^2, since this is an approximate SHO.
    # Pluging solutions for r and p_r leads to H_SHO = m*micro*e^2/(a*(1-e^2)^3) + f(angle), where f is a small angle dependent function
    # The frequency of this ossilator is sqrt(2micro/(a^3(1-e^2)^3)) = sqrt(2micro)/r_0^3
    # So J_SHO = m*e^2*sqrt(micro*a/(2*(1-e^2)^3)) = J_1/sqrt(2)*(J_1^3/J_2^3 - J_1/J_2)
    # Then im distributing particles according to exp{- J_SHO/epsilon}

    L = m*np.sqrt(2*micro*r_0)

    J_1 = expon.rvs(scale=epsilon, loc=L, size=size)*u.kg*u.km**2/u.s

    J_2 = np.repeat(L, size)

    J_3 = J_2

    M = uniform.rvs(loc=-2*np.pi, scale=2*np.pi, size=size) * u.rad
    w = uniform.rvs(loc=0, scale=2*np.pi, size=size) * u.rad
    Omega = np.repeat(0.0, size) * u.rad
    return createAstroidBeltFromActionAngleCoords(size, micro, m, J_1, J_2, J_3, M, w, Omega)

# # Exponential Distribution
# size = 100

# J_1 = expon.rvs(scale=1, loc=0.0, size=size)

# exponential_var2 = expon.rvs(scale=0.01, loc=0, size=size)
# J_2 = J_1 - exponential_var2

# for i in range(0, size):
#     if J_2[i] < 0.0:
#         J_2[i] = 0.0001

# exponential_var3 = expon.rvs(scale=0.1, loc=0, size=size)
# J_3 = J_2 - exponential_var3

# for i in range(0, size):
#     if J_3[i] < -J_2[i]:
#         J_3[i] = -J_2[i]

# M = uniform.rvs(loc=-np.pi, scale=np.pi, size=size) * u.rad
# w = uniform.rvs(loc=-np.pi, scale=np.pi, size=size) * u.rad
# Omega = uniform.rvs(loc=-np.pi, scale=np.pi, size=size) * u.rad


# # Near Circular Distribution
# size = 500
# r_0 = 1*u.km
# sigma_e = 0.01*np.sqrt(1*u.km)
# sigma_a = 0.000*np.sqrt(1*u.km)
# alpha_over_m = 1*u.km**3/u.s**2


# expvar1 = expon.rvs(scale=sigma_e, loc=0, size=size)*np.sqrt(1*u.km)

# #J_2 = expon.norm(scale=sigma_a, loc=r_0, size=size)
# J_2 = np.repeat(np.sqrt(r_0), size)

# J_1 = expvar1 + J_2
# J_3 = J_2

# M = uniform.rvs(loc=0, scale=2*np.pi, size=size) * u.rad
# w = uniform.rvs(loc=0, scale=2*np.pi, size=size) * u.rad
# Omega = np.repeat(0.0, size) * u.rad
