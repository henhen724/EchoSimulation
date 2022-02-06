import numpy as np

from scipy.stats import expon, uniform, norm, triang
from scipy.special import comb

from astropy import units as u

# Exponential Distribution


def expDist(size, sigma_J_1, sigma_J_2, sigma_J_3):

    J_1 = expon.rvs(scale=sigma_J_1, loc=0.0, size=size)

    exponential_var2 = expon.rvs(scale=sigma_J_2, loc=0, size=size)
    J_2 = J_1 - exponential_var2

    for i in range(0, size):
        if J_2[i] < 0.0:
            J_2[i] = 0.0001

    exponential_var3 = expon.rvs(scale=sigma_J_3, loc=0, size=size)
    J_3 = J_2 - exponential_var3

    for i in range(0, size):
        if J_3[i] < -J_2[i]:
            J_3[i] = -J_2[i]

    M = uniform.rvs(loc=-np.pi, scale=np.pi, size=size) * u.rad
    w = uniform.rvs(loc=-np.pi, scale=np.pi, size=size) * u.rad
    Omega = uniform.rvs(loc=-np.pi, scale=np.pi, size=size) * u.rad

    return J_1, J_2, J_3, M, w, Omega


# Near Circular Distribution
def nearCirDist(size, r_0, sigma_a, sigma_e):
    expvar1 = expon.rvs(scale=sigma_e, loc=0, size=size)*np.sqrt(1*u.km)

    #J_2 = expon.norm(scale=sigma_a, loc=r_0, size=size)
    J_2 = np.repeat(np.sqrt(r_0), size)

    J_1 = expvar1 + J_2
    J_3 = J_2

    M = uniform.rvs(loc=0, scale=2*np.pi, size=size) * u.rad
    w = uniform.rvs(loc=0, scale=2*np.pi, size=size) * u.rad
    Omega = np.repeat(0.0, size) * u.rad
    return J_1, J_2, J_3, M, w, Omega


# Near Gaussian in R Distribution
def nearGausRDist(size, r_0, sigma_r):
    expvar1 = expon.rvs(scale=1, loc=0, size=size)

    #J_2 = expon.norm(scale=sigma_a, loc=r_0, size=size)
    J_2 = np.repeat(np.sqrt(r_0), size)

    J_1 = np.sqrt(
        (np.power(J_2, 2) + np.sqrt(np.power(J_2, 4) + sigma_r*expvar1))/2.0)
    J_3 = J_2

    M = uniform.rvs(loc=-2*np.pi, scale=2*np.pi, size=size) * u.rad
    w = uniform.rvs(loc=0, scale=2*np.pi, size=size) * u.rad
    Omega = np.repeat(0.0, size) * u.rad
    return J_1, J_2, J_3, M, w, Omega


# Near SHO Distribution
# Here I'm setting J_SHO = sqrt(ma^3/2k) * (p^2/2m + k/a^3*x^2)
# This leads to J_SHO = sqrt(ka/2)e^2 = sqrt(k/2)(J_1 - J_2^2/J_1)
# Then im distributing particles according to exp{- J_SHO/epsilon}
def nearSHODistribution(size, r_0, epsilon, k, m):
    J_SHO = expon.rvs(scale=epsilon, loc=0, size=size)*1*u.kg*u.km**2/u.s

    #J_2 = expon.norm(scale=sigma_a, loc=r_0, size=size)
    J_2 = np.repeat(np.sqrt(r_0), size)

    J_1 = (J_SHO + np.sqrt(np.power(J_SHO, 2) + 4 *
                           k*m*np.power(J_2, 2)))/(2.0*np.sqrt(k*m))
    J_3 = J_2

    M = uniform.rvs(loc=0, scale=2*np.pi, size=size) * u.rad
    w = uniform.rvs(loc=0, scale=2*np.pi, size=size) * u.rad
    Omega = np.repeat(0.0, size) * u.rad
    return J_1, J_2, J_3, M, w, Omega


def beerCanSHODistribution(size, r_0, epsilon, k, m):
    J_SHO = uniform.rvs(scale=epsilon, loc=0, size=size)*1*u.kg*u.km**2/u.s

    J_2 = np.repeat(np.sqrt(r_0), size)

    J_1 = (J_SHO + np.sqrt(np.power(J_SHO, 2) + 4 *
                           k*m*np.power(J_2, 2)))/(2.0*np.sqrt(k*m))
    J_3 = J_2

    M = uniform.rvs(loc=0, scale=2*np.pi, size=size) * u.rad
    w = uniform.rvs(loc=0, scale=2*np.pi, size=size) * u.rad
    Omega = np.repeat(0.0, size) * u.rad
    return J_1, J_2, J_3, M, w, Omega
