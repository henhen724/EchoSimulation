import cupy as cp
from math import comb, factorial, ceil

from astropy import units as u

import time

CONST_ROWS = 13
KEPLER_EQ_INV_CONSTANTS = cp.zeros((CONST_ROWS + 1, int(CONST_ROWS/2) + 1))

KEPLER_EQ_INV_CONSTANTS[0, 0] = 1.0

for n in range(1, CONST_ROWS+1):
    for k in range(0, int(cp.ceil(n/2))):
        KEPLER_EQ_INV_CONSTANTS[n, k] = cp.power(-1, k)*comb(
            n, k)*cp.power(n/2.0-k, n-1)/factorial(n)
        if KEPLER_EQ_INV_CONSTANTS[n, k] == 0:
            print("recived a zero for ", n, ",", k)
        if KEPLER_EQ_INV_CONSTANTS[n, k] > 1.0:
            print("recived ",
                  KEPLER_EQ_INV_CONSTANTS[n, k], " for ", n, ",", k)


def nu_to_E(nu, e):
    return 2*cp.arctan(cp.sqrt((1 - e)/(1 + e))*cp.tan(nu/2.0))


def E_to_M(E, e):
    return E - e*cp.sin(E)


def M_to_E(M, e):
    series = cp.zeros(M.shape)
    sins_of_M = cp.sin(cp.einsum("i,jk->ijk", cp.arange(0, CONST_ROWS+1), M))
    for n in range(1, CONST_ROWS):
        for k in range(0, int(cp.ceil(n/2))):
            series += KEPLER_EQ_INV_CONSTANTS[n,
                                              k]*cp.einsum("ij,i->ij", sins_of_M[n-2*k], cp.power(e, n))
    return M + series


def propagate(belt, time_of_flight, timestep):
    before_init = time.time()
    num_of_steps = int(time_of_flight.to(u.s)/timestep.to(u.s))
    # if num_of_steps*CONST_ROWS*belt.size*8 > cp.get_default_memory_pool().get_limit():
    #     LIMIT = cp.get_default_memory_pool().get_limit()
    #     gpu_mem_swap_iterations = int(
    #         ceil(num_of_steps*CONST_ROWS*belt.size*8.0/LIMIT))

    time_array = cp.linspace(0.0, time_of_flight, num_of_steps)
    size = belt.size
    final_mean_anamoly = cp.zeros(size)
    e = cp.zeros(size)
    a = cp.zeros(size)
    T = cp.zeros(size)

    before_obj_extract = time.time()

    e = cp.array(belt.ecc.value)
    initial_mean_anamoly = cp.array(belt.M.to(u.rad).value)
    a = cp.array(belt.a.to(u.km).value)
    T = (2*cp.pi*cp.power(cp.array(belt.a.to(u.km).value), 3.0/2.0) /
         cp.sqrt(belt.micro.to(u.km**3/u.s**2).value))

    before_E_calc = time.time()

    fraction_of_period = cp.einsum("i,j->ij", 1/T, time_array)
    mean_anamoly = (fraction_of_period % 1)*2*cp.pi + \
        initial_mean_anamoly[:, cp.newaxis]
    final_mean_anamoly = mean_anamoly[:, num_of_steps - 1]
    eccentric_anamoly = M_to_E(mean_anamoly, e)
    delta_r = -cp.einsum("i,ik -> ik", e, cp.cos(eccentric_anamoly))
    radiuses = cp.einsum("i,ik -> ik", a, (1 + delta_r))
    r_mom_num = cp.einsum("i,ik -> ik", 2*cp.pi*a*e, cp.sin(eccentric_anamoly))
    r_mon_denom = cp.einsum("i,ik -> ik", T, 1 + delta_r)
    radial_momenta = r_mom_num/r_mon_denom

    before_ret = time.time()

    print("Initialization: ", before_obj_extract - before_init, " Obj Extract: ",
          before_E_calc - before_obj_extract, " E Calc: ", before_ret - before_E_calc)

    return cp.asnumpy(radiuses)*u.km, cp.asnumpy(radial_momenta)*u.km/u.s, cp.asnumpy(final_mean_anamoly)*u.rad


def average_scaler(quantity, start=None, end=None):
    if start is None:
        start = 0
    if end is None:
        end = quantity.shape[2]
    return cp.asnumpy(cp.einsum('ij->j', quantity[:, start:end].value))*quantity.unit/quantity.shape[0]


def average_3vector(quantity, start=None, end=None):
    if start is None:
        start = 0
    if end is None:
        end = quantity.shape[2]
    return cp.asnumpy(cp.einsum('ijk->jk', quantity[:, start:end, :].value))*quantity.unit/quantity.shape[0]
