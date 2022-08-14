import numpy as np
import cupy as cp
from math import comb, factorial, ceil, floor

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


# def arctan(array):

#     gpu_memory = cp.get_default_memory_pool()

#     gpu_memory_size = gpu_memory.get_limit()

#     max_steps_in_one_gpu_memory = floor((
#         gpu_memory_size - 64*array.shape[0]*array.shape[1])/3000)

#     num_of_memory_frames = int(
#         ceil(array.shape[0]/max_steps_in_one_gpu_memory))
#     print(array.shape)
#     answer = cp.zeros(array.shape)

#     for timestep in range(0, array.shape[1]):
#         for memory_frame in range(0, num_of_memory_frames):
#             start_index = memory_frame*max_steps_in_one_gpu_memory
#             end_index = min((memory_frame+1) *
#                             max_steps_in_one_gpu_memory, array.shape[0])
#             answer[start_index:end_index, timestep] = cp.arctan(
#                 array[start_index:end_index, timestep])
#     gpu_memory.free_all_blocks()

#     return answer

def nu_to_E(nu, e, numpyOutput=False):
    if type(nu) == np.ndarray:
        nu = cp.array(nu)
    if type(e) == np.ndarray:
        e = cp.array(e)

    rst = 2.0*cp.arctan(cp.einsum("i,ik->ik",
                                  cp.sqrt((1 - e)/(1 + e)), cp.tan(nu/2.0)))

    if numpyOutput:
        rst = cp.asnumpy(rst)
        gpu_memory = cp.get_default_memory_pool()
        gpu_memory.free_all_blocks()
        return rst
    else:
        return rst


def E_to_nu(E, e, numpyOutput=False):
    if type(E) == np.ndarray:
        E = cp.array(E)
    if type(e) == np.ndarray:
        e = cp.array(e)

    rst = 2.0 * cp.arctan(cp.einsum("i,ik->ik",
                                    cp.sqrt((1 + e) / (1 - e)), cp.tan(E / 2.0)))

    if numpyOutput:
        rst: np.ndarray = cp.asnumpy(rst)
        gpu_memory = cp.get_default_memory_pool()
        gpu_memory.free_all_blocks()
        return rst
    else:
        return rst


def E_to_M(E, e, numpyOutput=False):
    if type(E) == np.ndarray:
        E = cp.array(E)
    if type(e) == np.ndarray:
        e = cp.array(e)

    rst = E - cp.einsum("i,ik->ik", e, cp.sin(E))

    if numpyOutput:
        rst = cp.asnumpy(rst)
        gpu_memory = cp.get_default_memory_pool()
        gpu_memory.free_all_blocks()
        return rst
    else:
        return rst


def M_to_E(M, e, numpyOutput=False):
    if type(M) == np.ndarray:
        M = cp.array(M)
    if type(e) == np.ndarray:
        e = cp.array(e)
    # print("M's shape is ", M.shape, ". e's shape is ", e.shape)
    series = cp.zeros(M.shape)
    sins_of_M = cp.sin(cp.einsum("i,jk->ijk", cp.arange(0, CONST_ROWS+1), M))

    for n in range(1, CONST_ROWS):
        for k in range(0, int(cp.ceil(n/2))):
            A = KEPLER_EQ_INV_CONSTANTS[n, k]
            B = cp.einsum("ij,i->ij", sins_of_M[n-2*k], cp.power(e, n))
            series += A*B

    rst = M + series

    if numpyOutput:
        rst = cp.asnumpy(rst)
        gpu_memory = cp.get_default_memory_pool()
        gpu_memory.free_all_blocks()
        return rst
    else:
        return rst


# The maximum GPU memory is required during the M_to_E step
# The following arrays are in memory
# time_array (steps)
# e (size) -> sizeof(float)*size
# a (size) -> sizeof(float)*size
# T (size) -> sizeof(float)*size
# mean_anamoly (size, steps) -> sizeof(float)*size*steps
# series (size, steps) -> sizeof(float)*size*steps
# sins_of_M (CONST_ROWS+1, size, steps) -> sizeof(float)*(CONST_ROWS+1)*size*steps
# So the total Memory is sizeof(float)*((CONST_ROWS+1 + 2)*size*steps + 3*size)
FLOAT_SIZE = 8


def _propagate_single_timestep_memory_(size):
    return 5*FLOAT_SIZE*(CONST_ROWS+3)*size


def _propagate_fixed_memory_(size):
    return FLOAT_SIZE*3*size


def _propagate_single_gpu_memory_(belt, time_array, num_of_steps):

    time_array = cp.array(time_array)

    # before_obj_extract = time.time()

    initial_mean_anamoly = cp.array(belt.M.to(u.rad).value)

    # print(initial_mean_anamoly)

    e = cp.array(belt.ecc.value)

    # print(belt.ecc.value)

    a = cp.array(belt.a.to(u.km).value)
    T = (2*cp.pi*cp.power(cp.array(belt.a.to(u.km).value), 3.0/2.0) /
         cp.sqrt(belt.micro.to(u.km**3/u.s**2).value))

    # before_E_calc = time.time()

    fraction_of_period = cp.einsum("i,j->ij", 1/T, time_array)
    mean_anamoly = (fraction_of_period % 1)*2*cp.pi + \
        initial_mean_anamoly[:, cp.newaxis]

    # print(mean_anamoly)

    eccentric_anamoly = M_to_E(mean_anamoly, e)

    # print(eccentric_anamoly)

    delta_r = -cp.einsum("i,ik -> ik", e, cp.cos(eccentric_anamoly))
    radiuses = cp.einsum("i,ik -> ik", a, (1 + delta_r))
    r_mom_num = cp.einsum("i,ik -> ik", 2*cp.pi*a*e, cp.sin(eccentric_anamoly))
    r_mon_denom = cp.einsum("i,ik -> ik", T, 1 + delta_r)
    radial_momenta = r_mom_num/r_mon_denom

    final_mean_anamoly = mean_anamoly[:, num_of_steps - 1]

    # before_ret = time.time()

    # print(" Obj Extract: ", before_E_calc - before_obj_extract,
    #       " E Calc: ", before_ret - before_E_calc)

    return cp.asnumpy(radiuses)*u.km, cp.asnumpy(radial_momenta)*u.km/u.s, cp.asnumpy(final_mean_anamoly)*u.rad


def propagate(belt, time_of_flight, timestep):

    num_of_steps = int(time_of_flight.to(u.s)/timestep.to(u.s))

    gpu_memory = cp.get_default_memory_pool()

    gpu_memory_size = gpu_memory.get_limit()

    # print("GPU memory size: ", gpu_memory_size)

    max_steps_in_one_gpu_memory = floor(
        (gpu_memory_size - _propagate_fixed_memory_(belt.size))/_propagate_single_timestep_memory_(belt.size))
    memory_frames = int(ceil(num_of_steps/max_steps_in_one_gpu_memory))

    # print("Running ", memory_frames, " memory frames")

    time_array = np.linspace(0.0, time_of_flight, num_of_steps)

    # print(time_array)

    radiuses = np.zeros((belt.size, num_of_steps))
    radial_momentum = np.zeros((belt.size, num_of_steps))
    final_mean_anamoly = None
    time_array = cp.linspace(0.0, time_of_flight, num_of_steps)

    for i in range(0, memory_frames):
        start_index = i*max_steps_in_one_gpu_memory
        end_index = min((i+1)*max_steps_in_one_gpu_memory, num_of_steps)

        # print("Start index: ", start_index, " End index: ", end_index)

        radiuses[:, start_index:end_index], radial_momentum[:, start_index:end_index], final_mean_anamoly = _propagate_single_gpu_memory_(
            belt, time_array[start_index:end_index], end_index-start_index)

    gpu_memory.free_all_blocks()

    return cp.asnumpy(radiuses)*u.km, cp.asnumpy(radial_momentum)*u.km/u.s, cp.asnumpy(final_mean_anamoly)*u.rad
    # before_mempool = time.time()

    num_of_steps = int(time_of_flight.to(u.s)/timestep.to(u.s))

    mempool = cp.get_default_memory_pool()
    memory_gpu_size = mempool.get_limit()

    # before_batching = time.time()

    if memory_gpu_size == 0:
        mempool.set_limit(fraction=0.8)
        memory_gpu_size = mempool.get_limit()
    max_steps_in_one_gpu_memory = floor(
        (memory_gpu_size - _propagate_fixed_memory_(belt.size))/_propagate_single_timestep_memory_(belt.size))
    memory_frames = int(ceil(num_of_steps/max_steps_in_one_gpu_memory))

    # print("MEM LIMIT: ", memory_gpu_size)

    # print("FIXED MEM: ", _propagate_fixed_memory_(belt.size),
    #       "  STEP MEMORY: ", _propagate_single_timestep_memory_(belt.size))

    # print("Max Steps: ", (memory_gpu_size - _propagate_fixed_memory_(belt.size))/_propagate_single_timestep_memory_(belt.size),
    #       "  Frames: ", memory_frames)

    # before_init = time.time()

    time_array = np.linspace(0.0, time_of_flight, num_of_steps)

    radiuses = np.empty((belt.size, num_of_steps))*u.km
    radial_momentum = np.empty((belt.size, num_of_steps))*u.km/u.s
    final_mean_anamoly = None

    # print("Timing: Mempool ", before_batching-before_mempool, ", Batching ",
    #       before_init-before_batching, ", Initialization ", time.time() - before_init)

    # print("GPU Memory in use before frames: ", mempool.used_bytes())

    for i in range(0, memory_frames):
        start_index = i*max_steps_in_one_gpu_memory
        end_index = min((i+1)*max_steps_in_one_gpu_memory, num_of_steps)
        # print("Running propagation from ", start_index, " to ", end_index)
        radiuses[:, start_index:end_index], radial_momentum[:, start_index:end_index], final_mean_anamoly = _propagate_single_gpu_run_(
            belt, time_array[start_index:end_index], end_index-start_index)
        mempool.free_all_blocks()
        # print("After clearing: ", mempool.used_bytes())

    return radiuses, radial_momentum, final_mean_anamoly


def average_scaler(quantity, start=None, end=None):
    if start is None:
        start = 0
    if end is None:
        end = quantity.shape[0]
    return cp.asnumpy(cp.einsum('ij->j', quantity[:, start:end].value))*quantity.unit/quantity.shape[0]


def average_3vector(quantity, start=None, end=None):
    if start is None:
        start = 0
    if end is None:
        end = quantity.shape[0]
    return cp.asnumpy(cp.einsum('ijk->jk', quantity[:, start:end, :].value))*quantity.unit/quantity.shape[0]
