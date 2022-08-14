from scipy.special import ellipj as ellipjSCI
from lib.ellipj_gpu import ellipj as ellipjGPU
import numpy as np
import cupy as cp


u = np.linspace(-5.0, 5.0, 100)
m = np.linspace(-1.0, 2.0, 100)
uG, mG = np.meshgrid(u, m)

snSCI, cnSCI, dnSCI, phSCI = ellipjSCI(uG, mG)
snGPUcp, cnGPUcp, dnGPUcp, phGPUcp = ellipjGPU(cp.array(uG), cp.array(mG))
snGPU = cp.asnumpy(snGPUcp)
cnGPU = cp.asnumpy(cnGPUcp)
dnGPU = cp.asnumpy(dnGPUcp)
phGPU = cp.asnumpy(phGPUcp)

if not np.all(np.isnan(snSCI) == np.isnan(snGPU)):
    print("The NaNs did not match between the scipy version and custom version of sn.")
if not np.all(np.isnan(cnSCI) == np.isnan(cnGPU)):
    print("The NaNs did not match between the scipy version and custom version of cn.")
if not np.all(np.isnan(dnSCI) == np.isnan(dnGPU)):
    print("The NaNs did not match between the scipy version and custom version of dn.")
if not np.all(np.isnan(phSCI) == np.isnan(phGPU)):
    print("The NaNs did not match between the scipy version and custom version of ph.")


u = np.linspace(-5.0, 5.0, 100)
m = np.linspace(0.0, 1.0, 100)
uG, mG = np.meshgrid(u, m)

snSCI, cnSCI, dnSCI, phSCI = ellipjSCI(uG, mG)
snGPUcp, cnGPUcp, dnGPUcp, phGPUcp = ellipjGPU(cp.array(uG), cp.array(mG))
snGPU = cp.asnumpy(snGPUcp)
cnGPU = cp.asnumpy(cnGPUcp)
dnGPU = cp.asnumpy(dnGPUcp)
phGPU = cp.asnumpy(phGPUcp)

if np.any(snSCI - snGPU > 2**-48):
    print("The values did not match between the scipy version and custom version of sn.")
if np.any(cnSCI - cnGPU > 2**-48):
    print("The values did not match between the scipy version and custom version of cn.")
if np.any(dnSCI - dnGPU > 2**-48):
    print("The values did not match between the scipy version and custom version of dn.")
if np.any(phSCI - phGPU > 2**-48):
    print("The values did not match between the scipy version and custom version of ph.")
