import numpy as np
import matplotlib.pyplot as plt

from astropy import units as u


def displaySimulation(orb):
    size = len(orb)

    x = np.empty(size)*u.km
    y = np.empty(size)*u.km
    v_x = np.empty(size)*u.km/u.s
    v_y = np.empty(size)*u.km/u.s
    r = np.zeros(size)*u.km
    v_r = np.zeros(size)*u.km/u.s

    ecc = np.zeros(size)*u.one
    a = np.zeros(size)*u.km

    nu = np.zeros(size)*u.rad
    L = np.zeros(size)*u.km**2/u.s
    inc = np.zeros(size)*u.rad

    fig, axes = plt.subplots(5, 2, figsize=(14, 40))

    [[position, momentum], [radiusVsRadialMomentum, angleVsAngularMomentum], [radiusHist, radialMomentumHist],
     [eccHist, aHist], [angleHist, inclincationHist]] = axes

    for i in range(0, size):
        x[i] = orb[i].r[0]
        v_x[i] = orb[i].v[0]
        y[i] = orb[i].r[1]
        v_y[i] = orb[i].v[1]

        r[i] = np.linalg.norm(orb[i].r)
        unit_r = orb[i].r/r[i]
        v_r[i] = np.dot(unit_r, orb[i].v*u.s/u.km)*u.km/u.s

        ecc[i] = orb[i].ecc
        a[i] = orb[i].a

        nu[i] = orb[i].nu
        L[i] = np.linalg.norm(np.cross(orb[i].r, orb[i].v - v_r[i]*unit_r))
        inc[i] = orb[i].inc

    position.scatter(x, y, s=0.5)
    position.set_title("Position")
    momentum.scatter(v_x, v_y, s=0.5)
    momentum.set_title("Momentum")

    radiusVsRadialMomentum.scatter(r, v_r, s=0.5)
    radiusVsRadialMomentum.set_title("Radius vs Radial Momentum")
    angleVsAngularMomentum.scatter(nu, L, s=0.5)
    angleVsAngularMomentum.set_title("Angle vs Angular Momentum")

    radiusHist.hist(r.value, bins=25)
    radiusHist.set_title("Radius")
    radialMomentumHist.hist(v_r.value, bins=25)
    radialMomentumHist.set_title("Radial Momentum")

    eccHist.hist(ecc.value, bins=25)
    eccHist.set_title("Eccentricity")
    aHist.hist(a.value, bins=25)
    aHist.set_title("Semi-Major Axis")

    angleHist.hist(nu.value, bins=25)
    angleHist.set_title("Angle")
    inclincationHist.hist(inc.value, bins=25)
    inclincationHist.set_title("Inclination")
