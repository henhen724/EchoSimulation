import numpy as np

from astropy import units as u


from lib.GPUKeplerSimulation import propagate, average_scaler
from lib.AsteroidBelt import generateNearSHODistribution


def runOneEchoAmplitudeMeasurement(size, k, m, r_0, epsilon, dipoleKickStrength, quadrapoleKickStrength, tau, deltaTau, timestep):

    belt = generateNearSHODistribution(size, k, m, r_0, epsilon)

    belt.radial_kick(dipoleKickStrength)

    radiuses, velocities, final_mean_anamoly = propagate(belt, tau, tau/2)
    belt.setMeanAnamoly(final_mean_anamoly)

    belt.radial_quadrapole_kick(quadrapoleKickStrength, r_0)

    radiuses, velocities, final_mean_anamoly = propagate(
        belt, tau-deltaTau, (tau-deltaTau)/2)
    belt.setMeanAnamoly(final_mean_anamoly)

    radiuses, velocities, final_mean_anamoly = propagate(
        belt, 2*deltaTau, timestep)

    avr_radius = average_scaler(radiuses)

    avr_radius_deviation = np.absolute(avr_radius - r_0)

    echoAmp = max(avr_radius_deviation)

    echoIndex = np.argmax(avr_radius_deviation)

    return echoAmp, echoIndex*timestep + tau-deltaTau, avr_radius


size = 20000
k = 0.00112*u.kg*u.km**3/u.s**2
m = 0.1*u.kg
r_0 = 0.05*u.km
epsilon = 0.0000001*u.kg*u.km**2/u.s
deltaTau = 300*u.second
timestep = 0.5*u.second
tau = 1000*u.second


dipoleKickStrengths = np.linspace(0.0, 0.01, 100)*u.km
quadKickStrengths = np.linspace(0.0, 0.1, 100) * u.second**-1

for dipoleKickStrength in dipoleKickStrengths[61:100]:
    for quadKickStrength in quadKickStrengths:
        for j in range(0, 10):
            echoAmp, echoTime, avr_radius = runOneEchoAmplitudeMeasurement(
                size, k, m, r_0, epsilon, dipoleKickStrength, quadKickStrength, tau, deltaTau, timestep)

            with open('data/echo_amp.csv', 'a') as csvfile:
                np.savetxt(csvfile, [(size, k.value, m.value, r_0.value, epsilon.value*(10**6), dipoleKickStrength.value, quadKickStrength.value, tau.value, deltaTau.value,
                           timestep.value, echoAmp.value, echoTime.value)], fmt=['%d', '%f', '%f', '%f', '%f', '%f', '%f', '%f', '%f', '%f', '%f', '%f'], delimiter=',', comments='')
