from math import ceil
import numpy as np
import cupy as cp
from lib.ellipj_gpu import wrapped_ellipj_gpu, ellipj as ellipjGPU
from scipy.special import ellipj, ellipkinc, ellipk
from scipy.optimize import minimize
from scipy.stats import expon, uniform
from astropy import units as u


@u.quantity_input
def velocity_error(velocity: u.m/u.s, arr_theta: u.one, x_max: u.m, omega: 1/u.s, m: u.one):
    elliptic_tuple = wrapped_ellipj_gpu(
        arr_theta.to(u.one).value, m.to(u.one).value)
    return np.power(velocity + x_max*elliptic_tuple[0]*elliptic_tuple[2]*omega, 2)


@u.quantity_input
def duffing_total_cost(position: u.m, velocity: u.m/u.s, arr_theta: u.one, x_max: u.m, omega: 1/u.s, m: u.one):
    elliptic_tuple = wrapped_ellipj_gpu(
        arr_theta.to(u.one).value, m.to(u.one).value)
    return np.sum(np.power(velocity + x_max*elliptic_tuple[0]*elliptic_tuple[2]*omega, 2)).value + np.sum(np.power(position - x_max*elliptic_tuple[1], 2)).value


def duffing_cost_jacobian(position: u.m, velocity: u.m/u.s, arr_theta: u.one, x_max: u.m, omega: 1/u.s, m: u.one):
    elliptic_tuple = wrapped_ellipj_gpu(
        arr_theta.to(u.one).value, m.to(u.one).value)

    return (2*(velocity + x_max*elliptic_tuple[0]*elliptic_tuple[2]*omega)*x_max*omega*(elliptic_tuple[1]*np.power(elliptic_tuple[2], 2) - m*elliptic_tuple[0]*np.power(elliptic_tuple[1], 2))).value + (2*(position - x_max*elliptic_tuple[1])*(x_max*elliptic_tuple[0]*elliptic_tuple[2]*omega)).value

# V(x) = 1/2*alpha*x^2 + 1/4*beta*x^4


class DuffingEnsemble:
    size = 0
    alpha = 0 * u.s**-2
    beta = 0 * u.m**-2 * u.s**-2

    # I will be doing this simulation in terms of the maximum position of each oscillator
    # and instead of in terms of action because there is no reasonable way to invert the action
    # function for the duffing oscillator.

    x_max = np.array([])*u.m
    theta = np.array([])*u.one

    @u.quantity_input
    def __init__(self, _size, _alpha: u.s**-2, _beta: u.m**-2 * u.s**-2, epsilon: u.m**2/u.s**2):
        self.size = _size
        self.alpha = _alpha
        self.beta = _beta
        energy = expon.rvs(scale=epsilon.to(u.m**2/u.s**2).value, loc=0.0,
                           size=self.size)*u.km**2/u.s**2
        self.x_max = self.energy_to_x_max(energy)
        self.theta = uniform.rvs(
            loc=0, scale=4*ellipk(self.get_m().value), size=self.size)*u.one

    @u.quantity_input
    def energy_to_x_max(self, energy: u.m**2/u.s**2):
        return np.sqrt((- self.alpha + np.sqrt(self.alpha**2 + 4*self.beta*energy))/self.beta)

    @u.quantity_input
    def get_omega(self):
        return np.sqrt(self.alpha + self.beta*np.power(self.x_max, 2))

    def get_m(self):
        omega = self.get_omega()
        return np.piecewise(omega, [omega <= 0.0, omega > 0.0], [0, self.beta*np.power(self.x_max, 2)/(2*np.power(omega, 2))])

    def get_pos_vel(self):
        omega = self.get_omega()
        m = self.get_m()
        ellipj_tuple = wrapped_ellipj_gpu(
            self.theta.to(u.one).value, m.to(u.one).value)
        return (self.x_max*ellipj_tuple[1], -self.x_max*ellipj_tuple[0]*ellipj_tuple[2]*omega)

    @u.quantity_input
    def set_pos_vel(self, positions: u.m, velocity: u.m/u.s):
        energy = 1/2*np.power(velocity, 2) + 1/2*self.alpha * \
            np.power(positions, 2) + 1/4*self.beta*np.power(positions, 4)
        self.x_max = self.energy_to_x_max(energy)

        omega = self.get_omega()

        m = self.get_m()

        phi_mod_pi = np.arccos(positions/self.x_max)

        u1 = ellipkinc(phi_mod_pi.to(u.rad).value, m.value)*u.one
        u2 = ellipkinc(-phi_mod_pi.to(u.rad).value, m.value)*u.one

        is_u_1_better = velocity_error(velocity, u1, self.x_max, omega, m) < velocity_error(
            velocity, u2, self.x_max, omega, m)

        self.theta = is_u_1_better*u1 + np.logical_not(is_u_1_better)*u2

        if duffing_total_cost(positions, velocity, self.theta, self.x_max, omega, m) > self.size*10.0**-15:
            pos, vel = self.get_pos_vel()
            print("Warning: Solution for the theta array showed a high error: {}  Highest pos err: {}  Highest vel err: {}".format(
                duffing_total_cost(positions, velocity, self.theta, self.x_max, omega, m), np.max(positions - pos), np.max(velocity - vel)))

    @u.quantity_input
    def set_theta(self, _theta: u.one):
        self.theta = _theta

    @u.quantity_input
    def propagate(self, time: u.s):
        omega = self.get_omega()
        self.theta = omega*time + self.theta

    @u.quantity_input
    def get_average_position(self, time: u.s, timestep: u.s):
        num_of_steps = int(ceil(time/timestep))
        times = cp.linspace(0.0, time, num_of_steps)
        omega = self.get_omega().to(1/u.s).value
        m_matrix = cp.einsum("i,j->ij", np.ones(num_of_steps),
                             cp.array(self.get_m().to(u.one).value))
        theta_matrix = cp.einsum("i,j->ij", cp.ones(num_of_steps), cp.array(self.theta.to(u.one).value)) + \
            cp.einsum("i,j->ij", times, omega)
        position = cp.einsum(
            "ij,j->ij", ellipjGPU(theta_matrix, m_matrix)[:][:][1], self.x_max)
        average_position = cp.einsum("ij->i", position)/self.size
        return cp.asnumpy(average_position)*u.m

    @u.quantity_input
    def dipole_kick(self, kick_strength: u.m):
        pos, vel = self.get_pos_vel()
        new_pos = pos + kick_strength*np.ones(self.size)
        self.set_pos_vel(new_pos, vel)

    @u.quantity_input
    def quadrupole_kick(self, quad_strength: u.s**-1):
        pos, vel = self.get_pos_vel()
        new_vel = vel + quad_strength*pos
        self.set_pos_vel(pos, new_vel)
