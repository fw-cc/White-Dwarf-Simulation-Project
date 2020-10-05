from astropy.constants.iau2015 import R_sun, M_sun
from tqdm import tqdm

import scipy.constants as sciconst
import matplotlib.pyplot as plt

import numpy as np
import math


# Define constants used in the file namespace with the `_f` prefix
# Namespace pollution causes pandemics
_f_m_e = sciconst.electron_mass
_f_m_p = sciconst.proton_mass
_f_c = sciconst.speed_of_light
_f_pi = sciconst.pi
_f_big_g = sciconst.gravitational_constant
_f_hbar = sciconst.hbar


def runkut(n, x, y, h):
    """Advances the solution of diff eqn defined by derivs from x to x+h"""
    y0 = y[:]
    k1 = derivative_field(n, x, y)
    for i in range(1, n + 1):
        y[i] = y0[i] + 0.5 * h * k1[i]
    k2 = derivative_field(n, x + 0.5 * h, y)
    for i in range(1, n + 1):
        y[i] = y0[i] + h * (0.2071067811 * k1[i] + 0.2928932188 * k2[i])
    k3 = derivative_field(n, x + 0.5 * h, y)
    for i in range(1, n + 1):
        y[i] = y0[i] - h * (0.7071067811 * k2[i] - 1.7071067811 * k3[i])
    k4 = derivative_field(n, x + h, y)
    for i in range(1, n + 1):
        a = k1[i] + 0.5857864376 * k2[i] + 3.4142135623 * k3[i] + k4[i]
        y[i] = y0[i] + 0.16666666667 * h * a

    x += h
    return x, y


def derivative_field(_, x_val, state_vals):
    _, mu, q = state_vals
    y_prime = [
        0.0,
        dmu_by_dx(x_val, q),
        dq_by_dx(x_val, q, mu)
    ]
    return y_prime


def dq_by_dx(x_val, q_val, mu_val):
    if q_val < 0 or mu_val < 0 or x_val == 0:
        return 0.0
    gamma_func_result = (
        math.pow(q_val, 2/3) /
        (3 * math.sqrt(1 + math.pow(q_val, 2/3)))
    )
    return (
        (-1 * q_val * mu_val) /
        (x_val * x_val * gamma_func_result)
    )


def dmu_by_dx(x_val, q_val):
    return 3 * q_val * x_val * x_val


def rho_0(y_e_val):
    return (
        (_f_m_p * _f_m_e * _f_m_e * _f_m_e * _f_c * _f_c * _f_c) /
        (3 * _f_pi * _f_pi * _f_hbar * _f_hbar * _f_hbar * y_e_val)
    )


def big_r_0(y_e_val):
    return math.sqrt(
        (3 * y_e_val * _f_m_e * _f_c * _f_c) /
        (4 * _f_pi * _f_big_g * _f_m_p * rho_0(y_e_val))
    )


def mu_0(y_e_val):
    big_r_0_val = big_r_0(y_e_val)
    return (
        (4 * _f_pi * rho_0(y_e_val) * big_r_0_val * big_r_0_val * big_r_0_val) /
        3
    )


def sim_star(step_length, q_c):
    x_val = 0.0
    state_vec = [0.0, 0.0, q_c]
    x_list = []
    mu_list = []
    q_list = []
    while state_vec[2] > 1e-4 * q_c:
        # print(state_vec)
        (x_val, state_vec) = runkut(2, x_val, state_vec, step_length)
        x_list.append(x_val)
        mu_list.append(state_vec[1])
        q_list.append(state_vec[2])
    return x_list, mu_list, q_list


def __test_star_simulation(y_e_val):
    q_c_range = [-10, 20]
    q_c_set = np.around(np.exp(np.linspace(*q_c_range, num=60)), 10)
    # rho_c_true_set = np.around(np.exp(rho_c_set), 10)
    # q_c_set = rho_c_true_set / rho_0(y_e_val)
    star_list = []
    star_edge_radius_list = []
    star_edge_mass_list = []
    for q_c in tqdm(q_c_set):
        x_val = 0.0
        state_vec = [0.0, 0.0, q_c]
        x_list, mu_list, q_list = sim_star(q_c)
        # while state_vec[2] > 1e-3 * q_c:
        #     # print(state_vec)
        #     (x_val, state_vec) = runkut(2, x_val, state_vec, 1e-5)
        #     x_list.append(x_val)
        #     mu_list.append(state_vec[1])
        #     q_list.append(state_vec[2])
        r_list = np.multiply(x_list, big_r_0(y_e_val))
        m_list = np.multiply(mu_list, mu_0(y_e_val))
        rho_list = np.multiply(q_list, rho_0(y_e_val))
        star_list.append((r_list, m_list, rho_list))
        if len(r_list) > 0:
            star_edge_radius_list.append(r_list[-1])
            star_edge_mass_list.append(m_list[-1])
        # print(len(r_list))
    fig, ax = plt.subplots()
    ax.plot(star_list[5][0], star_list[5][2])
    ax.set(xlabel="Radius (m)", ylabel="Density (kgm$^{-3}$)",
           title=f"Test Star Density Radius Relation {q_c_set[5] * rho_0(y_e_val)}:.4f")
    fig.savefig("./iron_star_test.png")
    fig, ax = plt.subplots()
    ax.plot(np.divide(star_edge_mass_list, M_sun.value),
            np.divide(star_edge_radius_list, R_sun.value))
    ax.set(xlabel="Mass", ylabel="Radius",
           title="White Dwarf Mass Radius Relation")
    fig.savefig("./iron_star_mass_radius_test.png")


if __name__ == "__main__":
    plt.tight_layout()
    print(f"Test Iron Star Y_e")
    fe_y_e_val = 26/56
    __test_star_simulation(fe_y_e_val)
