import sim
import plot_gen

from tqdm import tqdm

import matplotlib.pyplot as plt

import numpy as np

import multiprocessing
import json
import os
import functools


def generate_star_data():
    """Returns a set of """
    q_c_range = [-9, 19]
    q_c_set = np.around(np.exp(np.linspace(*q_c_range, num=100)), 10)
    data_ret = []
    for step_length in [1e-6, 1e-5, 1e-4, 1e-3]:
        with multiprocessing.Pool(12) as w_pool:
            sim_star_partial = functools.partial(sim.sim_star, step_length)
            results = w_pool.imap(sim_star_partial, q_c_set)
            star_edge_list_x = []
            star_edge_list_mu = []
            star_result_set = []
            for x_list, mu_list, q_list in tqdm(results, total=len(q_c_set), desc="Solve ODEs"):
                star_edge_list_x.append(x_list[-1])
                star_edge_list_mu.append(mu_list[-1])
                star_result_set.append([x_list, mu_list, q_list])
        data_ret.append((step_length, list(q_c_set), star_result_set, star_edge_list_x, star_edge_list_mu))
    return data_ret


if __name__ == "__main__":
    if not os.path.exists("star_data.json"):
        out_data = generate_star_data()
        file_formatted_data = [
            {"step_length": step_length,
             "q_c_set": q_c_set,
             "star_radii": star_edge_list,
             "star_masses": star_mass_list}
            for step_length, q_c_set, _, star_edge_list, star_mass_list in out_data
        ]
        with open("star_data.json", "w") as json_obj:
            json.dump(file_formatted_data, json_obj, indent=4)
    with open("star_data.json", "r") as json_obj:
        saved_run_data = json.load(json_obj)
    q_c_vals = saved_run_data[0]["q_c_set"]
    radial_edge_list = saved_run_data[0]["star_radii"]
    star_mass_list = saved_run_data[0]["star_masses"]
    mass_radius_step_length_tuple_list = [
        (simulation_data["step_length"], simulation_data["star_masses"], simulation_data["star_radii"])
        for simulation_data in saved_run_data
    ]
    plot_gen.plot_mass_radius_relation(radial_edge_list, star_mass_list)
    plot_gen.plot_mass_radius_relation_relations(mass_radius_step_length_tuple_list)
    x_list, mu_list, q_list = sim.sim_star(1e-6, 10)
    plot_gen.plot_single_white_dwarf_values(x_list, q_list, mu_list, 10)
    exit(0)
