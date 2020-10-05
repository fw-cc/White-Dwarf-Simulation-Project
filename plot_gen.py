import sim

from matplotlib import rc
import matplotlib.pyplot as plt
import numpy as np

from astropy.constants.iau2015 import R_sun, M_sun
import matplotlib.font_manager

plt.tight_layout()

rc('font', **{
    'family': 'serif',
    'serif': ['Computer Modern'],
    'size': '20',
})
rc('text', usetex=True)
rc('figure', **{'autolayout': True})

# Uses solar units
_f_known_white_dwarf_data = {
    "names": ["Sirius B", "40 Eri B", "Stein 2051"],
    "masses": [1.053, 0.48, 0.50],
    "mass_err": [0.028, 0.02, 0.05],
    "radii": [0.0074, 0.0124, 0.0115],
    "radius_err": [0.0006, 0.0005, 0.0012],
    "annotation_configs": [
        {
            "xytext": (8, 8),
            "va": "bottom"
        },
        {
            "xytext": (8, 8),
            "va": "bottom"
        },
        {
            "xytext": (-60, -8),
            "va": "top"
        },
    ]
}

fe_y_e = 26 / 56
c_y_e = 0.5
fe_big_r_0 = sim.big_r_0(fe_y_e)
c_big_r_0 = sim.big_r_0(c_y_e)
fe_mu_0 = sim.mu_0(fe_y_e)
c_mu_0 = sim.mu_0(c_y_e)
fe_rho_0 = sim.rho_0(fe_y_e)
c_rho_0 = sim.rho_0(c_y_e)
rho_sun = M_sun.value / (4/3 * np.pi * R_sun * R_sun * R_sun)


def plot_mass_radius_relation_relations(mass_radius_sets_with_step_length):
    fig, ax = plt.subplots()
    linestyles = [":", "--", "-"]
    # colours = ["#910606", "#0306ab", "black"]
    colours = ["0.5", "0.3", "0.0"]
    max_res_data = mass_radius_sets_with_step_length[0]
    for (step_length, mass_set, radius_set), linestyle, colour in zip(mass_radius_sets_with_step_length[1:],
                                                                      linestyles, colours):
        star_mass_diff = np.log10(np.abs(np.subtract(max_res_data[1], mass_set)))
        star_rad_diff = np.log10(np.abs(np.subtract(max_res_data[2], radius_set)))
        ax.plot(star_mass_diff, star_rad_diff, label=f"{step_length:.2E}", color=colour)
    ax.set(xlabel=r"$\log_{10}\mathrm{|M_{max\,res}-M_{lower\,res}|}$",
           ylabel=r"$\log_{10}\mathrm{|R_{max\,res}-R_{lower\,res}|}$",
           title="RK4 Error Approximation Plot")
    fig.legend(loc="center right")
    fig.savefig("./test_error_plot.pdf")


def plot_mass_radius_relation(edge_x_set, edge_mu_set):

    fe_edge_r_set = np.multiply(edge_x_set, fe_big_r_0 / R_sun.value)
    c_edge_r_set = np.multiply(edge_x_set, c_big_r_0 / R_sun.value)

    fe_edge_m_set = np.multiply(edge_mu_set, fe_mu_0 / M_sun.value)
    c_edge_m_set = np.multiply(edge_mu_set, c_mu_0 / M_sun.value)

    fig, ax = plt.subplots()
    ax.plot(fe_edge_m_set, fe_edge_r_set,
            color="0.3", linestyle="--", label="$^{56}$Fe")
    ax.plot(c_edge_m_set, c_edge_r_set,
            color="0.3", linestyle=":", label="$^{12}$C")
    ax.set(ylabel=r"Radius (R$_\odot$)", xlabel=r"Mass (M$_\odot$)",
           ylim=(0.0, 0.04), xlim=(0.0, 2.0))
    ax.set_title("White Dwarf Mass Radius Relationship", pad=20)
    # ax.text(fe_edge_m_set[-1], 0.0, r"{:.3f}".format(fe_edge_m_set[-1]),
    #         ha="left", va="bottom", xytext=(-16, 0), textcoords="offsetpoints")
    # ax.text(c_edge_m_set[-1], 0.0, r"{:.3f}".format(c_edge_m_set[-1]),
    #         ha="left", va="bottom")
    ax.annotate(r"{:.4f}".format(fe_edge_m_set[-1]), (fe_edge_m_set[-1], 0.0),
                textcoords="offset points", va="bottom", xytext=(-65, 0), ha="left")
    ax.annotate(r"{:.4f}".format(c_edge_m_set[-1]), (c_edge_m_set[-1], 0.0),
                textcoords="offset points", va="bottom", ha="left", xytext=(0, 0))
    names = [name for name in _f_known_white_dwarf_data["names"]]
    mass_vals = [mass for mass in _f_known_white_dwarf_data["masses"]]
    radius_vals = [radius for radius in _f_known_white_dwarf_data["radii"]]
    mass_errs = [mass_err for mass_err in _f_known_white_dwarf_data["mass_err"]]
    radius_errs = [radius_err for radius_err in _f_known_white_dwarf_data["radius_err"]]
    ax.errorbar(mass_vals, radius_vals, xerr=mass_errs, yerr=radius_errs, ls="none",
                color="0.2", capsize=0, elinewidth=1.5, zorder=10)
    for star_index in range(len(names)):
        ax.annotate(names[star_index], (mass_vals[star_index], radius_vals[star_index]),
                    textcoords="offset points", zorder=20,
                    **_f_known_white_dwarf_data["annotation_configs"][star_index])
    ax.legend()
    fig.savefig("./mass_radius_relation.png", dpi=600)
    fig.savefig("./mass_radius_relation.svg")
    fig.savefig("./mass_radius_relation.pdf")


def plot_single_white_dwarf_values(radial_steps, density_steps, mass_steps, central_density):
    fig, ax_density = plt.subplots()
    ax_density.set(xlabel=r"Radius (R$_\odot$)", ylabel=r"Density ($\mathrm{kgm^-3}$)")
    ax_density.set_title(r"Single White Dwarf, $q_c=10$", pad=28)
    ax_mass = ax_density.twinx()
    ax_mass.set(ylabel=r"Mass (M$_\odot$)")
    c_radius_solar = np.multiply(radial_steps, c_big_r_0 / R_sun.value)
    fe_radius_solar = np.multiply(radial_steps, fe_big_r_0 / R_sun.value)
    c_mass_solar = np.multiply(mass_steps, c_mu_0 / M_sun.value)
    fe_mass_solar = np.multiply(mass_steps, fe_mu_0 / M_sun.value)
    c_density_solar = np.multiply(density_steps, c_rho_0)
    fe_density_solar = np.multiply(density_steps, fe_rho_0)
    ax_mass.plot(c_radius_solar, c_mass_solar, label=r"$^{12}$C", color="0.1",
                 linestyle="--")
    ax_mass.plot(fe_radius_solar, fe_mass_solar, label=r"$^{56}$Fe", color="0.4",
                 linestyle=":")
    ax_mass.legend(loc="center right")
    ax_density.plot(c_radius_solar, c_density_solar, color="0.1", linestyle="--")
    ax_density.plot(fe_radius_solar, fe_density_solar, color="0.4", linestyle=":")
    fig.savefig(f"./single_white_dwarf_example_sim.pdf")
