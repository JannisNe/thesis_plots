import logging
import matplotlib.pyplot as plt
import json
import pickle
from pathlib import Path
import numpy as np

from thesis_plots.plotter import Plotter


logger = logging.getLogger(__name__)


@Plotter.register("wide")
def energy_range():
    data_dir = Path(__file__).parent / "data"
    energy_range_file = data_dir / "energy_range_results.json"
    with open(energy_range_file, "r") as f:
        energy_range_results = json.load(f)
    energy_range_final_file = data_dir / "energy_range_nan.pkl"
    with open(energy_range_final_file, "rb") as f:
        energy_range_final = pickle.load(f)

    height = plt.rcParams["figure.figsize"][1] * 2 / 3
    width = plt.rcParams["figure.figsize"][0]
    fig, axs = plt.subplots(ncols=3, sharex=True, sharey=True, gridspec_kw={"wspace": 0.05}, figsize=(width, height))
    for i, gamma in enumerate(["1", "1.5", "2"]):
        ax = axs[i]
        res = energy_range_results[gamma]
        nominal_sens = res["nominal_sensitivity"]
        e_min_gev = np.array(res["e_min_gev"])
        e_max_gev = np.array(res["e_max_gev"])
        sens_min = np.array(res["sens_min"]) / nominal_sens
        sens_max = np.array(res["sens_max"]) / nominal_sens
        erange = energy_range_final[float(gamma)]

        emin_mask = e_min_gev <= 1e7
        low_handle = ax.plot(e_min_gev[emin_mask], sens_min[emin_mask], ls="--", marker="o", ms=2, c="C0")
        emax_mask = e_max_gev <= 1e7
        hi_handle = ax.plot(e_max_gev[emax_mask], sens_max[emax_mask], ls="-", marker="o", ms=2, c="C0")
        range_handle = ax.fill_betweenx(
            (.5, 3), [erange[0]], [erange[1]], color="black", alpha=0.2, zorder=0, ec="none",
        )
        ax.axhline(1, label="nominal sensitivity", color="black")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim((.8, 3))
        ax.annotate(f"$\gamma = {gamma}$", (0.05, 0.95), xycoords="axes fraction", ha="left", va="top")

    legend_handles = [
        low_handle[0],
        hi_handle[0],
        range_handle,
        plt.Line2D([], [], color="black", label="nominal sensitivity"),
    ]
    legend_labels = [
        r"E = [$E_\mathrm{cut}$, 10 PeV]",
        r"E = [100 GeV, $E_\mathrm{cut}$]",
        "Energy range",
        "Nominal sensitivity",
    ]
    fig.legend(legend_handles, legend_labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.1))

    fig.supxlabel(r"$E_\mathrm{cut}$ [GeV]", va="top", ha="center", y=-0.05)
    axs[0].set_ylabel("Sensitivity / Nominal Sensitivity")
    axs[0].set_xlim(5e2, 10**7.2)
    axs[0].set_xticks([1e3, 1e4, 1e5, 1e6, 1e7])

    axs[0].set_yticks([1, 2, 3])
    axs[0].set_yticklabels(["1", "2", "3"])
    return fig
