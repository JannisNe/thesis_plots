import logging
import matplotlib.pyplot as plt
import json
import pickle
from pathlib import Path

from IPython.core.pylabtools import figsize

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

    ylim = {
        "1": (8e-16, 1e-14),
        "2": (4e-9, 1e-8)
    }

    height = plt.rcParams["figure.figsize"][1] * 2 / 3
    width = plt.rcParams["figure.figsize"][0]
    fig, axs = plt.subplots(ncols=2, sharex=True, gridspec_kw={"wspace": 0.3}, figsize=(width, height))
    for ax, (gamma, res) in zip(axs, energy_range_results.items()):
        e_min_gev = res["e_min_gev"]
        e_max_gev = res["e_max_gev"]
        sens_min = res["sens_min"]
        sens_max = res["sens_max"]
        nominal_sens = res["nominal_sensitivity"]
        erange = energy_range_final[float(gamma)]

        low_handle = ax.plot(e_min_gev, sens_min, ls="--", marker="o", ms=2, c="C0")
        hi_handle = ax.plot(e_max_gev, sens_max, ls="-", marker="o", ms=2, c="C0")
        range_handle = ax.fill_betweenx(
           ylim[gamma], [erange[0]], [erange[1]], color="black", alpha=0.2, zorder=0, ec="none",
        )
        ax.axhline(nominal_sens, label="nominal sensitivity", color="black")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(ylim[gamma])
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

    fig.supxlabel(r"$E_\mathrm{cut}$ [GeV]", va="top", ha="center")
    axs[0].set_ylabel("Sensitivity flux [GeV$^{-1}$ cm$^{-2}$ s$^{-1}$]")
    return fig
