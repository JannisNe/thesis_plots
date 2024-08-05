import logging
import matplotlib.pyplot as plt
import numpy as np
from astropy.modeling import models
from astropy import units as u
from thesis_plots.plotter import Plotter
from thesis_plots.instruments.bandpasses import get_filter


logger = logging.getLogger(__name__)


@Plotter.register("margin")
def wise_blackbody():
    temp = np.array([1800, 1500, 1200]) * u.K
    filters = [("wise", "wise", "W1"), ("wise", "wise", "W2")]
    bb = [models.BlackBody(temperature=t) for t in temp]
    tables = {f"{fac}/{inst} ({band})": get_filter(fac, inst, band) for fac, inst, band in filters}
    wl_range = np.array((2e4, 1e5)) * u.AA
    wls = np.logspace(np.log10(wl_range[0].value), np.log10(wl_range[1].value), 1000) * u.AA
    bb_flux = [ibb(wls) for ibb in bb]
    ls = ["--", ":"]
    bbc = "C2"

    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    for ils, (f, table) in zip(ls, tables.items()):
        ax.plot(table["Wavelength"], table["Transmission"], ls=ils, label=f.strip(")").replace("wise/wise (", ""),
                 zorder=10)
    for ibb, ibbmod, itemp in zip(bb_flux, bb, temp):
        ax2.plot(wls, ibb.value / ibb.value.max(), label=f"{itemp.value:} K", color=bbc, zorder=2)
        ax2.annotate(f"{itemp.value:.0f} K", (7e4, ibb.value[-200] / ibb.value.max()),
                    va="center", ha="center", rotation=-45, bbox=dict(facecolor="white", edgecolor="none", alpha=1, pad=0.),
                    color=bbc, fontsize="small")
    ax.legend(bbox_to_anchor=(0.5, 1), loc="lower center", ncols=2)
    ax.set_xscale("log")
    ax.set_xlabel("Wavelength [$10^4$ AA]")
    ax2.set_ylabel("Flux [a.u.]", color=bbc)
    ax2.set(xticks=[2e4, 3e4, 4e4, 6e4, 1e5], xticklabels=["2", "3", "4", "6", "10"], yticks=[], yticklabels=[])
    ax.set_ylabel("Transmission")
    ax.set_zorder(1)
    ax.patch.set_visible(False)
    ax2.spines["right"].set_color(bbc)
    ax.spines["right"].set_color(bbc)

    for a in [ax, ax2]:
        a.tick_params(bottom=plt.rcParams["xtick.bottom"], top=plt.rcParams["xtick.top"],
                      left=plt.rcParams["ytick.left"], right=plt.rcParams["ytick.right"], which="both")

    return fig
