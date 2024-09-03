import logging
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import lines, patches
from matplotlib.legend_handler import HandlerPatch
import pandas as pd

from thesis_plots.plotter import Plotter
from thesis_plots.icecube_diffuse_flux import get_diffuse_flux_functions


logger = logging.getLogger(__name__)


# PoS2019, 1016, DOI: 10.22323/1.358.1016
sens_flux_100tev_times_esq = 5.8e-9
sens_gamma = -2.5
sens_erange = (5.5e2, 4.4e5)


@Plotter.register()
def winter_lunardini():
    data_dir = Path(__file__).parent / "data"
    model_filenames = {
        "IR": "winter_lunardini_ir_diffuse_flux.csv",
        "X-ray": "winter_lunardini_xray_diffuse_flux.csv",
        "OUV": "winter_lunardini_ouv_diffuse_flux.csv",
    }
    data = {
        key: pd.read_csv(data_dir / fn, decimal=",", delimiter=";", names=["E", "flux"])
        for key, fn in model_filenames.items()
    }
    exp = np.array([-1, -2, -3])
    start_x = 800 # data["IR"]["E"].iloc[0]
    end_x = 3000
    start_y = 10**-10.8  #data["IR"]["flux"].iloc[0]
    x = np.logspace(np.log10(start_x), np.log10(end_x), 100)
    y = start_y * (x / start_x) ** (exp[:, np.newaxis] + 2) * 1.5
    x_sens = np.logspace(np.log10(sens_erange[0]), np.log10(sens_erange[1]), 3)
    logger.debug(x_sens)
    y_sens = sens_flux_100tev_times_esq * (x_sens / 100e3) ** (sens_gamma + 2)

    best_f, lower_f, upper_f, e_range = get_diffuse_flux_functions("joint_15")

    fig, ax = plt.subplots()
    for ls, (key, v) in zip(["--", ":", "-."], data.items()):
        ax.plot(v["E"], v["flux"], label=key, ls=ls, zorder=2)
    for iy, iexp in zip(y, exp):
        ax.plot(x, iy, ls="--", color="grey", zorder=0)
        ax.annotate(f"$\Phi \propto E^{ {iexp} }$", (x[-1], iy[-1]), color="grey", bbox=dict(facecolor="white", pad=-.5),
                    zorder=1)
    ax.errorbar(x_sens, y_sens, yerr=.2 * y_sens, zorder=3, uplims=True, c="grey")
    ax.fill_between(e_range, lower_f(e_range) * e_range ** 2, upper_f(e_range) * e_range ** 2,
                    color="black", alpha=.2, label="Diffuse Flux", zorder=4, ec="none")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(bottom=1e-12)
    ax.set_xlim(1e3, 1e9)
    ax.set_xlabel("Energy [GeV]")
    ax.set_ylabel(r"$\Phi\,E^2$ [GeV$^{-1}$ cm$^{-2}$ s$^{-1}$ sr$^{-1}$]")

    # Define a custom handler to ensure the arrow is correctly scaled in the legend
    class HandlerArrow(HandlerPatch):
        def create_artists(self, legend, orig_handle,
                           xdescent, ydescent, width, height, fontsize, trans):
            pos_a = (xdescent + 0.5 * width, ydescent + height * 0.9)
            pos_b = (xdescent + 0.5 * width, ydescent + height * 0.1)
            logger.debug(f"Creating arrow from {pos_a} to {pos_b}")
            arrow = patches.FancyArrowPatch(
                pos_a, pos_b,
                arrowstyle=orig_handle.get_arrowstyle(),
                mutation_scale=orig_handle.get_mutation_scale(),
                color=orig_handle.get_facecolor(),
                shrinkA=orig_handle.shrinkA,
                shrinkB=orig_handle.shrinkB,
            )
            arrow.set_transform(trans)
            return [arrow]

    ulim_handle = patches.FancyArrowPatch((0, 0), (0, -0.5), color="grey", arrowstyle="-|>",
                                          mutation_scale=10, shrinkA=0, shrinkB=0)
    legend_handles = ax.get_legend_handles_labels()[0] + [ulim_handle]
    legend_labels = ax.get_legend_handles_labels()[1] + ["TDE Upper Limit"]

    ax.legend(handles=legend_handles, labels=legend_labels, ncol=3, bbox_to_anchor=(0.5, 1.), loc="lower center",
              handler_map={ulim_handle: HandlerArrow()})
    return fig
