import logging
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
import pandas as pd

from thesis_plots.plotter import Plotter
from thesis_plots.icecube_diffuse_flux import load_spectrum
from thesis_plots.arrow_handler import HandlerArrow


logger = logging.getLogger(__name__)


# PoS2019, 1016, DOI: 10.22323/1.358.1016
sens_flux_100tev_times_esq = 5.8e-9
sens_gamma = -2.5
sens_erange = (5.5e2, 4.4e5)


model_colors = {
    "IR": "C3",
    "X-ray": "C0",
    "OUV": "C2",
}
model_gammas = {
    "IR": -1,
    "X-ray": -2.5,
    "OUV": -1.5,
}


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
    x_sens = np.logspace(np.log10(sens_erange[0]), np.log10(sens_erange[1]), 3)
    logger.debug(x_sens)
    y_sens = sens_flux_100tev_times_esq * (x_sens / 100e3) ** (sens_gamma + 2)

    s = load_spectrum("joint15")
    srange = s.get_energy_range()
    slower = s.lower(68, srange) * srange ** 2
    supper = s.upper(68, srange) * srange ** 2

    fig, ax = plt.subplots()
    modelx = np.logspace(5, 6, 10)
    for ls, (key, v) in zip(["--", ":", "-."], data.items()):
        ax.plot(v["E"], v["flux"], label=key, ls=ls, zorder=2, c=model_colors[key])
        norm_arg = (1e6 - v["E"]).abs().idxmin()
        norm = v["flux"].loc[norm_arg] * 2
        logger.debug(f"Normalisation for {key}: {norm}")
        modely = norm * (modelx / 1e6) ** (model_gammas[key] + 2)
        ax.plot(modelx, modely, ls="-", c="grey", zorder=2, alpha=0.5)
    ax.errorbar(x_sens, y_sens, yerr=.2 * y_sens, zorder=1, uplims=True, c="grey")
    ax.fill_between(srange, slower, supper,
                    color="black", alpha=.2, label="Diffuse Flux", zorder=4, ec="none")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(bottom=1e-12)
    ax.set_xlim(1e3, 1e9)
    ax.set_xlabel("Energy [GeV]")
    ax.set_ylabel(r"$E^2 \Phi_\mu^{\nu + \bar{\nu}}$ [GeV cm$^{-2}$ s$^{-1}$ sr$^{-1}$]")

    ulim_handle = patches.FancyArrowPatch((0, 0), (0, -0.5), color="grey", arrowstyle="-|>",
                                          mutation_scale=10, shrinkA=-2, shrinkB=0)
    legend_handles = ax.get_legend_handles_labels()[0] + [ulim_handle]
    legend_labels = ax.get_legend_handles_labels()[1] + ["TDE Upper Limit"]
    _sort = [0, 3, 1, 4, 2]

    legend1 = ax.legend(legend_handles[3:], legend_labels[3:], loc="lower center", ncol=2, bbox_to_anchor=(0.5, 1.),
                        handler_map={patches.FancyArrowPatch: HandlerArrow()})
    legend2 = ax.legend(legend_handles[:3], legend_labels[:3], loc="lower center", ncol=3, bbox_to_anchor=(0.5, 1.1))

    ax.add_artist(legend1)
    return fig
