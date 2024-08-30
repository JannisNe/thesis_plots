import logging
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from thesis_plots.plotter import Plotter


logger = logging.getLogger(__name__)


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

    fig, ax = plt.subplots()
    for ls, (key, v) in zip(["-", "--", ":"], data.items()):
        ax.plot(v["E"], v["flux"], label=key, ls=ls, zorder=2)
    for iy, iexp in zip(y, exp):
        ax.plot(x, iy, ls="--", color="grey", zorder=0)
        ax.annotate(f"$\Phi \propto E^{ {iexp} }$", (x[-1], iy[-1]), color="grey", bbox=dict(facecolor="white", pad=-.5),
                    zorder=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(bottom=1e-12)
    ax.set_xlim(1e3, 1e9)
    ax.set_xlabel("Energy [GeV]")
    ax.set_ylabel(r"$\Phi\,E^2$ [GeV$^{-1}$ cm$^{-2}$ s$^{-1}$ sr$^{-1}$]")
    ax.legend(ncol=3, bbox_to_anchor=(0.5, 1.), loc="lower center")
    return fig
