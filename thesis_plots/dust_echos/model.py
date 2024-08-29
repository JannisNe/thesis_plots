import logging
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from thesis_plots.plotter import Plotter


logger = logging.getLogger(__name__)


@Plotter.register()
def winter_lunardini():
    fn = Path(__file__).parent / "data" / "winter_lunardini_dust_echos_diffuse_flux.csv"
    data = pd.read_csv(fn, decimal=",", delimiter=";", names=["E", "flux"])
    exp = np.array([-1])
    x = np.logspace(np.log10(data["E"].iloc[0]), 7, 100)
    y = data["flux"].iloc[0] * (x / data["E"].iloc[0]) ** (exp[:, np.newaxis] + 2) * 1.5

    fig, ax = plt.subplots()
    ax.plot(data["E"], data["flux"], label="Winter & Lunardini (2023)")
    for iy, iexp in zip(y, exp):
        ax.plot(x, iy, ls="--", label=f"$\Phi \propto E^{ {iexp} }$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Energy [GeV]")
    ax.set_ylabel(r"$\Phi\,E^2$ [GeV$^{-1}$ cm$^{-2}$ s$^{-1}$ sr$^{-1}$]")
    ax.legend()
    return fig
