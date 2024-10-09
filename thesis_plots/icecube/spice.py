import logging
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

from thesis_plots.plotter import Plotter


logger = logging.getLogger(__name__)


@Plotter.register("margin", orientation="portrait")
def spice321():
    filename = Path(__file__).parent / "data" / "icemodel.dat"
    logger.debug(f"loading data from {filename}")
    data = pd.read_csv(filename, delim_whitespace=True, header=None, names=["depth", "scat", "abs", "aniso"])

    fig, ax = plt.subplots()
    ax.plot(data["scat"], data["depth"], label="Scattering", ls="-")
    ax.plot(data["abs"], data["depth"], label="Absorption", ls="--")
    ax.fill_between([0, 0.18], [2000, 2000], [2100, 2100], color="grey", alpha=0.3, label="dust layer", ec="none")
    ax.set_ylabel("Depth [m]")
    ax.set_xlabel("Coefficient [1/m]")
    ax.set_xlim(0, 0.18)
    ax.set_ylim(2450, 1450)
    ax.legend()
    return fig
