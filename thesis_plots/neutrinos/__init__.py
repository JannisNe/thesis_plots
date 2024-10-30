import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import gaussian_filter

from thesis_plots.plotter import Plotter


logger = logging.getLogger(__name__)


@Plotter.register("wide")
def spectrum():
    data_filename = Path(__file__).parent / "wpd_datasets.csv"
    headers = pd.read_csv(data_filename, nrows=2, header=None)
    headers_zip = [(headers.loc[0, np.floor(i / 2)*2], headers.loc[1, i]) for i in range(len(headers.columns))]
    index = pd.MultiIndex.from_tuples(headers_zip)
    data = pd.read_csv(data_filename, skiprows=2, header=None, names=index)

    annotations = {
        "cosmo": ("cosmological", (1e-6, 1e-8), None),
        "solar": ("solar", (10, 1e3), None),
        "sn1987a": ("SN 1987A", (1e7, 1e11), None),
        "reactor": ("reactor", (1e7, 3e2), (3.6e10, 8.5e4)),
        "dsnb": ("DSNB", (5e7, 5e-2), None),
        "atmospheric": ("atmopsheric", (9e10, 1e-12), None),
        "agn": ("astrophysical", (1e5, 1.5e-18), None),
        "cosmogen": ("cosmogenic", (1e15, 1e-25), None),
        "earyh anti nu": ("terrestrial", (9.6e5, 2.5e6), (1, 2.5e-9))
    }

    fig, ax = plt.subplots()
    for i, nn in enumerate(data.columns.get_level_values(0).unique()):
        logger.debug(f"plotting {nn}")
        d = data[nn].astype(float).sort_values("X").apply(gaussian_filter, sigma=.6)
        ax.plot(d["X"] * 1e-9, d["Y"], label=nn, lw=4, c=f"C{i}")
        t, xy, xytext = annotations[nn]
        a = {"arrowstyle": "-|>", "mutation_scale": 10, "color": f"C{i}"} if xytext else None
        _xy = (xy[0] / 1e9, xy[1])
        _xytext = (xytext[0] / 1e9, xytext[1]) if xytext else None
        ax.annotate(t, _xy, xytext=_xytext, textcoords="data", color=f"C{i}", ha="left", va="bottom", arrowprops=a)

    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel("Neutrino Energy [GeV]")
    ax.set_ylabel("Flux [MeV$^{-1}$ cm$^{-2}$ s$^{-1}$ sr$^{-1}$]")

    return fig
