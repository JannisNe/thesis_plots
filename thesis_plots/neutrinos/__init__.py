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

    fig, ax = plt.subplots()
    for n in data.columns.get_level_values(0).unique():
        logger.debug(f"plotting {n}")
        d = data[n].sort_values("X").apply(lambda x: gaussian_filter(x, sigma=1))
        ax.plot(d["X"], d["Y"])

    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel("Neutrino Energy [eV]")
    ax.set_ylabel("Flux [MeV$^{-1}$ cm$^{-2}$ s$^{-1}$ sr$^{-1}$]")

    return fig
