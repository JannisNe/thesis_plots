import logging
from thesis_plots.plotter import Plotter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


logger = logging.getLogger(__name__)


@Plotter.register()
def sn2022hsu():
    # Load the data
    data_fn = Path(__file__).parent / "data" / "sn2022hsu.csv"
    logger.debug(f"Loading data from {data_fn}")
    data = pd.read_csv(data_fn, index_col=0)
    ulm = data.mag.isna()
    logger.debug(f"Found {ulm.sum()} upper limits")
    mjd_ref = data[~ulm]["mjd"].min()
    data["time"] = data["mjd"] - mjd_ref
    logger.debug(f"earliest upper limit: {data[ulm].time.min()}")

    # Plot the data
    fig, ax = plt.subplots()
    for fi, (f, c) in enumerate(zip(["g", "r"], ["C0", "C1"])):
        bandm = data["filter"] == f"ztf{f}"
        ul_data = data[bandm & ulm & (data.time < 40)]
        ax.scatter(ul_data.time, ul_data.limiting_mag, marker="v", color=c, alpha=0.5, edgecolors="none")
        meas_data = data[bandm & ~ulm]
        ax.errorbar(meas_data.time, meas_data.mag, yerr=meas_data.magerr, fmt="o", color=c, label=f"band {fi}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Brightness")
    ylim = ax.get_ylim()
    ax.set_ylim(ylim[1], ylim[0])
    ax.set_xlim(left=data.time.min()-2, right=data.time.max()+10)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.legend(bbox_to_anchor=(.5, 1.2), loc="upper center", ncol=2)

    return fig
