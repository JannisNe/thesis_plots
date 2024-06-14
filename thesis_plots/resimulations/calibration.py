import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle

from thesis_plots.plotter import Plotter


logger = logging.getLogger(__name__)


def get_data(event_name: str):
    fn = Path(__file__).parent / "data" / f"{event_name}_calib_data.pkl"
    logger.debug(f"loading {fn}")
    with open(fn, "rb") as f:
        data = pickle.load(f)
    return data


@Plotter.register(["margin", "notopright"], arg_loop=["bran", "txs", "tywin"])
def metric_histogram(event_name):
    data = get_data(event_name)
    Emeas = data["Emeas"]
    Esim = data["Esim"]
    Esim_trunc = np.array([E[:len(Emeas)] for E in Esim])
    Eratio = np.array([E / Emeas for E in Esim_trunc])
    max_e_ratio = np.max(abs(np.log10(Eratio)), axis=1)

    fig, ax = plt.subplots()
    ax.hist(max_e_ratio, density=True, cumulative=True, zorder=1)
    ax.axvline(0.8, ls="--", zorder=2, color="C1")
    ax.set_xlabel(r"Maximum of |$\log_{10}(E_\mathrm{ratio})$|")
    ax.set_ylabel("cumulative density")
    ax.set_xlim(left=0)

    return fig
