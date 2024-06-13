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
    Esim_trunc = [e[:len(ee)] for e, ee in zip(Esim, Emeas)]
    maxabslog = np.array([np.max(abs(np.log10(iEmeas / iEsim))) for iEmeas, iEsim in zip(Emeas, Esim_trunc)])
    m = maxabslog < np.inf

    fig, ax = plt.subplots()
    ax.hist(maxabslog[m], density=True, cumulative=True)
    ax.set_xlabel("Maximum of |$\log_{10}(E_\mathrm{ratio})$|")
    ax.set_ylabel("cumulative density")
    ax.set_xlim(left=0)


    return fig
