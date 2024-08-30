import logging
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

from thesis_plots.plotter import Plotter


logger = logging.getLogger(__name__)


@Plotter.register()
def alert_number_constraint():
    storage_fn = Path(__file__).parent / "data" / "Ns_above_100.00tev.csv"
    df = pd.read_csv(storage_fn)
    gammas = df.gamma
    Ns = df.N

    fig, ax = plt.subplots()
    ax.errorbar(gammas, Ns, yerr=0.2, uplims=True, label='limits')
    ax.set_ylabel(r'$N_{\nu}(E>100\,\mathrm{TeV})$')
    ax.set_xlabel(r"$\gamma$")
    ax.axhline(3, label=r'$N_{\nu}$=3', c='gray', ls='--')
    ax.legend(loc="lower left")
    return fig


@Plotter.register(["margin", "notopright"])
def ts_distribution():
    background_filename = Path(__file__).parent / "data" / "0.pkl"
    with open(background_filename, "rb") as f:
        ts_background = pickle.load(f)["TS"]

    unblinding_filename = Path(__file__).parent / "data" / "unblinding_results.pkl"
    with open(unblinding_filename, "rb") as f:
        u = pickle.load(f)

    fig, ax = plt.subplots()
    ax.hist(ts_background, bins=10, density=True, label='background \ndistribution', alpha=1)

    ls = '-'
    ax.axvline(u["TS"], label="$\lambda_\mathrm{observed}$", c=f"C1", ls=ls)
    med_ts = np.median(ts_background)
    ax.axvline(med_ts, label="$\lambda_\mathrm{median}$", color='k', alpha=1, ls='--')
    ax.set_yscale('log')
    ax.legend(bbox_to_anchor=(0.5, 1.1), loc="lower center", borderaxespad=0.0, ncol=1)
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel("density")
    return fig
