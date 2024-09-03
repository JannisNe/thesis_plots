import logging
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
import pandas as pd
import pickle

from thesis_plots.plotter import Plotter
from thesis_plots.dust_echos.model import model_colors
from thesis_plots.arrow_handler import HandlerArrow


logger = logging.getLogger(__name__)


@Plotter.register()
def alert_number_constraint():
    storage_fn = Path(__file__).parent / "data" / "Ns_above_100.00tev.csv"
    df = pd.read_csv(storage_fn)
    gammas = df.gamma
    Ns = df.N

    fig, ax = plt.subplots()
    ax.errorbar(gammas, Ns, yerr=0.2, uplims=True)
    ax.set_ylabel(r'$N_{\nu}(E>100\,\mathrm{TeV})$')
    ax.set_xlabel(r"$\gamma$")
    ax.axhline(3, c='gray', ls='-', alpha=0.5)
    ax.annotate(r'$N_{\nu}$=3', (max(ax.get_xlim()), 3), xytext=(-2, 2), textcoords="offset points", ha="right",
                va="bottom", color='gray')
    ax.axvline(2, c=model_colors["X-ray"], ls=":", label="X-ray / OUV")
    ax.axvline(1, c=model_colors["IR"], ls="--", label="IR")
    limit_handle = patches.FancyArrowPatch((1.5, 3), (1.5, 3.5), arrowstyle="-|>", mutation_scale=10, color="C0")
    handles = ax.get_legend_handles_labels()[0] + [limit_handle]
    labels = ax.get_legend_handles_labels()[1] + [r"limits"]
    ax.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 1.01),
              handler_map={patches.FancyArrowPatch: HandlerArrow()})
    return fig


@Plotter.register(["margin", "notopright"])
def ts_distribution():
    background_filename = Path(__file__).parent / "data" / "0.pkl"
    with open(background_filename, "rb") as f:
        ts_background = pickle.load(f)["TS"]
    med_ts = np.median(ts_background)

    unblinding_filename = Path(__file__).parent / "data" / "unblinding_results.pkl"
    with open(unblinding_filename, "rb") as f:
        u = pickle.load(f)

    logger.info(f"background median TS: {med_ts}")
    logger.info(f"observed TS: {u['TS']}")

    fig, ax = plt.subplots()
    ax.hist(ts_background, bins=10, density=True, label='background \ndistribution', alpha=1)

    ls = '-'
    ax.axvline(u["TS"], label="$\lambda_\mathrm{obs}$", c=f"C1", ls=ls)
    ax.axvline(med_ts, label="$\lambda_\mathrm{bkg}$", color='k', alpha=1, ls='--')
    ax.set_yscale('log')
    ax.legend(bbox_to_anchor=(0.5, 1.1), loc="lower center", borderaxespad=0.0, ncol=1)
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel("density")
    return fig
