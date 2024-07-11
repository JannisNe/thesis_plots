import logging
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import numpy as np

from thesis_plots.plotter import Plotter


logger = logging.getLogger(__name__)


@Plotter.register()
def performance():
    data_dir = Path(__file__).parent / "data"
    old_time = pd.read_csv(data_dir / "old_scheme_time.csv", index_col=0)
    logger.debug(old_time)
    new_time = pd.read_csv(data_dir / "new_scheme_time2.csv", index_col=0)
    logger.debug(new_time)
    x = ["Generate", "Propagate", "Calculate Energy", "Select Energy", "Photons", "Detector", "L1", "L2", "Charge selection", "total"]
    y = [[t.loc[ix, "time_per_sim"] if ix in t.index else np.nan for ix in x] for t in [old_time, new_time]]
    yratio = [y[1][i] / y[0][i] for i in range(len(x))]
    logger.debug(y)
    yerr = [[t.loc[ix, "u_time_per_sim"] if ix in t.index else np.nan for ix in x] for t in [old_time, new_time]]
    yratio_err = [yratio[i] * np.sqrt((yerr[0][i] / y[0][i])**2 + (yerr[1][i] / y[1][i])**2) for i in range(len(x))]
    logger.debug(yerr)

    fig, ((ax, total_ax), (ratio_ax, total_ratio_ax)) = plt.subplots(
        ncols=2, nrows=2,
        gridspec_kw={"width_ratios": [9, 1], "wspace": 0, "hspace": 0, "height_ratios": [4, 1]},
        sharey="row", sharex="col"
    )
    for i, (t, m) in enumerate(zip(["Old scheme", "New scheme"], ["o", "s"])):
        ax.errorbar(x[:-1], y[i][:-1], yerr=yerr[i][:-1], label=t, ls="", capsize=5, capthick=2, marker=m)
        total_ax.errorbar("Total", y[i][-1], yerr=yerr[i][-1], ls="", capsize=5, capthick=2, marker=m)

    ratio_ax.errorbar(x[:-1], yratio[:-1], yerr=yratio_err[:-1], ls="", capsize=5, capthick=2, marker="o", color="k")
    total_ratio_ax.errorbar("Total", yratio[-1], yerr=yratio_err[-1], ls="", capsize=5, capthick=2, marker="o", color="k")
    for iax in [ratio_ax, total_ratio_ax]:
        iax.axhline(1, color="k", ls="--")
    ax.set_ylabel("Time [s]")
    ax.legend(bbox_to_anchor=(0.5, 1), loc="lower center", ncol=2)
    ax.set_yscale("log")
    total_ax.set_yscale("log")
    ratio_ax.set_ylabel("Ratio")
    ratio_ax.set_yscale("log")
    ratio_ax.set_xticklabels(x, rotation=45, ha="right")
    total_ratio_ax.set_xticklabels(["Total"], rotation=45, ha="right")

    return fig
