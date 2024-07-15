import logging
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import numpy as np

from thesis_plots.plotter import Plotter


logger = logging.getLogger(__name__)


@Plotter.register(arg_loop=["tywin", "txs"])
def performance(event_name: str):
    data_dir = Path(__file__).parent / "data"
    old_time = pd.read_csv(data_dir / f"old_scheme_time_{event_name}.csv", index_col=0)
    logger.debug(old_time)
    new_time = pd.read_csv(data_dir / f"new_scheme_time_{event_name}.csv", index_col=0)
    logger.debug(new_time)
    xlabels = [
        "Generate", "Propagate", "Calculate Energy", "Select Energy",
        "Photons", "Detector", "L1", "L2", "Charge selection", "total"
    ]
    x = np.arange(len(xlabels))
    y = [[t.loc[ix, "time_per_sim"] if ix in t.index else np.nan for ix in xlabels] for t in [old_time, new_time]]
    ydiff = [y[1][i] - y[0][i] for i in range(len(x))]
    logger.debug(y)
    yerr = [[t.loc[ix, "u_time_per_sim"] if ix in t.index else np.nan for ix in xlabels] for t in [old_time, new_time]]
    ydiff_err = [np.sqrt(yerr[0][i] ** 2 + yerr[1][i] ** 2) for i in range(len(x))]
    logger.debug(yerr)

    fig, (ax, diff_ax) = plt.subplots(
        nrows=2, gridspec_kw={"hspace": 0, "height_ratios": [1, 3]}, sharey="row", sharex="col"
    )
    for i, (t, m) in enumerate(zip(["Old scheme", "New scheme"], ["o", "s"])):
        offset = 0.05 if i == 0 else -0.05
        ax.errorbar(x + offset, y[i], yerr=yerr[i], label=t, ls="", marker=m, markersize=2, lw=1)

    diff_ax.errorbar(x, ydiff, yerr=ydiff_err, ls="", marker="s", color="C1", lw=1, markersize=2)
    diff_ax.axhline(0, color="k", ls="--")
    ax.set_ylabel("Time [s]")
    ax.legend(bbox_to_anchor=(0.5, 1), loc="lower center", ncol=2)
    ax.set_yscale("log")
    diff_ax.set_ylabel(r"$\Delta$ Time [s]")
    diff_ax.set_xticks(x, xlabels[:-1] + ["Total"], rotation=45, ha='right')
    for ax in [ax, diff_ax]:
        ax.axvline(8.5, color="k", ls=":")
    diff_ax.set_yscale("symlog", linthresh=10)

    return fig
