import logging
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from matplotlib import ticker
import pandas as pd
from scipy import stats
import numpy as np
import math

from thesis_plots.flaires.data.load import load_data
from thesis_plots.plotter import Plotter


from timewise.utils import plot_panstarrs_cutout


logger = logging.getLogger(__name__)


@Plotter.register("margin")
def redshift_bias():
    data = load_data()
    luminosity_summary = data["luminosity_summary"]
    redshifts = data["redshifts"][["z"]]
    info = pd.concat([luminosity_summary, redshifts.set_index(redshifts.index.astype(str))], axis=1)
    zbins = np.array([0, 0.05, 0.1, 0.2, 0.3, 0.4])
    grouped = info.groupby(pd.cut(info["z"], zbins))
    zbin_mids = (zbins[1:] + zbins[:-1]) / 2

    fig, ax = plt.subplots()
    for v, label, c, exp in zip(
            ["peak_temperature", "radius_at_peak"],
            ["$(T/T_0)^4$", "$(R/R_0)^2$"],
            ["C0", "C1"],
            [4, 2]
    ):
        q = grouped[v].quantile([0.16, 0.5, 0.84])
        q /= q.loc[0.05, 0.5]
        ax.plot(zbin_mids, q.loc[:, 0.5].values ** exp, marker="", ls="-", color=c, label=label)
        ax.fill_between(zbin_mids, q.loc[:, 0.16].values ** exp, q.loc[:, 0.84].values ** exp, color=c, alpha=.5, ec="none")

    ax.legend(bbox_to_anchor=(.5, 1), loc="lower center", ncol=2)
    ax.set_yscale("log")
    ax.set_yticks([1, 2, 5, 10])
    ax.set_yticklabels(["1", "2", "5", "10"])
    ax.set_xlabel("redshift")
    ax.set_ylabel("relative evolution")


    return fig


@Plotter.register("fullpage")
def offset_cutouts():
    data = load_data()
    offset_objects = data["offset_objects"]
    separations = data["separations"]
    aux_data = data["loaded_datas"]

    # --------- make offset mosaik ----------- #

    fig, axs = plt.subplots(
        ncols=3,
        nrows=4,
        gridspec_kw={"wspace": 0.1, "hspace": 0.1},
        sharex="all", sharey="all",
    )
    axs = axs.flatten()
    ticks = [-10, -5, 0, 5, 10]
    for i, (j, r) in enumerate(offset_objects.iterrows()):
        logger.debug(f"plotting {j}: {r['name']}")

        loaded_data, pos, flare_mask = aux_data[j]
        ra = pos["RAdeg"]
        dec = pos["DEdeg"]
        sep = separations.loc[j]

        ax = axs[i]
        plot_panstarrs_cutout(ra, dec, arcsec=20, interactive=True, plot_color_image=True, ax=ax)
        ax.scatter(loaded_data[~flare_mask]["rel_ra"], loaded_data[~flare_mask]["rel_dec"],
                   s=3, ec="k", marker="o", fc="w", lw=0.5, label="baseline", zorder=5)
        ax.scatter(sep["rel_baseline_ra"], sep["rel_baseline_dec"],
                   s=50, ec="k", marker="X", fc="w", lw=1, zorder=10)
        ax.scatter(loaded_data[flare_mask]["rel_ra"], loaded_data[flare_mask]["rel_dec"],
                   s=3, ec="r", marker="o", fc="w", lw=0.5, label="flare", zorder=5)
        ax.scatter(sep["rel_flare_ra"], sep["rel_flare_dec"],
                   s=50, ec="r", marker="X", fc="w", lw=1, zorder=10)
        ax.set_aspect(1, adjustable="box")
        ax.scatter([], [], marker="x", color="r", label="parent sample position")

        ax.set_title(r.title, pad=-14, y=1, color="white")
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.grid(False)
        ax.set_xlabel("")
        ax.set_ylabel("")

    bsl_dot = axs[0].scatter([], [], s=3, ec="k", marker="o", fc="w", lw=0.5)
    bsl_cross = axs[0].scatter([], [], s=50, ec="k", marker="X", fc="w", lw=1)
    data_dot = axs[0].scatter([], [], s=3, ec="r", marker="o", fc="w", lw=0.5)
    data_cross = axs[0].scatter([], [], s=50, ec="r", marker="X", fc="w", lw=1)
    ps_pos = axs[0].scatter([], [], marker="x", color="red")
    fig.legend(
        [ps_pos, (bsl_dot, bsl_cross), (data_dot, data_cross)],
        ["parent sample", "baseline", "flare"],
        handler_map={tuple: HandlerTuple(ndivide=None)},
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.92),
    )

    fig.supylabel(r"$\Delta$Dec [arcsec]", x=0.03, ha="left")
    fig.supxlabel(r"$\Delta$RA [arcsec]", y=0.03, ha="center")

    return fig


@Plotter.register("fullpage", arg_loop=[("W1", 0), ("W1", 1), ("W2", 0), ("W2", 1)])
def chi2(args: tuple[str, int]):
    band, page = args
    page = int(page)
    hists = load_data()["chi2"]
    nominal_rows = 4
    col = 3
    start = 2 + page * col * nominal_rows
    end = 2 + (page + 1) * col * nominal_rows
    ns_in_hist = [n for n in range(start, end) if n in hists]
    rows = math.ceil(len(ns_in_hist) / col)
    w, h = plt.rcParams["figure.figsize"]
    h *= rows / nominal_rows
    logger.debug(f"plotting chi2 histograms for {ns_in_hist}")
    x_dense = np.linspace(0, 4, 1000)

    gridspec = {"wspace": 0., "hspace": 0}
    fig, axs = plt.subplots(ncols=col, nrows=rows, sharex="all", sharey="all", gridspec_kw=gridspec, figsize=(w, h))
    axs_flat = axs.flatten()
    merge_bins = 3

    for j, (i, ax) in enumerate(zip(ns_in_hist, axs_flat)):
        h, d, bins, p = hists[i][band]
        new_bins = bins[::merge_bins]
        logger.debug(f"reduced bins from {len(bins)} to {len(new_bins)}")
        new_d = np.add.reduceat(d, np.arange(0, len(d), merge_bins)) / merge_bins
        logger.debug(f"reduced data from {len(d)} to {len(new_d)}")
        phist = ax.bar(new_bins[:-1], new_d, width=np.diff(new_bins), align="edge", color="C0", alpha=0.5)
        ax.bar(new_bins[:-1], new_d, width=np.diff(new_bins), align="edge", color="none", ec="w")
        pchi2 = ax.plot(x_dense, stats.chi2(i - 1, 0, 1 / (i - 1)).pdf(x_dense), ls="--", color="C1", lw=2)
        pf = ax.plot(x_dense, stats.f(i - 1, 1, 0).pdf(x_dense), ls="-", color="C2", lw=2)
        ax.set_ylim(0, 1.4)
        ax.set_title(f"N={i}", pad=-14, y=1, color="black")

    fig.supylabel("density")
    fig.supxlabel(r"$\chi^2 / (N - 1)$")
    axs[0, 1].legend(
        [phist, pchi2[0], pf[0]],
        ["data", r"$\chi^2$-distribution", r"$F$-distribution"],
        loc="lower center", ncol=3, bbox_to_anchor=(0.5, 1.05)
    )

    return fig


@Plotter.register("wide")
def coverage():
    data = load_data()["coverage"]
    fig, axs = plt.subplots(ncols=2, sharex="all", sharey="all", gridspec_kw={"wspace": 0.})
    merge_bins = 3
    for i, (b, (h, bins, l)) in enumerate(data.items()):
        logger.info(f"plotting coverage for {b}: {l}")
        new_bins = bins[::merge_bins]
        logger.debug(f"reduced bins from {len(bins)} to {len(new_bins)}")
        new_h = np.add.reduceat(h, np.arange(0, len(h), merge_bins))
        logger.debug(f"reduced data from {len(h)} to {len(new_h)}")
        axs[i].bar(new_bins[:-1], new_h, width=np.diff(new_bins), align="edge")
        med = float(l.strip("$").split(" ^")[0])
        axs[i].axvline(med, color="k", ls="--", label=f"median: {l}")
        axs[i].set_xlabel(f"W{i+1}")
        axs[i].legend()
    axs[0].set_ylabel("number of objects")
    fig.supxlabel("coverage", ha="center", y=-0.05)
    return fig
