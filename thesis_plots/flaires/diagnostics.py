import logging
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
import pandas as pd
import numpy as np

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
    for v, label, c in zip(
            ["peak_temperature", "radius_at_peak"],
            ["T [K]", "R [pc]"],
            ["C0", "C1"],
    ):
        q = grouped[v].quantile([0.16, 0.5, 0.84])
        q /= q.loc[0.05, 0.5]
        ax.plot(zbin_mids, q.loc[:, 0.5].values, marker="", ls="-", color=c, label=label)
        ax.plot(zbin_mids, q.loc[:, 0.16].values, marker="", ls="--", color=c)
        ax.plot(zbin_mids, q.loc[:, 0.84].values, marker="", ls="--", color=c)

    ax.axhline(1, ls=":", color="grey")
    ax.legend(bbox_to_anchor=(.5, 1), loc="lower center", ncol=2)
    ax.set_xlabel("redshift")
    ax.set_ylabel("relative evolution")

    return fig


@Plotter.register("fullpage")
def make_offset_cutouts():
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
