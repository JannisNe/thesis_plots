import logging
import pickle
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from matplotlib import colormaps
from matplotlib.legend_handler import HandlerTuple
import pandas as pd
import numpy as np
from pathlib import Path
from astropy import constants
from astropy import units as u
from astropy.cosmology import Planck18
from astropy.time import Time
from itertools import chain
import healpy as hp

from thesis_plots.flaires.data.load import load_data
from thesis_plots.plotter import Plotter

from timewise_sup.environment import load_environment
from timewise_sup.mongo import DatabaseConnector
from timewise_sup.meta_analysis.flux import get_band_nu
from timewise_sup.meta_analysis.diagnostics import get_baseline_changes, get_baseline_magnitudes
from timewise_sup.meta_analysis.luminosity import (
    get_ir_luminosities_index,
    luminosity_key,
    luminosity_err_key,
    ref_time_key
)
from timewise_sup.plots.diagnostic_plots import plot_separation

from air_flares.plots.temperature_plots import (
    make_temperature_lightcurve_plot,
    make_temperature_radius_plot,
    make_temperature_fit_plot
)


logger = logging.getLogger(__name__)


@Plotter.register("margin")
def make_redshift_bias_plots():
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
    ax.legend()
    ax.set_xlabel("redshift")
    ax.set_ylabel("relative evolution")

    return fig


@Plotter.register("wide")
def make_offset_cutouts():
    logger.info("making offset cutouts")
    indices = dict(
        sn=[21145408, 28794291, 41859807],
        candidate=[34024570, 26227543, 12928731],
        unclear=[38769188, 30921575, 30141662],  # nearby galaxy
        weird=[39158789, 33690240],  # weird offset already in baseline data
        example=[39079503]
    )

    wd = NEWSPlusPS1xWISEStrmPlusLVSWISEData()
    table = get_table()

    offset_objects = pd.DataFrame(index=list(chain.from_iterable(indices.values())), columns=[])
    for k, v in indices.items():
        offset_objects.loc[v, "type"] = k
    offset_objects["chunks"] = pd.Series(
        [wd._get_chunk_number(parent_sample_index=i) for i in offset_objects.index],
        index=offset_objects.index
    )
    offset_objects["positional"] = [
        wd.query_types_subsamples[wd.subsample_chunk_map[c]] == "positional" for c in offset_objects.chunks
    ]
    offset_objects = pd.concat([
        offset_objects, table.set_index("parent sample index").loc[offset_objects.index, ["name", "TNS_name"]]
    ], axis=1)
    offset_objects["title"] = offset_objects["name"].str.replace(r"\.\d*", "", regex=True)
    offset_objects.loc[~pd.isna(offset_objects.TNS_name), "title"] = offset_objects.TNS_name

    logger.debug(f"offset_objects:\n {offset_objects}")

    # --------- make offset mosaik ----------- #

    logger.info("making supernova offset mosaik")
    fig_width = 2 * plt.rcParams["figure.figsize"][0]
    fig_height = fig_width * 4/3
    figsize = (fig_width, fig_height)
    fig, axs = plt.subplots(
        ncols=3,
        nrows=4,
        gridspec_kw={"wspace": 0.1, "hspace": 0.1},
        sharex="all", sharey="all",
        figsize=figsize,
    )
    axs = axs.flatten()
    ticks = [-10, -5, 0, 5, 10]
    for i, (j, r) in enumerate(offset_objects.iterrows()):
        logger.debug(f"plotting {j}: {r['name']}")
        plot_separation(
            base_name=wd.base_name,
            database_name=database_name,
            wise_data=wd,
            index=j,
            mask_by_position=r.positional,
            ax=axs[i],
            save=False,
            arcsec=20
        )
        axs[i].set_title(r.title, pad=-14, y=1, color="white")
        axs[i].get_legend().remove()
        axs[i].set_xticks(ticks)
        axs[i].set_yticks(ticks)
        # axs[i].set_aspect(1, adjustable="box")
        # axs[i].set_xlim(-10, 10)
        # axs[i].set_ylim(-10, 10)

    for ax in axs:
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

    fig.supylabel(r"$\Delta$Dec [arcsec]", x=0.05, ha="left")
    fig.supxlabel(r"$\Delta$RA [arcsec]", y=0.05, ha="left")

    fn = get_plots_dir() / "offset_cutouts.pdf"
    logger.info(f"saving under {fn}")
    fig.tight_layout()
    fig.savefig(fn, bbox_inches="tight")
    plt.close()

