import logging
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from astropy.time import Time
from thesis_plots.flaires.data.load import load_data
from thesis_plots.plotter import Plotter
from timewise_sup.meta_analysis.luminosity import luminosity_key
from air_flares.plots.paper_plots import indicate_news_cutoff
from air_flares.export.rates import control_region_mjd, get_wise_times


logger = logging.getLogger(__name__)


@Plotter.register()
def luminosity_function():

    data = load_data()
    luminosity_summary = data["luminosity_summary"]
    type_masks = data["type_masks_lum_fct"]
    redshifts = data["redshifts"]
    redshift_bins = data["redshift_bins"]
    n_galaxies = data["n_galaxies"]
    n_lum_bins = 20
    luminosity_key = "peak_luminosity"

    redshifts = redshifts.loc[luminosity_summary.index.astype(int), ["z"]]
    redshifts.set_index(redshifts.index.astype(str), inplace=True)
    bin_indices = np.digitize(redshifts.loc[luminosity_summary.index, "z"], redshift_bins)
    n_z_bins = len(redshift_bins) - 1

    good_mask = (luminosity_summary[luminosity_key] > 0) & ~type_masks["Quasar"]
    logger.debug(f"removed {sum(type_masks['Quasar'])} quasars")
    lum_bins = np.logspace(
        np.log10(luminosity_summary[luminosity_key][good_mask].min()),
        np.log10(luminosity_summary[luminosity_key][good_mask].max()),
        n_lum_bins,
    )
    logger.debug(f"luminosity bins: {lum_bins}")
    lum_bin_width = np.diff(lum_bins)

    fig, axes = plt.subplots(
        n_z_bins, 1,
        sharex=True, sharey=True,
        gridspec_kw={"hspace": 0},
    )
    axes = np.atleast_1d(axes)
    for i, ax in enumerate(axes):
        bin_mask = bin_indices == (i + 1)
        logger.debug(f"found {np.sum(bin_mask)} sources in redshift bin {i}")
        if n_z_bins > 1:
            ax.annotate(
                f"{redshift_bins[i]:.2f} < z < {redshift_bins[i+1]:.2f}",
                (0, 1), xycoords="axes fraction", ha="left", va="top", fontsize="small",
                xytext=(2, -2), textcoords="offset points"
            )

        vals = luminosity_summary[luminosity_key][bin_mask].astype(float)
        nan_mask = np.isnan(vals)
        logger.debug(f"found {np.sum(nan_mask)} NaN values in redshift bin {i}")
        if all(vals.isna()):
            logger.debug("all values are NaN, skipping")
            continue

        logger.debug(f"found {n_galaxies[i]} galaxies in redshift bin {i}")

        h, b = np.histogram(vals[~nan_mask], bins=lum_bins)
        ax.bar(
            b[:-1],
            h / n_galaxies[i],
            width=lum_bin_width,
            align="edge",
            color="C0",
            ec="k",
            alpha=0.6,
        )
        ax.set_xscale("log")
        ax.set_yscale("log")

    fig.supylabel("Rate [galaxy$^{-1}$]")
    xlabel = r"L$_\mathrm{bol,\,peak}$ [erg s$^{-1}$]"
    axes[-1].set_xlabel(xlabel)

    return fig


@Plotter.register(["margin", "notopright"])
def redshifts():
    data = load_data()
    redshifts = data["redshifts"]
    types_mask = data["types_mask_zhist"]
    good_mask = (redshifts["z"] > 0) & ~types_mask["Quasar"]
    z = redshifts["z"][good_mask]

    fig, ax = plt.subplots()
    bins = np.linspace(z.min(), z.max(), 20)
    ax.hist(z.values, bins=bins, alpha=0.5, ec="k")
    ax.set_xlabel("redshift")
    ax.set_ylabel("number of objects")
    return fig


@Plotter.register(["notopright"])
def subsamples():
    info = load_data()["subsamples"]
    xbins = info["xbins"]
    xkey = info["xkey"]
    xlabel = info["xlabel"]
    news_cutoff = info["news_cutoff"]
    hists = info["hists"]

    fig, ax = plt.subplots()
    ls = ["--", "-.", ":"]
    for (subsample, (h, b)), ils in zip(hists.items(), ls):
        ax.step(b[:-1], h, alpha=0.5, label=subsample, ls=ils, where="post")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("number of objects")
    ax.set_yscale("log")
    ax.set_xlim(17.5, 7.5)
    indicate_news_cutoff(ax, annotate="bottom", cutoff=news_cutoff)
    ax.legend()
    return fig


@Plotter.register()
def peak_times():
    data = load_data()
    luminosity_summary = data["luminosity_summary"]
    types_mask = data["type_masks_lum_fct"]
    good_mask = ~types_mask["Quasar"]
    x = Time(luminosity_summary["ref_time_mjd"][good_mask], format="mjd").to_datetime()
    control_region_time = Time(control_region_mjd, format="mjd").to_datetime()
    control_region_mid_time = Time(sum(control_region_mjd) / 2, format="mjd").to_datetime()
    _, pause_start, pause_end, _ = get_wise_times()
    bin_width = 365
    n_bins_before_pause = int(np.ceil((pause_start - x.min()).days / bin_width))
    n_bins_after_pause = int(np.ceil((x.max() - pause_end).days / bin_width))
    bins = (
            list(pd.date_range(pause_start, x.min(), n_bins_before_pause + 1).values[::-1]) +
            list(pd.date_range(pause_end, x.max(), n_bins_after_pause + 1).values)
    )
    pause_mid_date = pause_start + (pause_end - pause_start) / 2

    fig, ax = plt.subplots()
    h, b, p = ax.hist(x, bins=bins, alpha=0.5, color="C0")
    ax.set_xlabel("Peak Date")
    ax.set_ylabel("number of objects")
    ylim = np.array(ax.get_ylim())
    ax.fill_betweenx(ylim * 1.2, x1=[control_region_time[0]], x2=[control_region_time[1]],
                     color="grey", alpha=0.3, ec="none")
    ax.annotate("reference", (control_region_mid_time, 0), xytext=(0, 2),
                textcoords="offset points", ha="center", va="bottom", fontsize="small", color="grey")
    ax.axvline(pause_start, ls="--", color="grey", alpha=0.5)
    ax.axvline(pause_end, ls="--", color="grey", alpha=0.5)
    ax.annotate("no data", (pause_mid_date, 0), xytext=(0, 2),
                textcoords="offset points", ha="center", va="bottom", fontsize="small", color="grey")
    ax.set_ylim(ylim)
    return fig
