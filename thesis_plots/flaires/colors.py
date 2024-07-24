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
from air_flares.plots.paper_plots import indicate_news_cutoff


logger = logging.getLogger(__name__)


@Plotter.register()
def histogram():
    info = load_data()["colors"]
    agn_in_dust_echoes = info["agn_in_dust_echoes"]
    agn_in_all_sources = info["agn_in_all_sources"]
    bins = info["b"]
    bm = (bins[:-1] + bins[1:]) / 2
    width = np.diff(bins)
    agn_color = 0.8

    logger.info(f"fraction of AGN in dust echoes: {agn_in_dust_echoes:.2f}")
    logger.info(f"fraction of AGN in all sources: {agn_in_all_sources:.2f}")

    fig, ax = plt.subplots()
    ax.bar(bins, info["h1"], width=width, color="C0", alpha=0.5, align="edge", label="dust echoes")
    # make a histogram of all sources weighted with N_dust_echoes / N_all
    ax.step(bm, info["h2"], width=width, color="C1", alpha=0.5, align="edge", label="all (scaled)")
    ax.axvline(agn_color, ls="--", color="grey", label="Stern+12")
    ax.set_yscale("log")
    ylim = list(ax.get_ylim())
    ylim[1] = ylim[1] * 1.5
    ax.annotate(
        "AGN", (agn_color, ylim[1]), xycoords="data", xytext=(2, -2), textcoords="offset points", ha="left", va="top",
        fontsize="small", color="grey"
    )
    ax.annotate(
        "non-AGN", (agn_color, ylim[1]), xycoords="data", xytext=(-2, -2), textcoords="offset points", ha="right",
        va="top",
        fontsize="small", color="grey"
    )
    ax.set_ylim(ylim)
    ax.set_xlim(min(bins), max(bins))
    ax.set_xlabel("W1 - W2")
    ax.set_ylabel("number of objects")
    ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', ncols=3, mode="expand", borderaxespad=0.)

    return fig


@Plotter.register("upright")
def baselines():
    info = load_data()["baseline_changes"]
    xkey = info["xkey"]
    xlabel = info["xlabel"]
    ykeys = info["ykeys"]
    ylabels = info["ylabels"]
    xbins = info["xbins"]
    xbin_mids = (xbins[1:] + xbins[:-1]) / 2
    diff_samples = info["diff_samples"]

    fig, axs = plt.subplots(sharex=True, nrows=len(ykeys), gridspec_kw={"hspace": 0})
    for i, (ax, ykey, ylabel) in enumerate(zip(axs, ykeys, ylabels)):
        ax.scatter(diff_samples[xkey], diff_samples[ykey], s=1, alpha=0.01, label="data", zorder=1)
        qs = info["qs"][ykey]
        ax.plot(xbin_mids, qs[:, 0.5], color="k", alpha=0.5, label="median", zorder=3)
        ax.plot(xbin_mids, qs[:, 0.16], color="k", ls="--", alpha=0.5, label=r"1 $\sigma$", zorder=3)
        ax.plot(xbin_mids, qs[:, 0.84], color="k", ls="--", alpha=0.5, zorder=3)
        ax.axhline(0, ls=":", color="k", alpha=0.5, zorder=2)
        ax.set_ylabel(ylabel)
        ax.set_ylim(-2.2, 2.2)
        indicate_news_cutoff(ax, annotate="bottom" if i == len(ykeys) - 1 else False, cutoff=info["news_cutoff"])
    axs[0].legend(loc="lower right")
    axs[-1].set_xlabel(xlabel)
    axs[-1].set_xlim(17.5, 7.5)

    return fig
