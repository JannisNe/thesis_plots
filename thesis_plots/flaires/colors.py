import logging
from copy import copy
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
import numpy as np
import pandas as pd
from thesis_plots.flaires.data.load import load_data
from thesis_plots.plotter import Plotter
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
    ax.bar(bm, info["h1"], width=width, color="C0", align="center", label="dust echoes")
    # make a histogram of all sources weighted with N_dust_echoes / N_all
    ax.step(bins[:-1], info["h2"], color="C1", where="post", label="all (scaled)")
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
    ylabels = [r"$\Delta$ W1", r"$\Delta$ W2", r"$\Delta$ (W1 - W2)"]
    xbins = info["xbins"]
    xbin_mids = (xbins[1:] + xbins[:-1]) / 2
    splits = [info["diff_sample"].query(f"{xkey} >= {d} and {xkey} < {u}") for d, u in zip(xbins[:-1], xbins[1:])]
    diff_samples = pd.concat([s.sample(100) if len(s) > 100 else s for s in splits])

    fig, axs = plt.subplots(sharex=True, nrows=len(ykeys), gridspec_kw={"hspace": 0})
    for i, (ax, ykey, ylabel) in enumerate(zip(axs, ykeys, ylabels)):
        ax.scatter(diff_samples[xkey], diff_samples[ykey], s=1, alpha=0.2, label="data", zorder=1)
        qs = info["qs"][ykey]
        ax.plot(xbin_mids, qs[:, 0.5], color="k", alpha=0.5, label="median", zorder=3)
        ax.plot(xbin_mids, qs[:, 0.16], color="k", ls="--", alpha=0.5, label=r"1 $\sigma$", zorder=3)
        ax.plot(xbin_mids, qs[:, 0.84], color="k", ls="--", alpha=0.5, zorder=3)
        ax.axhline(0, ls=":", color="k", alpha=0.5, zorder=2)
        ax.set_ylabel(ylabel)
        ax.set_ylim(-2.2, 2.2)
        indicate_news_cutoff(ax, annotate="bottom" if i == len(ykeys) - 1 else False, cutoff=info["news_cutoff"])
    axs[0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3)
    axs[-1].set_xlabel(xlabel)
    axs[-1].set_xlim(17.5, 7.5)

    return fig


@Plotter.register("margin")
def baselines_zoom():
    info = load_data()["baseline_changes"]
    xkey = info["xkey"]
    xlabel = info["xlabel"]
    ykeys = info["ykeys"]
    ylabels = [r"$\Delta$ W1", r"$\Delta$ (W1 - W2)"]
    xbins = info["xbins"]
    xbin_mids = (xbins[1:] + xbins[:-1]) / 2
    splits = [info["diff_sample"].query(f"{xkey} >= {d} and {xkey} < {u}") for d, u in zip(xbins[:-1], xbins[1:])]
    diff_samples = pd.concat([s.sample(100) if len(s) > 100 else s for s in splits])
    m = (diff_samples["W1_diff"] > .2) & (diff_samples["W2_diff"] > .2)

    height = plt.rcParams["figure.figsize"][1] * 2
    width = plt.rcParams["figure.figsize"][0]

    fig, axs = plt.subplots(sharex=True, nrows=2, gridspec_kw={"hspace": 0}, figsize=(width, height))

    for i, (ax, ykey, ylabel) in enumerate(zip(axs, ["W1_diff", "color_change"], ylabels)):
        handlers = []
        for im, w, c, ls, mark in zip([m, ~m], ["high", "low"], ["C1", "C0"], ["-", "--"], ["o", "s"]):
            p1 = ax.scatter(diff_samples[xkey][im], diff_samples[ykey][im],
                       s=3, alpha=.1, label="data", zorder=1, color=c, ec="none", marker=mark)
            qs = info[f"qs_{w}"][ykey]
            p2 = ax.plot(xbin_mids, qs[:, 0.5], color=c, alpha=1, label="median", zorder=3, ls=ls)
            ax.fill_between(xbin_mids, qs[:, 0.16], qs[:, 0.84], color=c, alpha=.5, ec="none")
            p1 = copy(p1)
            p1.set_sizes([10])
            p1.set_alpha(.5)
            handlers.append((p1, p2[0]))
        ax.axhline(0, ls=":", color="k", alpha=0.5, zorder=2)
        ax.set_ylabel(ylabel)
        ax.set_ylim(-2.2, 2.2)
        indicate_news_cutoff(ax, annotate="bottom" if i == len(ykeys) - 1 else False, cutoff=info["news_cutoff"])
    axs[0].legend(handlers, ["variable", "steady"], loc="lower center", bbox_to_anchor=(0.5, 1.02),
                  ncol=3, handler_map={tuple: HandlerTuple(ndivide=None)})
    axs[-1].set_xlabel(xlabel)
    axs[-1].set_xlim(17.5, 7)
    axs[-1].set_ylim(-.25, .25)

    return fig
