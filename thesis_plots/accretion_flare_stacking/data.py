import logging
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.gridspec import GridSpec
import pandas as pd
import os
from astropy.time import Time
from scipy.interpolate import InterpolatedUnivariateSpline

from thesis_plots.plotter import Plotter


logger = logging.getLogger(__name__)


def get_icecube_data_dir():
    icecube_dataset_dir = os.environ["FLARESTACK_DATASET_DIR"]
    if not icecube_dataset_dir:
        raise ValueError("FLARESTACK_DATASET_DIR not set. You need access to IceCube data to make this plot.")
    return Path(icecube_dataset_dir) / "ps_tracks" / "version-004-p00"


def get_binning():
    sindec_bins = np.unique(
        np.concatenate(
            [
                np.linspace(-1.0, -0.93, 4 + 1),
                np.linspace(-0.93, -0.3, 10 + 1),
                np.linspace(-0.3, 0.05, 9 + 1),
                np.linspace(0.05, 1.0, 18 + 1),
            ]
        )
    )
    energy_bins = np.arange(1.0, 9.5 + 0.01, 0.125)
    return sindec_bins, energy_bins


def alert_number_ps_tracks():
    ps_v004p00_dir = get_icecube_data_dir()
    mc_file = ps_v004p00_dir / "IC86_2016_MC.npy"
    best_fit_flux_100tev = 6.7e-18 / 3  # (u.GeV ** -1 * u.cm ** -2 * u.s ** -1 * u.sr ** -1)
    best_fit_gamma = 2.5
    best_fit_flux_gev = best_fit_flux_100tev * (100e3) ** best_fit_gamma
    ps_v004p00_ic86 = pd.DataFrame(np.load(mc_file))
    ps_v004p00_ic86["energy_weight"] = ps_v004p00_ic86["trueE"] ** (-best_fit_gamma)
    ps_v004p00_ic86["final_weight"] = (
            ps_v004p00_ic86["ow"] *
            365 * 24 * 3600
            * best_fit_flux_gev
            * ps_v004p00_ic86["trueE"] ** (-best_fit_gamma)
    )
    ps_energy_mask = ps_v004p00_ic86["trueE"] > 100e3
    n_ps_tracks = ps_v004p00_ic86["final_weight"][ps_energy_mask].sum()
    logger.info(f"Found {n_ps_tracks} events with energy > 100 TeV in PS Tracks")

    alert_infos = pd.read_csv(Path(__file__).parent / "data" / "IceCube_Gold_Bronze_Tracks.csv")
    alert_dates = pd.to_datetime(alert_infos["START"])
    alert_energy_mask = alert_infos["ENERGY"] < 100
    logger.info(f"Found {alert_energy_mask.sum()} alerts with energy < 100 TeV in alerts")

    time_bins = pd.date_range(start="2011-01-01", end="2020-01-01", freq="1Y", inclusive="both")
    alerts_per_year = alert_infos.groupby(pd.cut(alert_dates, time_bins))


@Plotter.register("wide")
def distribution_dec_energy():
    ps_v004p00_dir = get_icecube_data_dir()
    data_files = list(ps_v004p00_dir.glob(f"IC86_201*_exp.npy"))
    logger.debug(f"Found {len(data_files)} data files")
    data = np.concatenate([np.load(f) for f in data_files])
    start_mdj = Time("2018-05-23").mjd
    end_mjs = Time("2020-05-29").mjd
    mask = (data["time"] > start_mdj) & (data["time"] < end_mjs)
    data = data[mask]
    logger.debug(f"Loaded data files")
    decs = np.sin(data["dec"])
    energies = data["logE"]

    sindec_bins, energy_bins = get_binning()
    norm = colors.SymLogNorm(linthresh=1, linscale=1, vmin=0, vmax=5000)

    gridspec_kw = {
        "hspace": 0,
        "wspace": 0.,
        "height_ratios": [1, 2, .8],
        "width_ratios": [2, 1],

    }
    fig, axs = plt.subplots(nrows=3, ncols=2, gridspec_kw=gridspec_kw)
    # Main 2D histogram plot
    cmap = "Blues"
    h2d = axs[1][0].hist2d(decs, energies, bins=[sindec_bins, energy_bins], norm=norm, cmap=cmap)
    logger.debug(max(h2d[0].flatten()))
    axs[1][0].set_xticks([-1, -0.5, 0, 0.5, 1])
    axs[1][0].set_yticks([1, 3, 5, 7, 9])
    axs[1][0].set_xlabel(r"$\sin(\delta)$")
    axs[1][0].set_ylabel(r"$\log(E/\mathrm{GeV})$")

    # Top marginal histogram
    axs[0][0].hist(decs, bins=sindec_bins, density=True)
    axs[0][0].set_xticks([])  # Hide x-ticks on the top marginal plot
    axs[0][0].set_yticks([])  # Hide y-ticks on the top marginal plot
    axs[0][0].spines[['right', 'top', "left"]].set_visible(False)
    axs[0][0].set_xlim(-1, 1)

    # Right marginal histogram
    axs[1][1].hist(energies, bins=energy_bins, orientation="horizontal", density=True)
    axs[1][1].set_xticks([])  # Hide x-ticks on the right marginal plot
    axs[1][1].set_yticks([])  # Hide y-ticks on the right marginal plot
    axs[1][1].spines[['right', 'top', "bottom"]].set_visible(False)
    axs[1][1].set_ylim(axs[1][0].get_ylim())

    # Custom colorbar below the main plot
    cax = fig.add_subplot(axs[2][0])
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(mappable, ax=cax, orientation="horizontal")
    cbar.set_label("Counts")
    cax.set_xticks([])  # Hide x-ticks on the colorbar subplot
    cax.set_yticks([])  # Hide y-ticks on the colorbar subplot
    cax.spines[['right', 'top', "left", "bottom"]].set_visible(False)

    # Hide unnecessary subplots
    axs[2][1].axis('off')
    axs[2][0].axis('off')
    axs[0][1].axis('off')

    # Manually enable ticks on the main plot and remove ticks on the side plot
    axs[1][0].set_xticks([-1, -0.5, 0, 0.5, 1])
    axs[1][0].set_yticks([1, 3, 5, 7])

    # indicated most southern accretion flare
    dec_min = -11
    sin_dec_min = np.sin(np.radians(dec_min))
    axs[1][0].axvline(sin_dec_min, color="grey", ls="--")
    axs[1][0].annotate(rf"$\delta=${dec_min:.0f}$^\circ$", (sin_dec_min, max(axs[1][0].get_ylim())), xytext=(3, -3),
                       textcoords="offset points", color="grey", ha="left", va="top", rotation=90)

    return fig


@Plotter.register("wide")
def energy_pdf():
    gamma = 2
    ps_v004p00_dir = get_icecube_data_dir()
    data_files = list(ps_v004p00_dir.glob(f"IC86_201*_exp.npy"))
    logger.debug(f"Found {len(data_files)} data files")
    data = np.concatenate([np.load(f) for f in data_files])
    start_mdj = Time("2018-05-23").mjd
    end_mjs = Time("2020-05-29").mjd
    mask = (data["time"] > start_mdj) & (data["time"] < end_mjs)
    data = data[mask]
    logger.debug(f"Loaded data")
    mc_file = ps_v004p00_dir / "IC86_2016_MC.npy"
    mc = np.load(mc_file)
    mc_weights = mc["ow"] * mc["trueE"] ** -gamma

    sindec_bins, energy_bins = get_binning()

    # hist mc
    mc_hist, _ = np.histogramdd((np.sin(mc["dec"]), mc["logE"]), bins=[sindec_bins, energy_bins], weights=mc_weights)
    mc_hist = mc_hist.T
    mc_hist_normed = mc_hist / mc_hist.sum(axis=0)

    # hist data
    data_hist, _ = np.histogramdd((np.sin(data["dec"]), data["logE"]), bins=[sindec_bins, energy_bins])
    data_hist = data_hist.T
    data_hist[(data_hist == 0) & (mc_hist > 0)] = 1
    data_hist_normed = data_hist / data_hist.sum(axis=0)

    # ratio
    ratio = mc_hist_normed / data_hist_normed
    ratio[np.isnan(ratio)] = 0
    ratio[np.isinf(ratio)] = 0

    # ipdb.set_trace()

    vmin = 1e-4
    vmax = max([mc_hist_normed.max(), data_hist_normed.max()])
    logger.debug(f"vmin: {vmin}, vmax: {vmax}")
    norm = colors.LogNorm(vmin=vmin, vmax=vmax)

    height = plt.rcParams["figure.figsize"][1] / 2
    fs = (plt.rcParams["figure.figsize"][0], height)
    fig = plt.figure(figsize=fs)
    gs0 = GridSpec(1, 3, figure=fig, width_ratios=[1, 1, 1])
    gs01 = gs0[:2].subgridspec(2, 2, hspace=0, wspace=0.1, height_ratios=[.15, 1])
    gs02 = gs0[2].subgridspec(2, 1, height_ratios=[.15, 1], hspace=0)

    ax_sig = fig.add_subplot(gs01[1, 0])
    ax_sig.pcolormesh(sindec_bins, energy_bins, mc_hist_normed, cmap="gist_heat", norm=norm)
    ax_bkg = fig.add_subplot(gs01[1, 1], sharey=ax_sig, sharex=ax_sig)
    ax_bkg.pcolormesh(sindec_bins, energy_bins, data_hist_normed, cmap="gist_heat", norm=norm)
    ax_bkg.set_yticks([])
    ax_sig.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax_sig.set_xticklabels(["-1", "-0.5", "0", "0.5", "1"])

    # show colorbar
    cax = fig.add_subplot(gs01[0, :])
    mappable = cm.ScalarMappable(norm=norm, cmap="gist_heat")
    cbar = fig.colorbar(mappable, ax=cax, orientation="horizontal", location="top", fraction=.3)
    cbar.ax.tick_params(size=0, which="both")
    cbar.set_label("column-normalized density")
    cax.axis("off")

    ax_ratio = fig.add_subplot(gs02[1], sharex=ax_sig, sharey=ax_sig)
    ratio_norm = colors.LogNorm(vmin=1e-3, vmax=1e3)
    ax_ratio.pcolormesh(sindec_bins, energy_bins, ratio, cmap="RdBu_r", norm=ratio_norm)
    ax_ratio.set_yticks([])
    cax_ratio = fig.add_subplot(gs02[0])
    mappable = cm.ScalarMappable(norm=ratio_norm, cmap="RdBu_r")
    cbar_ratio = fig.colorbar(mappable, ax=cax_ratio, orientation="horizontal", location="top", fraction=.3)
    cbar_ratio.set_label("Signal/Background")
    cbar_ratio.ax.tick_params(size=0)
    cax_ratio.axis("off")

    axs = [ax_sig, ax_bkg, ax_ratio]
    ax_sig.annotate("Signal", (0, 1),
                    xycoords="axes fraction", xytext=(2, 2), textcoords="offset points", ha="left", va="bottom")
    ax_bkg.annotate("Background", (1, 1),
                    xycoords="axes fraction", xytext=(-2, 2), textcoords="offset points", ha="right", va="bottom")
    ax_bkg.set_xlabel(r"$\sin(\delta)$")
    ax_sig.set_ylabel(r"$\log(E/\mathrm{GeV})$")
    ax_sig.set_ylim(2, 7)

    return fig


@Plotter.register(["margin", "notopright"])
def background_spatial():
    ps_v004p00_dir = get_icecube_data_dir()
    data_files = list(ps_v004p00_dir.glob(f"IC86_201*_exp.npy"))
    logger.debug(f"Found {len(data_files)} data files")
    data = np.concatenate([np.load(f) for f in data_files])
    start_mdj = Time("2018-05-23").mjd
    end_mjs = Time("2020-05-29").mjd
    mask = (data["time"] > start_mdj) & (data["time"] < end_mjs)
    data = data[mask]
    logger.debug(f"Loaded data")
    sindecs = np.sin(data["dec"])

    sindec_bins, energy_bins = get_binning()

    hist, bins = np.histogram(
        sindecs,
        density=True,
        bins=sindec_bins,
    )

    bins = np.concatenate([bins[:1], bins, bins[-1:]])
    hist = np.concatenate([hist[:1], hist, hist[-1:]])

    bkg_spline = InterpolatedUnivariateSpline((bins[1:] + bins[:-1]) / 2.0, hist, k=2)
    x = np.linspace(-1, 1, 1000)
    y = bkg_spline(x)

    fig, ax = plt.subplots()
    ax.bar(bins[:-1], hist, width=np.diff(bins), align="edge", label="data", ec="white")
    ax.plot(x, y, c="C1", label="interpolation")
    ax.set_xlim(-1, 1)
    ax.set_xlabel(r"$\sin(\delta)$")
    ax.set_ylabel(r"density")
    ax.legend(bbox_to_anchor=(0.5, 1.1), loc="lower center", borderaxespad=0.0, ncol=1)
    return fig
