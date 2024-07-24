import logging
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import numpy as np
from thesis_plots.plotter import Plotter
from thesis_plots.flaires.data.load import load_data
from timewise_sup.meta_analysis.luminosity import ref_time_key
from air_flares.plots.temperature_plots import (
    make_temperature_lightcurve_plot,
    make_temperature_radius_plot,
    make_temperature_fit_plot
)


logger = logging.getLogger(__name__)


@Plotter.register()
def lightcurve():
    data = load_data()["ngc7392_data"]
    info = data["info"]
    good_temp = data["good_temp"]
    lc_kwargs = data["lc_kwargs"]

    fig, axs = plt.subplots(
        ncols=1,
        nrows=2,
        gridspec_kw={"hspace": 0, "wspace": 0},
        sharex="all",
        figsize=np.array(plt.rcParams["figure.figsize"]) * np.array([1, 2]),
    )
    make_temperature_lightcurve_plot(axs[0], info, **lc_kwargs)
    temp_ax, radius_ax = make_temperature_radius_plot(axs[1], info[good_temp], x_key=ref_time_key)
    radius_ax.set_yscale("linear")
    axs[0].grid(False)
    axs[0].set_ylim(-5e42, 6e43)
    axs[1].set_xlabel("t [d]")
    axs[1].set_xlim(right=1300)

    return fig


@Plotter.register()
def temperature_fit():
    data = load_data()["ngc7392_data"]
    info = data["info"]
    lc_kwargs = data["lc_kwargs"]
    nu = data["nu"]
    good_temp_with_uncertainty = data["good_temp_with_uncertainty"]
    chains = data["chains"]

    # --- make temperature fit plot --- #

    n_cols = 4
    n_rows = 2
    fig, axs = plt.subplots(
        ncols=n_cols,
        nrows=n_rows,
        gridspec_kw={"wspace": 0., "hspace": 0.},
        sharex="all", sharey="row",
        figsize=np.array(plt.rcParams["figure.figsize"]) * 2,
    )

    nu_lim_exponent = [13.5, 14.2]

    for ii, (i, i_info) in enumerate(info[good_temp_with_uncertainty].iterrows()):
        ax = axs.flatten()[ii]
        chain = chains[i]
        n_choies = min((100, chain.shape[0]))
        logger.debug(f"{i}: plotting {n_choies} samples from the chain")
        choice = np.random.choice(chain.shape[0], n_choies, replace=False)
        chain_selected = chain[choice].T
        temperature_chain, radius_chain = chain_selected

        make_temperature_fit_plot(
            ax,
            temperature=i_info["temperature"],
            temperature_err_low=i_info["temperature_16perc"],
            temperature_err_high=i_info["temperature_84perc"],
            radius=i_info["radius"],
            radius_err_low=i_info["radius_16perc"],
            radius_err_high=i_info["radius_84perc"],
            nu=nu,
            fluxes=[i_info[f"{b}_{lc_kwargs['y_key']}"] for b in ["W1", "W2"]],
            fluxes_err=[i_info[f"{b}_{lc_kwargs['y_err_key']}"] for b in ["W1", "W2"]],
            xlim_exponent=nu_lim_exponent,
            temperature_chain=temperature_chain,
            radius_chain=radius_chain,
        )
        ax.set_title(f"t={i_info[ref_time_key]:.0f}d", pad=-14, y=1)

    axs[1, 0].set_xscale("log")
    axs[1, 0].set_xlim(10 ** nu_lim_exponent[0], 10 ** nu_lim_exponent[1])
    for ax in axs.flatten():
        ax.set_ylim(4e41, 4e43)
        ax.set_yscale("log")
    offset_value = 1e14
    xticks = np.array([0.5, 1, 1.5])
    xtick_labels = ["0.5", "1", "1.5"]
    for ax in axs[1]:
        ax.set_xticks(xticks * offset_value)
        ax.set_xticklabels(xtick_labels, fontsize="small")
        ax.xaxis.set_minor_formatter(NullFormatter())
    last_ax = axs.flatten()[-1]
    last_ax.annotate(f"{offset_value:.0e}", (1, 0), xycoords="axes fraction", fontsize="small")

    fig.supxlabel(r"$\nu$ [Hz]")
    fig.supylabel(r"$\nu L_\nu$ [erg s$^{-1}$]")

    return fig
