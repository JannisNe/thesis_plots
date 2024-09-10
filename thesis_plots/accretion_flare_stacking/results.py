import logging
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches, lines
import pandas as pd
import pickle
import json
from scipy.interpolate import interp1d

from matplotlib.lines import Line2D

from thesis_plots.plotter import Plotter
from thesis_plots.dust_echos.model import model_colors
from thesis_plots.arrow_handler import HandlerArrow
from thesis_plots.icecube_diffuse_flux import get_diffuse_flux_functions


logger = logging.getLogger(__name__)


@Plotter.register()
def alert_number_constraint():
    storage_fn = Path(__file__).parent / "data" / "Ns_above_100.00tev.csv"
    df = pd.read_csv(storage_fn)
    gammas = df.gamma
    Ns = df.N

    fig, ax = plt.subplots()
    ax.errorbar(gammas, Ns, yerr=0.2, uplims=True)
    ax.set_ylabel(r'$N_{\nu}(E>100\,\mathrm{TeV})$')
    ax.set_xlabel(r"$\gamma$")
    ax.axhline(3, c='gray', ls='-', alpha=0.5)
    ax.annotate(r'$N_{\nu}$=3', (max(ax.get_xlim()), 3), xytext=(-2, 2), textcoords="offset points", ha="right",
                va="bottom", color='gray')
    ax.axvline(2, c=model_colors["OUV"], ls=":", label="X-ray / OUV")
    ax.axvline(1, c=model_colors["IR"], ls="--", label="IR")
    limit_handle = patches.FancyArrowPatch((1.5, 3), (1.5, 3.5), arrowstyle="-|>", mutation_scale=10, color="C0")
    handles = ax.get_legend_handles_labels()[0] + [limit_handle]
    labels = ax.get_legend_handles_labels()[1] + [r"limits"]
    ax.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 1.01),
              handler_map={patches.FancyArrowPatch: HandlerArrow()})
    return fig


@Plotter.register(["margin", "notopright"])
def ts_distribution():
    background_filename = Path(__file__).parent / "data" / "0.pkl"
    with open(background_filename, "rb") as f:
        ts_background = pickle.load(f)["TS"]
    med_ts = np.median(ts_background)

    unblinding_filename = Path(__file__).parent / "data" / "unblinding_results.pkl"
    with open(unblinding_filename, "rb") as f:
        u = pickle.load(f)

    logger.info(f"background median TS: {med_ts}")
    logger.info(f"observed TS: {u['TS']}")

    fig, ax = plt.subplots()
    ax.hist(ts_background, bins=10, density=True, label='background \ndistribution', alpha=1)

    ls = '-'
    ax.axvline(u["TS"], label="$\lambda_\mathrm{obs}$", c=f"C1", ls=ls)
    ax.axvline(med_ts, label="$\lambda_\mathrm{bkg}$", color='k', alpha=1, ls='--')
    ax.set_yscale('log')
    ax.legend(bbox_to_anchor=(0.5, 1.1), loc="lower center", borderaxespad=0.0, ncol=1)
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel("density")
    return fig


@Plotter.register("wide")
def diffuse_flux():
    model_dir = Path(__file__).parent.parent / "dust_echos" / "data"
    model_filenames = {
        "IR": "winter_lunardini_ir_diffuse_flux.csv",
        "X-ray": "winter_lunardini_xray_diffuse_flux.csv",
        "OUV": "winter_lunardini_ouv_diffuse_flux.csv",
    }
    model_ls = {"IR": "--", "X-ray": ":", "OUV": "-."}
    model_data = {
        key: pd.read_csv(model_dir / fn, decimal=",", delimiter=";", names=["E", "flux"])
        for key, fn in model_filenames.items()
    }
    models = {
        1: ["IR", "OUV"],
        2: ["X-ray"],
    }

    data_dir = Path(__file__).parent / "data"
    with (data_dir / "cumulative_fluxes.json").open("r") as f:
        fluxes = json.load(f)
    with open(data_dir / "energy_range_nan.pkl", "rb") as f:
        energy_range = pickle.load(f)
    best_f, lower_f, upper_f, e_range = get_diffuse_flux_functions("joint_15")

    fig, ax = plt.subplots()
    diffuse_handle = ax.fill_between(e_range, lower_f(e_range) * e_range ** 2, upper_f(e_range) * e_range ** 2,
                                     color="black", alpha=.2, label="Diffuse Flux", zorder=1, ec="none")
    for i, gamma in enumerate(energy_range):
        x = np.logspace(*np.log10(energy_range[gamma]), 3)
        y = fluxes[str(gamma)] * x**(2 - gamma)
        ax.errorbar(x, y, yerr=0.2 * y, uplims=True, zorder=3, c="C0")

        if gamma == 1:
            fct = interp1d(model_data["IR"]["E"], model_data["IR"]["flux"], fill_value="extrapolate")
            ir_100tev = (fct(1e5) / 1e5**2) * 500
            gamma1_at_100tev = fluxes[str(gamma)] * 1e5**(-gamma)
            logger.info(
                f""
                f"IR model at 100 TeV: {ir_100tev:.2e}, "
                f"gamma=1 at 100 TeV: {gamma1_at_100tev:.2e}, "
                f"ratio: {gamma1_at_100tev / ir_100tev:.4e}"
            )

    for model in model_ls:
        ax.plot(model_data[model]["E"], model_data[model]["flux"], c=model_colors[model], ls=model_ls[model],
                zorder=2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e3, 1e9)
    ax.set_ylim(bottom=1e-12)

    model_handles = [
        Line2D([], [], color=model_colors[model], ls=model_ls[model], label=model)
        for model in model_ls
    ]
    legend_handles = [
        diffuse_handle,
        *model_handles,
        patches.FancyArrowPatch((0, 0), (0, -0.5), color="C0", arrowstyle="-|>", mutation_scale=10, shrinkA=-2, shrinkB=0)
    ]
    legend_labels = [
        "Diffuse Flux",
        *model_ls.keys(),
        "Upper Limit"
    ]

    ax.set_ylabel(r"$E^2\Phi_\mu^{\nu + \bar{\nu}}$ [GeV$^{-1}$ cm$^{-2}$ s$^{-1}$ sr$^{-1}$]")
    ax.legend(legend_handles, legend_labels, loc="lower center", ncol=5, bbox_to_anchor=(0.5, 1.01),
                  handler_map={patches.FancyArrowPatch: HandlerArrow()})
    ax.set_xlabel("Energy [GeV]", va="top")
    return fig
