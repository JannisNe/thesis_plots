import logging
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
import pandas as pd
import pickle
import json
from scipy.interpolate import interp1d

from matplotlib.lines import Line2D

from thesis_plots.plotter import Plotter
from thesis_plots.dust_echos.model import model_colors
from thesis_plots.arrow_handler import HandlerArrow
from thesis_plots.icecube_diffuse_flux import load_spectrum, Spectrum


logger = logging.getLogger(__name__)


@Plotter.register()
def alert_number_constraint():
    storage_fn = Path(__file__).parent / "data" / "Ns_above_100.00tev.csv"
    df = pd.read_csv(storage_fn)
    gammas = df.gamma
    Ns = df.N

    fig, ax = plt.subplots()
    ax.errorbar(gammas, Ns, yerr=0.2, uplims=True)
    ax.set_ylabel(r'$N_{\mathrm{alerts}}$ proxy')
    ax.set_xlabel(r"$\gamma$")
    ax.axhline(3, c='gray', ls='-', alpha=0.5)
    ax.annotate(r'$N_{\mathrm{alerts}}$=3', (max(ax.get_xlim()), 3), xytext=(-2, 2), textcoords="offset points", ha="right",
                va="bottom", color='gray')
    ax.axvline(1, c=model_colors["IR"], ls="--", label="IR")
    ax.axvline(1.5, c=model_colors["OUV"], ls=":", label="OUV")
    ax.axvline(2.5, c=model_colors["X-ray"], ls="-.", label="X-ray")
    limit_handle = patches.FancyArrowPatch((1.5, 3), (1.5, 3.5), arrowstyle="-|>", mutation_scale=10, color="C0")
    handles = ax.get_legend_handles_labels()[0] + [limit_handle]
    labels = ax.get_legend_handles_labels()[1] + [r"Upper limits"]
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
    gammas = {
        "X-ray": 2.5,
        "OUV": 1.5,
        "IR": 1,
    }

    data_dir = Path(__file__).parent / "data"
    with (data_dir / "cumulative_fluxes.json").open("r") as f:
        fluxes = json.load(f)
    with open(data_dir / "energy_range_nan.pkl", "rb") as f:
        energy_range = pickle.load(f)

    s = load_spectrum("joint15")
    srange = s.get_energy_range()
    slower = s.lower(68, srange) * srange ** 2
    supper = s.upper(68, srange) * srange ** 2
    diffuse_100tev = s.best(1e5)
    diffuse_1pev = s.best(1e6)

    fig, axs = plt.subplots(ncols=3, sharex=True, sharey=True, gridspec_kw={"wspace": 0.0},
                            figsize=(plt.rcParams["figure.figsize"][0], plt.rcParams["figure.figsize"][1] * 2/3))

    for ax in axs:
        diffuse_handle = ax.fill_between(srange, slower, supper,
                                         color="black", alpha=.2, label="Diffuse Flux", zorder=1, ec="none")
    for i, (model, gamma) in enumerate(gammas.items()):
        ax = axs[i]
        x = np.logspace(*np.log10(energy_range[gamma]), 3)
        y = fluxes[str(gamma)] * x**(2 - gamma)
        ax.errorbar(x, y, yerr=0.2 * y, uplims=True, zorder=3, c="C0")

        fct = interp1d(model_data[model]["E"], model_data[model]["flux"], fill_value="extrapolate")
        ir_100tev = (fct(1e5) / 1e5**2) * 500
        gamma1_at_100tev = fluxes[str(gamma)] * 1e5**(-gamma)
        at1pev = fluxes[str(gamma)] * 1e6**(-gamma)
        logger.info(
            f""
            f"{model} model at 100 TeV: {ir_100tev:.2e}, "
            f"gamma={gamma} at 100 TeV: {gamma1_at_100tev:.2e}, "
            f"ratio: {gamma1_at_100tev / ir_100tev:.4e}"
        )
        diffuse_perc_100tev = gamma1_at_100tev / diffuse_100tev * 100
        diffuse_perc_1pev = at1pev / diffuse_1pev * 100
        logger.info(f"   result: {diffuse_perc_100tev:.4f}% of diffuse flux at 100 TeV")
        logger.info(f"   result: {diffuse_perc_1pev:.4f}% of diffuse flux at 1 PeV")
        ax.plot(model_data[model]["E"], model_data[model]["flux"], c=model_colors[model], ls=model_ls[model],
                zorder=2)

    axs[0].set_xscale("log")
    axs[0].set_yscale("log")
    axs[0].set_xlim(10**3.8, 10**7.5)
    axs[0].set_ylim(bottom=1e-11)

    model_handles = [
        Line2D([], [], color=model_colors[model], ls=model_ls[model], label=model)
        for model in gammas
    ]
    legend_handles = [
        diffuse_handle,
        *model_handles,
        patches.FancyArrowPatch((0, 0), (0, -0.5), color="C0", arrowstyle="-|>", mutation_scale=10, shrinkA=-2, shrinkB=0)
    ]
    legend_labels = [
        "Diffuse Flux",
        *gammas.keys(),
        "Upper Limit"
    ]

    axs[0].set_ylabel(r"$E^2\Phi_\mu^{\nu + \bar{\nu}}$ [GeV cm$^{-2}$ s$^{-1}$ sr$^{-1}$]")
    axs[1].legend(legend_handles, legend_labels, loc="lower center", ncol=5, bbox_to_anchor=(0.5, 1.01),
                  handler_map={patches.FancyArrowPatch: HandlerArrow()})
    axs[1].set_xlabel("E [GeV]", va="top")
    return fig
