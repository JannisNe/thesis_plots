import logging
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import pandas as pd
import os
import ipdb
from networkx.algorithms.bipartite import density

from thesis_plots.plotter import Plotter


logger = logging.getLogger(__name__)


def get_icecube_data_dir():
    icecube_dataset_dir = os.environ["FLARESTACK_DATASET_DIR"]
    if not icecube_dataset_dir:
        raise ValueError("FLARESTACK_DATASET_DIR not set. You need access to IceCube data to make this plot.")
    return Path(icecube_dataset_dir) / "ps_tracks" / "version-004-p00"


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
    logger.debug(f"Loaded data files")
    decs = np.sin(data["dec"])
    energies = data["logE"]

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
    norm = colors.SymLogNorm(linthresh=1, linscale=1, vmin=0, vmax=100000)

    gridspec_kw = {
        "hspace": 0.,
        "wspace": 0.,
        "height_ratios": [1, 2, .5],
        "width_ratios": [2, 1],
    }
    fig, axs = plt.subplots(nrows=3, ncols=2, gridspec_kw=gridspec_kw)
    axs[1][0].hist2d(decs, energies, bins=[sindec_bins, energy_bins], norm=norm)
    cbar = fig.colorbar(cm.ScalarMappable(norm=norm), ax=axs[2][0], orientation="horizontal")
    cbar.set_label("Counts")
    axs[0][0].sharex(axs[1][0])
    axs[0][0].hist(decs, bins=sindec_bins, density=True)
    axs[1][1].sharey(axs[1][0])
    axs[1][1].hist(energies, bins=energy_bins, orientation="horizontal", density=True)
    for ax in [axs[2][0], axs[1][1], axs[0][0], axs[2][1], axs[0][1]]:
        ax.spines[['right', 'top', "left", "bottom"]].set_visible(False)
        ax.set_yticks([])
        ax.set_xticks([])
    axs[1][0].set_xticks([-1, -.5, 0, .5, 1])
    axs[1][0].set_yticks([1, 3, 5, 7])
    axs[1][0].set_xlabel(r"$\sin(\delta)$")
    axs[1][0].set_ylabel(r"$\log(E/\mathrm{GeV})$")
    return fig
