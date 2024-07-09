import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle

from thesis_plots.plotter import Plotter
from thesis_plots.resimulations.results import ic_event_name


logger = logging.getLogger(__name__)


def get_data(event_name: str):
    fn = Path(__file__).parent / "data" / f"{event_name}_calib_data.pkl"
    logger.debug(f"loading {fn}")
    with open(fn, "rb") as f:
        data = pickle.load(f)
    return data


def get_max_e_ratio(data):
    Emeas = data["Emeas"]
    Esim = data["Esim"]
    Esim_trunc = np.array([E[:len(Emeas)] for E in Esim])
    Eratio = np.array([E / Emeas for E in Esim_trunc])
    return np.max(abs(np.log10(Eratio)), axis=1)


@Plotter.register(["margin", "notopright"], arg_loop=["bran", "txs", "tywin"])
def metric_histogram(event_name):
    fig, ax = plt.subplots()
    max_e_ratio = get_max_e_ratio(get_data(event_name))
    logger.info(f"{len(max_e_ratio)} simulations for {event_name}")
    ax.hist(max_e_ratio, density=True, cumulative=True, zorder=1)
    ax.set_xlabel("$M$")
    ax.set_ylabel("CDF")
    ax.set_xlim(left=0)
    return fig


@Plotter.register(["margin", "notopright"])
def metric_calibration():
    tywin_m = get_max_e_ratio(get_data("tywin"))
    txs_m = get_max_e_ratio(get_data("txs"))
    m_thresh = np.linspace(0, 2, 100)
    p_tywin = np.sum(tywin_m[:, np.newaxis] > m_thresh, axis=0) / len(tywin_m)
    p_txs = np.sum(txs_m[:, np.newaxis] < m_thresh, axis=0) / len(txs_m)
    logger.info("1-CDF(0.8) for IC170922A: {:.2f}".format(sum(txs_m > 0.8) / len(txs_m)))

    fig, ax = plt.subplots()
    ax.plot(m_thresh, p_tywin, label="purity")
    ax.plot(m_thresh, p_txs, label="efficiency", ls=":", color="C3")
    ax.set_xlabel("$M$")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1), ncol=2)
    ax.axvline(0.8, ls="--", color="C1")
    return fig


@Plotter.register()
def original_resimulations_ratios():
    figsize = plt.rcParams["figure.figsize"]
    figsize[1] *= 2
    fig, axs = plt.subplots(figsize=figsize, nrows=2, sharex=True, gridspec_kw={"hspace": 0})

    for ev, ax in zip(["txs", "tywin"], axs):
        data = get_data(ev)
        Emeas_ev = data["Emeas"]
        Esim = data["Esim"]
        Esim_trunc = np.array([E[:len(Emeas_ev)] for E in Esim])
        Eratio = np.array([E / Emeas_ev for E in Esim_trunc])

        ax.axhline(1, color="k", lw=2, label="Original")
        for i, (iE, iEratio) in enumerate(zip(Esim_trunc, Eratio)):
            marker = ""
            label = "Resimulations" if i == 0 else ""
            ax.plot(iEratio, marker=marker, color="C0", alpha=0.3, label=label)
        ax.set_yscale("log")
        ax.annotate(ic_event_name[ev], (0, 0), xycoords="axes fraction", xytext=(3, 3), textcoords="offset points",
                    ha="left", va="bottom")

    axs[1].set_xlim(0, 8)
    axs[1].set_yticks([1e-2, 1e-1, 1, 1e1, 1e2])
    axs[1].set_xlabel("Segment $k$")
    axs[0].legend(loc="lower center", borderaxespad=0.5, ncol=1, bbox_to_anchor=(0.5, 1), ncols=2)
    fig.supylabel(r"$E_\mathrm{k,sim} / E_\mathrm{k,meas}$", x=-0.05)
    return fig


@Plotter.register()
def tywin_original_resimulations_charge():
    data = get_data("tywin")
    charge_alert = data["charge_alert"]
    charge_simul = data["charge_simul"]
    z_om = data["z_om"]
    ylim = [-500, 0],

    alert_color = "k"
    sim_color = "C2"

    fig, axs = plt.subplots(ncols=2, gridspec_kw={"wspace": 0}, sharey="row", sharex="col")

    z_sorted = np.argsort(z_om)
    y = z_om[z_sorted]

    for i, ax in enumerate(axs):
        ax.fill_betweenx(
            y, charge_simul[:, 1 + i * 3][z_sorted], charge_simul[:, 2 + i * 3][z_sorted],
            alpha=0.3, color=sim_color, label="Resimulations Min/Max", linewidth=0, zorder=2
        )
        ax.plot(charge_simul[:, 3 + i * 3][z_sorted], y, color=sim_color, ls="--", label="Resimulations median", zorder=10)
        xlim = ax.get_xlim()
        ax.fill_between(xlim - np.array([100, -100]), -50, -150, color="grey", alpha=0.3, linewidth=0)
        ax.plot(charge_alert[:, 1 + i][z_sorted], y, ls="-", lw=4, c=alert_color, label="Original", zorder=5)
        ax.set_xlim(xlim)

    axs[1].annotate("dust layer", (axs[1].get_xlim()[1], -150), ha="right", va="baseline", color="grey")
    axs[0].set_ylim(*ylim)

    axs[0].legend(bbox_to_anchor=(1, 1), loc="lower center", borderaxespad=0.5, ncol=2)
    axs[0].set_ylabel("z [m]")
    axs[1].set_xlabel("# hit DOMs")
    axs[0].set_xlabel("Charge")
    return fig
