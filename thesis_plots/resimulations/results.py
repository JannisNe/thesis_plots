import logging
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker

from thesis_plots.plotter import Plotter


logger = logging.getLogger(__name__)

events = ["tywin", "lancel", "bran", "txs"]

ic_event_name = {
    "tywin": "IC200530A",
    "bran": "IC191001A",
    "lancel": "IC191119A",
    "txs": "IC170922A"
}


def get_data(event_name: str):
    files = {
        "lancel": "lancel__data_user_jnecker_tde_neutrinos_resim_lancel_out_separate_selection_2m_posvar_M=0.60_E=0.20_OnlineL2_SplineMPE_5_data.npz",
        "tywin": "tywin__data_user_jnecker_tde_neutrinos_resim_tywin_out_separate_selection2_M=0.80_E=0.20_OnlineL2_SplineMPE_6_data.npz",
        "txs": "txs__data_user_jnecker_tde_neutrinos_resim_txs_out_separate_selection2_M=0.60_E=0.20_OnlineL2_SplineMPE_6_data.npz",
        "bran": "bran__data_user_jnecker_tde_neutrinos_resim_bran_out8_charge_1-1400_truncated_energy.i3.zst__OnlineL2_SplineMPE_6_data.npz"
    }
    filename = Path(__file__).parent / "data" / "resim" / files[event_name]
    logger.debug(f"loading {filename}")
    return np.load(filename)


@Plotter.register("upright", arg_loop=events)
def abs_log_ratios(event_name: str):

    logger.debug(f"making plot for {event_name}")
    data = get_data(event_name)
    Esim_trunc = data["Esim_trunc"]
    Eratio = data["Eratio"]
    Emeas_ev = data["Emeas_ev"]

    fig, axs = plt.subplots(nrows=2, gridspec_kw={"height_ratios": [3, 1], "hspace": 0}, sharex="all")

    for i, (iE, iEratio) in enumerate(zip(Esim_trunc, Eratio)):
        marker = ""
        alpha = 0.1
        label = "Re-simulations" if i == 0 else ""
        axs[0].plot(iE, marker=marker, color="C0", alpha=alpha, label=label)
        axs[1].plot(iEratio, marker=marker, color="C0", alpha=alpha)

    axs[0].plot(Emeas_ev, color="k", label=ic_event_name.get(event_name, event_name), lw=2)
    axs[0].set_ylabel("E [GeV]")
    axs[1].set_xlabel("Segment")
    axs[1].set_ylabel("$E_\mathrm{sim} / E_\mathrm{data}$")
    axs[0].set_yscale("log")
    axs[1].set_yscale("log")
    axs[0].get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
    axs[1].axhline(1, color="k", lw=2)
    ylim = np.array(axs[0].get_ylim())
    ylim[1] *= 2
    axs[0].set_ylim(*ylim)
    axs[1].get_yaxis().set_major_locator(ticker.LogLocator(numticks=10, subs=[1.0, 2.0, 5.0]))
    axs[1].yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.1g}"))
    axs[0].legend(loc="upper right")

    return fig


def angle_distribution(event_name: str):
    data = get_data(event_name)
    bm = data["angle_bins"]
    qs = data["angle_quantiles"]
    cl = data["amgle_cl"]

    fig, ax = plt.subplots()
    ax.plot(bm, qs[:, 0], label="median")
    ax.fill_between(bm, qs[:, 1], qs[:, 2], alpha=0.4, label=f"{cl * 100:.0f}% IC")
    ax.set_xlabel("Simulated offset [deg]")
    ax.set_ylabel("max(|log$_{10}$(E$_{ratio}$)|)")
    ax.legend()

    return fig


@Plotter.register("fullpage")
def charge_plot():

    alert_color = "k"
    sim_color = "C2"

    fig, axss = plt.subplots(ncols=2, nrows=4, gridspec_kw={"wspace": 0, "hspace": 0.1}, sharey="row", sharex="col")

    for event_name, axs in zip(events, axss):
        data = get_data(event_name)
        charge_alert = data["charge_alert"]
        charge_simul = data["charge_simul"]
        z_om = data["z_om"]

        ylim = {
            "lancel": [0, 500],
            "tywin": [-500, 0],
            "txs": [-500, 0],
            "bran": [-500, 500]
        }[event_name]

        z_sorted = np.argsort(z_om)
        y = z_om[z_sorted]

        for i, ax in enumerate(axs):
            ax.plot(charge_alert[:, 1 + i][z_sorted], y, ls="-", lw=4, c=alert_color)
            ax.fill_betweenx(
                y, charge_simul[:, 1 + i * 3][z_sorted], charge_simul[:, 2 + i * 3][z_sorted],
                alpha=0.3, color=sim_color, label="Re-simulations Min/Max", linewidth=0
            )
            ax.plot(charge_simul[:, 3 + i * 3][z_sorted], y, color=sim_color, ls="--", label="Re-simulations median")
            xlim = ax.get_xlim()
            ax.fill_between(xlim - np.array([100, -100]), -50, -150, color="grey", alpha=0.3, linewidth=0)
            ax.set_xlim(xlim)

        # note the event name in top right corner, offset down and to the left
        axs[1].annotate(ic_event_name.get(event_name, event_name), (1, 1),
                        xycoords="axes fraction", xytext=(-2, -2), textcoords='offset points', ha="right", va="top")
        axs[1].annotate("dust layer", (axs[1].get_xlim()[1], -150), ha="right", va="baseline", color="grey")
        axs[0].set_ylim(*ylim)

    axss[0][0].legend(bbox_to_anchor=(1, 1), loc="lower center", borderaxespad=0.0, frameon=False, ncol=2)
    fig.supylabel("z [m]")
    axss[-1][1].set_xlabel("# hit DOMs")
    axss[-1][0].set_xlabel("Charge")

    return fig
