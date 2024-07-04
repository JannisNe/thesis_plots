import logging
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from astropy.coordinates import SkyCoord

from thesis_plots.plotter import Plotter
from thesis_plots.resimulations.ratio_segments import ratio_plot


logger = logging.getLogger(__name__)

events = ["tywin", "lancel", "bran", "txs"]

ic_event_name = {
    "tywin": "IC200530A",
    "bran": "IC191001A",
    "lancel": "IC191119A",
    "txs": "IC170922A"
}

em_counterpart = {
    "tywin": ["AT2019fdr", (257.2786, 26.8557)],  # Reusch et al. (2022)
    "bran": ["AT2019dsg", (314.26, 14.20)],  # Stein et al. (2021)
    "lancel": ["AT2019aalc", (231.069435, 4.855293)],  # https://www.wis-tns.org/object/2019aalc
    "txs": ["TXS 0506+056", (77.35818525834, 05.69314816610)]  # http://cdsportal.u-strasbg.fr/?target=TXS%200506%2B056
}


def get_data(event_name: str):
    files = {
        "lancel": "lancel__data_user_jnecker_tde_neutrinos_resim_lancel_out_separate_selection_posvar_M=0.60_E=0.20_OnlineL2_SplineMPE_5_data.npz",
        "tywin": "tywin__data_user_jnecker_tde_neutrinos_resim_tywin_out_separate_selection2_M=0.80_E=0.20_OnlineL2_SplineMPE_6_data.npz",
        "txs": "txs__data_user_jnecker_tde_neutrinos_resim_txs_out_separate_selection2_M=0.60_E=0.20_OnlineL2_SplineMPE_6_data.npz",
        "bran": "bran__data_user_jnecker_tde_neutrinos_resim_bran_out8_charge_1-1400_truncated_energy.i3.zst__OnlineL2_SplineMPE_6_data.npz"
    }
    filename = Path(__file__).parent / "data" / "resim" / files[event_name]
    logger.debug(f"loading {filename}")
    return np.load(filename, allow_pickle=True)


@Plotter.register("upright", arg_loop=events)
def abs_log_ratios(event_name: str):
    logger.debug(f"making plot for {event_name}")
    data = get_data(event_name)
    Esim_trunc = data["Esim_trunc"]
    Eratio = data["Eratio"]
    Emeas_ev = data["Emeas_ev"]
    return ratio_plot(Esim_trunc, Emeas_ev, Eratio, alpha=0.1,
                      formatter=ticker.FuncFormatter(lambda x, _: f"{x:.1f}"))


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


def alert_scatter(event_name: str):
    data = get_data(event_name)
    alert_coord = data["alert_coord"].tolist()
    offsets = data["offsets"]

    em_counterpart_coord = SkyCoord(*em_counterpart[event_name][1], unit="deg")
    em_counterpart_offset = [a.to("deg").value for a in alert_coord.spherical_offsets_to(em_counterpart_coord)]

    fig, ax = plt.subplots()
    ax.scatter(*np.array(offsets).T,
               marker="o", label="Re-simulations", s=2, alpha=0.3, edgecolors="none")
    ax.scatter(0, 0,
               marker="X", label="Best Fit", edgecolors="k", linewidths=0.5)
    ax.scatter(*em_counterpart_offset,
               marker="*", label=em_counterpart[event_name][0], edgecolors="k", linewidths=0.5)
    ax.set_aspect("equal")
    ax.set_xlim([-5, 5])
    ax.set_ylim([-5, 5])
    ax.set_xlabel(r"$\Delta$RA [deg]")
    ax.set_ylabel(r"$\Delta$Dec [deg]")
    ax.legend()

    return fig


@Plotter.register()
def alert_scatter_combined():
    width = plt.rcParams["figure.figsize"][0]
    figsize = width, width
    fig, axss = plt.subplots(
        ncols=2, nrows=2,
        gridspec_kw={"wspace": 0.08, "hspace": 0.08},
        sharex="all", sharey="all",
        figsize=figsize
    )

    for event_name, ax in zip(events, axss.flatten()):
        data = get_data(event_name)
        alert_coord = data["alert_coord"].tolist()
        offsets = data["offsets"]

        em_counterpart_coord = SkyCoord(*em_counterpart[event_name][1], unit="deg")
        em_counterpart_offset = [a.to("deg").value for a in alert_coord.spherical_offsets_to(em_counterpart_coord)]

        ax.scatter(*np.array(offsets).T, marker="o", label="Re-simulations", alpha=0.3, edgecolors="none", s=2)
        ax.scatter(0, 0, marker="X", label="Best Fit", edgecolors="k", linewidths=0.5)
        ax.scatter(*em_counterpart_offset, marker="*", edgecolors="k", linewidths=0.5, label="EM counterpart")
        ax.set_aspect("equal")
        lim = 4
        ax.set_xlim([-lim, lim])
        ax.set_ylim([-lim, lim])

        # note the event name in top right corner, offset down and to the left
        ax.annotate(ic_event_name.get(event_name, event_name) + "\n" + em_counterpart[event_name][0], (0, 1),
                    xycoords="axes fraction", xytext=(2, -2), textcoords='offset points', ha="left", va="top")

    axss[0][0].legend(bbox_to_anchor=(1, 1.05), loc="lower center", borderaxespad=0.0, frameon=False, ncol=3)
    fig.supylabel(r"$\Delta$Dec [deg]", x=0)
    fig.supxlabel(r"$\Delta$RA [deg]", y=0)

    return fig
