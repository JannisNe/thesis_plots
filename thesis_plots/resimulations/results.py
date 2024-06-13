import logging
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker

from thesis_plots.plotter import Plotter


logger = logging.getLogger(__name__)

events = ["tywin", "lancel", "txs"]

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
        "txs": "txs__data_user_jnecker_tde_neutrinos_resim_txs_out_separate_selection2_M=0.60_E=0.20_OnlineL2_SplineMPE_6_data.npz"
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
