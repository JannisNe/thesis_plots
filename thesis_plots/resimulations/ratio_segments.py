from matplotlib import ticker
import matplotlib.pyplot as plt
import numpy as np


def ratio_plot(
    Esim_trunc: list[list[float]],
    Emeas_ev: list[float],
    Eratio: list[float],
    event_name: str,
    alpha: float = 0.3,
    formatter: ticker.Formatter | None = None
) -> plt.Figure:

    fig, axs = plt.subplots(nrows=2, gridspec_kw={"height_ratios": [3, 1], "hspace": 0}, sharex="all")

    for i, (iE, iEratio) in enumerate(zip(Esim_trunc, Eratio)):
        marker = ""
        label = "Re-simulations" if i == 0 else ""
        axs[0].plot(iE, marker=marker, color="C0", alpha=alpha, label=label)
        axs[1].plot(iEratio, marker=marker, color="C0", alpha=alpha)

    axs[0].plot(Emeas_ev, color="k", label=event_name, lw=2)
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
    if formatter:
        axs[1].yaxis.set_major_formatter(formatter)
    axs[0].legend(loc="upper right")

    return fig
