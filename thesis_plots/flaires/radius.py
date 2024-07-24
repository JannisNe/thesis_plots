import logging
import matplotlib.pyplot as plt
from matplotlib import colormaps
import numpy as np
from astropy import constants
from astropy import units as u

from thesis_plots.plotter import Plotter
from thesis_plots.flaires.data.load import load_data


logger = logging.getLogger(__name__)


@Plotter.register()
def validation():
    data = load_data()
    info = data["radius_info"]
    mask = data["radius_mask"]
    types_masks = data["types_masks_radius"]

    x_key = "radius_at_peak"
    xlabel = r"R$_\mathrm{eff, \, peak}$"
    y_key = "intrinsic_equivalent_width_pc"
    yerr_key = "intrinsic_equivalent_width_pc_err"
    ylabel = r"R$_\mathrm{LC}$"
    unit = "pc"
    mask &= ~types_masks["Quasar"] & (info["n_emcee_epochs"] > 2) & (abs(info[y_key] / info[yerr_key]) > 1)
    r_x = info[x_key][mask]
    x_err = np.array(info.loc[mask, ["radius_at_peak_lower", "radius_at_peak_upper"]].abs())
    r_y = info[y_key][mask]
    y_err = info.loc[mask, yerr_key]

    ring_model_factor = 2 / (1 * u.d * constants.c.to("pc s-1")).to("pc").value
    delta_t = [100, 300, 1000]
    factors = [(ring_model_factor / idt, 2) for idt in delta_t]
    model_lineystyle = ":"
    model_names = [f"<{idt:.0f}d" for idt in delta_t]
    model_names += [f">{delta_t[-1]:.0f}d"]
    colors = [colormaps["summer"](cv) for cv in np.linspace(0, 1, len(factors) + 1)]

    fig, ax = plt.subplots()

    markers, caps, bars = ax.errorbar(
        r_x, r_y,
        xerr=x_err.T, yerr=y_err,
        ls="", marker="o", c="C0", mec="k", ms=2, ecolor="k", elinewidth=0.2, capsize=0, mew=0.2)
    for obj in list(caps) + list(bars):
        obj.set_alpha(0.5)

    upper_ax_lim = 0.7
    x = np.linspace(0, upper_ax_lim, 100)
    annot_kwargs = {
        "xytext": (0, -2),
        "textcoords": "offset points",
        "va": "top", "ha": "center",
        "color": "grey",
        "fontsize": "small"
    }
    xprev = 0
    for i in range(len(delta_t)):
        a, b = factors[i]
        logger.debug(f"plotting {a} * x ** {b}")
        y = a * x ** b
        ax.plot(x, y, ls=model_lineystyle, color="grey")
        yprev = [upper_ax_lim] * len(y) if i == 0 else factors[i-1][0] * x ** factors[i-1][1]
        ax.fill_between(x, yprev, y, color=colors[i], alpha=0.5)
        x_annotate = (upper_ax_lim / a) ** (1/b)
        ax.annotate(model_names[i], ((x_annotate + xprev)/2, upper_ax_lim), **annot_kwargs)
        xprev = x_annotate
    ax.fill_between(x, y, color=colors[-1], alpha=0.5)
    ax.annotate(model_names[-1], ((upper_ax_lim + xprev)/2, upper_ax_lim), **annot_kwargs)

    ax.set_xlim(0, upper_ax_lim)
    ax.set_ylim(0, upper_ax_lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(xlabel + f" [{unit}]")
    ax.set_ylabel(ylabel + f" [{unit}]")

    return fig


@Plotter.register("wide")
def correlations():
    data = load_data()
    blackbody_summary = data["luminosity_summary"]
    types_mask = data["type_masks_lum_fct"]
    selected_summary = blackbody_summary[~types_mask["Quasar"]]

    xkeys = np.array(["energy", "peak_luminosity", "temperature_at_peak"])
    xerr_keys = np.array(["energy_err", "peak_luminosity_sym_err", "temperature_at_peak_sym_err"])
    xlabels = np.array([r"E$\mathrm{bol}$ [erg]", r"L$_\mathrm{bol,\,peak}$ [erg s$^{-1}$]", r"T [K]"])
    correlation_factors = [1 / 2, 1, -2.]

    fig, axs = plt.subplots(
        ncols=len(xkeys),
        figsize=np.array(plt.rcParams["figure.figsize"]) * np.array([2, 1]),
        sharey=True,
        gridspec_kw={"wspace": 0}
    )

    y = selected_summary["intrinsic_equivalent_width_pc"]
    yerr = selected_summary["intrinsic_equivalent_width_pc_err"]
    yerr_rel = yerr / y

    for i in range(3):
        logger.debug(f"plotting radius vs {xkeys[i]}")
        x = selected_summary[xkeys[i]]
        xerr = selected_summary[xerr_keys[i]]
        xerr_rel = xerr / x
        m = (x > 0) & (y > 0) & (xerr_rel < 1) & (yerr_rel < 1)
        logx = np.log10(x[m])
        logy = np.log10(y[m])
        logger.debug(f"nans in logx: {any(np.isnan(logx))}, nans in logy: {any(np.isnan(logy))}")

        a = correlation_factors[i]
        logx_med = np.median(logx)
        logy_med = np.median(logy)
        b = logy_med - a * logx_med

        logger.debug(f"y = {a:.2f}x + {b:.2f}")
        xplot = np.logspace(logx.min(), logx.max(), 100)
        yplot = 10 ** (a * np.log10(xplot) + b)

        ax = axs[i]
        markers, caps, bars = ax.errorbar(
            x[m], y[m], xerr=xerr[m], yerr=yerr[m],
            mec="k", ms=1, ecolor="k", elinewidth=0.2, capsize=0, mew=0.2, zorder=2, ls="", marker="o"
        )
        for obj in list(caps) + list(bars):
            obj.set_alpha(0.5)
        ax.plot(xplot, yplot, ls="--", color="k", lw=2, zorder=1, label=f"y = {a:.2f}x + {b:.2f}", alpha=0.5)
        ax.set_xlim(0.5 * x[m].min(), 2 * x[m].max())
        ax.set_ylim(0.5 * y[m].min(), 2 * y[m].max())
        ax.set_xlabel(xlabels[i])
        ax.set_xscale("log")

    axs[0].set_ylabel(r"R$_\mathrm{LC}$ [pc]")
    axs[0].set_yscale("log")

    return fig
