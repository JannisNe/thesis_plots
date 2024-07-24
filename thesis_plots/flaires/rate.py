import logging
import matplotlib.pyplot as plt
import numpy as np
from thesis_plots.flaires.data.load import load_data
from thesis_plots.plotter import Plotter


logger = logging.getLogger(__name__)


@Plotter.register("upright")
def rate():
    data = load_data()
    rates = data["rates"]
    mag_bins = data["mag_bins"]

    z_bins = np.array([0, 0.05, 0.1, 0.2, 0.3, 0.4])
    n_z_bins = len(z_bins) - 1
    width = np.diff(mag_bins)
    annotation_kwargs = {
        "xy": [0, 1],
        "xycoords": "axes fraction",
        "xytext": (2, -2),
        "textcoords": "offset points",
        "ha": "left",
        "va": "top",
        "fontsize": "small",
    }
    xlabel = r"$M_\mathrm{W1}$"
    xlim = -15, -30

    fig, axs = plt.subplots(nrows=n_z_bins, sharex=True, sharey=True, gridspec_kw={"hspace": 0})
    axs = np.atleast_1d(axs)
    rate_axs = [ax.twinx() for ax in axs]
    mag_bin_mids = (mag_bins[1:] + mag_bins[:-1]) / 2
    for i, (ax, (relative_rates, relative_rates_err, all_hist, flares_hist, superthresh, _, _)) in enumerate(
            zip(axs, rates)):
        ax.bar(mag_bins[:-1], all_hist, width=width, color="C0", alpha=0.5, zorder=1, ec="none", align="edge")
        ax.bar(mag_bins[:-1], flares_hist, width=width, color="forestgreen",
               zorder=2, ec="k", align="edge")
        ax.bar(mag_bins[:-1], flares_hist - superthresh, width=width, color="forestgreen",
               zorder=3, ec="k", align="edge", hatch="///", bottom=superthresh)
        ax.set_yscale("log")
        ax.annotate(f"{z_bins[i]:.2f} < z < {z_bins[i + 1]:.2f}", **annotation_kwargs)
        ax.set_ylim(5e-1, 2e7)

        rate_ax = rate_axs[i]
        if i > 0:
            rate_ax.sharey(rate_axs[0])
        ulm = relative_rates == 0
        rate_ax.errorbar(mag_bin_mids[~ulm], relative_rates[~ulm], yerr=relative_rates_err[~ulm].T,
                         marker="o", color="grey", zorder=1, ls="", capsize=2, ms=4, ecolor="k", barsabove=True)
        rate_ax.plot(mag_bin_mids[ulm], relative_rates_err[ulm, 1], marker="v", color="grey", zorder=1, ls="", ms=2)
        rate_ax.set_yscale("log")

    axs[-1].set_xlabel(xlabel)
    axs[-1].set_xlim(xlim)
    # twin = axs[0].twiny()
    # twin.set_xticks(mass_ticks_in_abs_mag)
    # twin.set_xticklabels(mass_xticklabels)
    # twin.set_xlabel(mass_xlabel)
    # twin.set_xlim(xlim)
    coords1 = axs[0].transAxes.inverted().transform(fig.transFigure.transform((.02, .5)))
    logger.debug(f"coords1: {coords1}")
    axs[0].set_ylabel("number of sources", x=coords1[0], y=coords1[1], ha="center", va="bottom", labelpad=5)
    coords2 = rate_axs[0].transAxes.inverted().transform(fig.transFigure.transform((.98, .5)))
    logger.debug(f"coords2: {coords2}")
    rate_axs[0].set_ylabel(r"Rate [yr$^{-1}$ galaxy$^{-1}$]", x=coords2[0], y=coords2[1], ha="center", va="top",
                           labelpad=5)
    rate_axs[0].set_ylim(2e-7, 2e-4)

    return fig


@Plotter.register()
def evolution():
    data = load_data()
    rates = data["rates"]
    mag_bins = data["mag_bins"]
    mag_bin_mids = (mag_bins[1:] + mag_bins[:-1]) / 2
    total_rates = data["total_rates"]
    total_rates_err = data["total_rates_err"]

    z_bins = np.array([0, 0.05, 0.1, 0.2, 0.3, 0.4])
    z_mids = (z_bins[1:] + z_bins[:-1]) / 2
    z_plot = np.linspace(min(z_mids), max(z_bins), 100)
    # rates per mpc^3 have to account for the comoving volume
    # differential_comoving_volume = (Planck18.differential_comoving_volume(z_plot) * 4*np.pi*u.sr).to("Mpc^3").value
    # differential_comoving_volume_normed = differential_comoving_volume / differential_comoving_volume[0]
    # https://ui.adsabs.harvard.edu/abs/2015ApJ...812...33S/abstract
    eta = -2
    f_z_hui = (
                      (1 + z_plot) ** (0.2 * eta)
                      + ((1 + z_plot) / 1.43) ** (-3.2 * eta)
                      + ((1 + z_plot) / 2.66) ** (-7 * eta)
              ) ** (1 / eta)  # * differential_comoving_volume_normed
    # https://www.annualreviews.org/doi/10.1146/annurev-astro-081811-125615
    f_z_sfr = (1 + z_plot) ** 2.7 / (1 + ((1 + z_plot) / 2.9) ** 5.6)  # * differential_comoving_volume_normed

    # normalize to the first redshift bin
    total_rates_err /= total_rates[0]
    total_rates /= total_rates[0]
    logger.debug(f"total rates: {total_rates}")
    logger.debug(f"total rates err: {total_rates_err}")

    fig, ax = plt.subplots()
    zbin_mids = (z_bins[1:] + z_bins[:-1]) / 2
    marker = ["s", "D", "^", "<", ">", "v", "p", "P", "*", "h", "H", "X", "d", "8"]
    bin_indices = np.where((-30 < mag_bin_mids) & (mag_bin_mids < -20))[0]
    i = 0
    for i_mag_bin in bin_indices:
        bin_mid = mag_bin_mids[i_mag_bin]
        logger.debug(f"plotting {bin_mid}")
        y = np.array([r[5][i_mag_bin] for r in rates])
        if (y[0] == 0) or (np.sum(y > 0) == 1):
            logger.debug(f"skipping {bin_mid} because it has no flares in first redshift bin")
            continue
        yerr = np.array([r[6][i_mag_bin] for r in rates])
        yerr /= y[0]
        y /= y[0]
        ulm = y == 0
        color = f"C{i}"
        label = r"M$_\mathrm{W1}$=" + f"$[${round(mag_bins[i_mag_bin]):.0f}, {round(mag_bins[i_mag_bin + 1]):.0f}$]$"
        ax.errorbar(
            zbin_mids[~ulm], y[~ulm], yerr=yerr[~ulm].T,
            ls="--", marker=marker[i], color=color, zorder=1, label=label,
            capsize=2, capthick=1, ms=4, mew=0.5, lw=1
        )
        ax.plot(zbin_mids[ulm], yerr[ulm, 1], marker=marker[i], mec=color, zorder=1, ls="", ms=2, mfc="none", mew=0.5)
        i += 1

    ax.errorbar(
        zbin_mids, total_rates, yerr=total_rates_err,
        ls="--", marker="o", color="k",
        zorder=2, label="total", capsize=2, capthick=1, ms=4, mew=0.5, lw=1
    )

    ax.plot(z_plot, f_z_hui, color="grey", zorder=2, lw=1, ls="--", marker="")
    ax.annotate(
        "TDE", (z_plot[-1], f_z_hui[-1]), xycoords="data",
        xytext=(-2, -2), textcoords="offset points", ha="right", va="top", fontsize="small", color="grey"
    )
    ax.plot(z_plot, f_z_sfr, color="grey", zorder=2, lw=1, ls=":", marker="")
    ax.annotate(
        "SFR", (z_plot[-1], f_z_sfr[-1]), xycoords="data",
        xytext=(-2, 2), textcoords="offset points", ha="right", va="bottom", fontsize="small", color="grey"
    )
    ax.set_ylim(5e-3, 2e1)
    ax.set_xlim(min(z_mids), max(z_bins))
    ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', ncols=2, mode="expand", borderaxespad=0.)
    ax.set_yscale("log")
    ax.set_xlabel("Redshift")
    ax.set_ylabel(r"Normalized Rate")

    return fig
