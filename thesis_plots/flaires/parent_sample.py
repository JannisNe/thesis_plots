import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import healpy as hp
from astropy.cosmology import Planck18, z_at_value
from astropy import units as u
from scipy import optimize
from thesis_plots.flaires.data.load import load_data
from thesis_plots.plotter import Plotter


logger = logging.getLogger(__name__)


@Plotter.register()
def skymap():
    data = load_data()
    m = data["m"]
    a = data["a"]
    plot_props = {
        "figure_width": plt.rcParams["figure.figsize"][0],
        "figure_size_ratio": plt.rcParams["figure.figsize"][1] / plt.rcParams["figure.figsize"][0],
    }
    hp.projview(
        m / a,
        title="",
        norm="symlog2",
        coord="E",
        cmap="gist_heat",
        graticule=True,
        graticule_labels=True,
        graticule_color="grey",
        override_plot_properties=plot_props,
        max=2000,
        unit="density [deg$^{-2}$]"
    )
    return plt.gcf()


def deltaM_star(lambda1, lambda2, exponent, deltaM_sol):
    return -2.5 * np.log10((lambda1 / lambda2) ** (exponent + 4)) + deltaM_sol


@Plotter.register()
def redshifts():
    data = load_data()["redshifts_parent_sample"]
    bins = data["zbins"]
    width = np.diff(bins)
    hist_full = data["zhist"]
    hists_mstar = data["zhist_mstar"]
    logger.debug(f"M_star = {data['mstar']}")
    bm = (bins[:-1] + bins[1:]) / 2

    def dist(z, norm, exp):
        return norm * (Planck18.luminosity_distance(z).to("Mpc").value / (1+z) / np.sqrt(Planck18.efunc(z))) ** exp

    mstars_bin = {-22: (0, 10), -24: (0, 9), -26: (0, -1), data['mstar'][0]: (0, 10)}
    res_mastar = dict()
    for mstar, hist_mstar in hists_mstar.items():
        popt, pcov = optimize.curve_fit(
            dist,
            bm[mstars_bin[mstar][0]:mstars_bin[mstar][1]],
            hist_mstar[mstars_bin[mstar][0]:mstars_bin[mstar][1]],
            p0=[1e-3, 2]
        )
        logger.debug(f"mstar={mstar}: normalization fit mstar: {popt}, "
                     f"z = {bm[mstars_bin[mstar][0]]:.2f}-{bm[mstars_bin[mstar][1]]:.2f}")
        res_mastar[mstar] = popt

    full_bin = (0, 10)
    popt_full, _ = optimize.curve_fit(dist, bm[full_bin[0]:full_bin[1]], hist_full[full_bin[0]:full_bin[1]], p0=[1e-3, 2])
    logger.debug(f"normalization fit full: {popt_full}, z = {bm[full_bin[1]]}")
    zplot1 = np.linspace(0, .2, 100)


    fig, ax = plt.subplots()
    pfull = ax.plot(zplot1, dist(zplot1, *popt_full), color="C0", ls="-", lw=2)
    hfull = ax.bar(bins[:-1], hist_full, width=width, color="C0", align="edge", label="all", ec="w", alpha=0.5)
    handles = [hfull, pfull[0]]
    ax.bar(bins[:-1], hist_full, width=width, color="none", align="edge", ec="w")
    for ci, mstar in enumerate([-24, -26]):
        c = f"C{ci+1}"
        hist_mstar = hists_mstar[mstar]
        hmstar = ax.bar(bins[:-1], hist_mstar, alpha=.5, width=width, color=c, align="edge", ec="w",
                        label=f"M$_\mathrm{{W1}}$ < {mstar}$")
        ax.bar(bins[:-1], hist_mstar, width=width, color="none", align="edge", ec="w")
        valid_until = bm[mstars_bin[mstar][1]]
        zplot2 = np.linspace(0, valid_until, 100)
        pmstar = ax.plot(zplot2, dist(zplot2, *res_mastar[mstar]), ls="-", lw=2, color=c)
        zplot_ext = np.linspace(valid_until, 0.4, 100)
        ax.plot(zplot_ext, dist(zplot_ext, *res_mastar[mstar]), ls="--", lw=2, color=c)
        handles.extend([hmstar, pmstar[0]])

    ax.set_ylabel("number of objects")
    ax.set_xlabel("redshift")
    ax.set_ylim(0, 2e6)
    ax.legend()
    ax.legend(
        handles,
        [
            "all",
            rf"$N \propto D^{{ {popt_full[1]:.2f} }}$",
            "M$_\mathrm{W1} < -24$",
            rf"$N \propto D^{{ {res_mastar[-24][1]:.2f} }}$",
            "M$_\mathrm{W1} < -26$",
            rf"$N \propto D^{{ {res_mastar[-26][1]:.2f} }}$",
        ],
        ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.05)
    )

    return fig


def m(z):
    # this is wring!
    delta_M = 4
    M = -23.83 + 5 * np.log10(0.7) + delta_M
    mu = 5 * np.log10(Planck18.luminosity_distance(z).to(u.pc).value) - 5
    K = -2.5 * np.log10(1 + z)
    return M + mu + K


def limiting_mag():
    z = np.linspace(1e-2, 0.5, 100)
    mlim = 17.1

    def rootfunc(z):
        return m(z) - mlim

    zlim = optimize.root_scalar(rootfunc, bracket=[0, 0.5]).root
    logger.info(f"limiting magnitude reached at z = {zlim:.2f}")

    fig, ax = plt.subplots()
    ax.plot(z, m(z))
    ax.axhline(mlim, color="k", ls="--", label=f"limiting magnitude: {mlim}")
    ax.axvline(zlim, color="k", ls="--", label=f"redshift: {zlim:.2f}")
    ax.set_xlabel("z")
    ax.set_ylabel("m$_\mathrm{W1}$")

    return fig
