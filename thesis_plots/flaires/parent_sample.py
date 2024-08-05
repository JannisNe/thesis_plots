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
    hist_mstar = data["zhist_mstar"]
    logger.debug(f"M_star = {data['mstar2']}")
    bm = (bins[:-1] + bins[1:]) / 2
    dl_mids = Planck18.luminosity_distance(bm).to("Mpc").value

    def minfunc(args, hist, max_bin):
        norm, exp = args
        return np.sum((hist[:max_bin] - norm * dl_mids[:max_bin] ** exp) ** 2 / hist[:max_bin])

    res_mstar = optimize.minimize(minfunc, x0=[1e-4, 2], args=(hist_mstar, -2))
    logger.debug(f"normalization fit mstar: {res_mstar.x}")
    res_full = optimize.minimize(minfunc, x0=[1e-4, 2], args=(hist_full, 10))
    logger.debug(f"normalization fit full: {res_full.x}")

    fig, ax = plt.subplots()
    pfull = ax.plot(bm[:10], res_full.x[0] * dl_mids[:10] ** res_full.x[1], color="C0", ls="-", lw=2)
    pmstar = ax.plot(bm, res_mstar.x[0] * dl_mids ** res_mstar.x[1], color="C1", ls="-", lw=2)
    hfull = ax.bar(bins[:-1], hist_full, width=width, color="C0", align="edge", label="all", ec="w", alpha=0.5)
    ax.bar(bins[:-1], hist_full, width=width, color="none", align="edge", ec="w")
    hmstar = ax.bar(bins[:-1], hist_mstar, width=width, color="C1", align="edge", ec="w", alpha=0.5,
           label="M$_\mathrm{W1}$ < M$^\star_\mathrm{W1}$")
    ax.bar(bins[:-1], hist_mstar, width=width, color="none", align="edge", ec="w")
    ax.set_ylabel("number of objects")
    ax.set_xlabel("redshift")
    ax.legend(
        [hfull, pfull[0], hmstar, pmstar[0]],
        [
            "all",
            rf"$N \propto d_\mathrm{{L}}^{{ {res_full.x[1]:.2f} }}$",
            "M$_\mathrm{W1}$ < M$^\star_\mathrm{W1}$",
            rf"$N \propto d_\mathrm{{L}}^{{ {res_mstar.x[1]:.2f} }}$"
        ],
        ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.05)
    )

    return fig


def m(z):
    delta_M = 4
    M = -23.83 + 5 * np.log10(0.7) + delta_M
    mu = 5 * np.log10(Planck18.luminosity_distance(z).to(u.pc).value) - 5
    K = -2.5 * np.log10(1 + z)
    return M + mu + K


@Plotter.register("margin")
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
