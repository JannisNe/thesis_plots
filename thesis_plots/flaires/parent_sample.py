import logging
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
from astropy.cosmology import Planck18
from astropy import units as u
from scipy.optimize import root_scalar
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

    zlim = root_scalar(rootfunc, bracket=[0, 0.5]).root
    logger.info(f"limiting magnitude reached at z = {zlim:.2f}")

    fig, ax = plt.subplots()
    ax.plot(z, m(z))
    ax.axhline(mlim, color="k", ls="--", label=f"limiting magnitude: {mlim}")
    ax.axvline(zlim, color="k", ls="--", label=f"redshift: {zlim:.2f}")
    ax.set_xlabel("z")
    ax.set_ylabel("m$_\mathrm{W1}$")

    return fig
