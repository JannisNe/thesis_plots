import logging
import matplotlib.pyplot as plt
from kcorrect.kcorrect import Kcorrect
from astropy.modeling.physical_models import BlackBody
from astropy import units as u
from astropy import constants as const
from timewise.wise_data_base import WISEDataBase
import numpy as np
import pandas as pd

from thesis_plots.plotter import Plotter
from thesis_plots.cache import DiskCache


logger = logging.getLogger(__name__)


@DiskCache.cache
def kcorrection_data(temp: float, fit_wave=(1e4, 1e5)):
    k = Kcorrect(responses=["sdss_g0", "twomass_Ks", "wise_w1", "wise_w2"])
    wavelengths = k.templates.restframe_wave
    fluxes = [pd.Series(f / max(f)).rolling(100).median() for f in k.templates.restframe_flux]
    bb = BlackBody(temperature=temp * u.K)
    bb_flux = (bb(wavelengths * u.AA) * const.c / (wavelengths * u.AA)**2).to("erg s-1 cm-2 AA-1 sr-1")
    bb_flux = bb_flux / max(bb_flux)

    # fit_wave = (1e4, 1e5)
    fit_mask = (wavelengths > fit_wave[0]) & (wavelengths < fit_wave[1])
    fit_wavelengths = np.log10(wavelengths[fit_mask])
    fits = [np.polyfit(fit_wavelengths, np.log10(f[fit_mask]), 1) for f in fluxes]
    return wavelengths, fluxes, bb_flux, fits, fit_wave


@Plotter.register("margin")
def kcorrection():
    logger.info("plotting k-correction")

    temp = 2000
    wavelengths, fluxes, bb_flux, fits, fit_wave = kcorrection_data(temp=temp)
    logger.info(f"fitted spectral indices: {[fi[0] for fi in fits]}")

    def wavelengthfromz(z):
        w = WISEDataBase.band_wavelengths["W1"].to("AA").value / (1 + z)
        logger.debug(f"wavelength from z {z} is {w}")
        return w

    fig, ax = plt.subplots()
    ax.plot(wavelengths, bb_flux, label=f"Blackbody, {temp} K", zorder=10)
    for i, (f, fit) in enumerate(zip(fluxes, fits)):
        ax.plot(wavelengths, f, label=f"galaxy templates" if i == 0 else None, color="dimgrey", zorder=2, alpha=0.7)
        ax.plot(fit_wave, 10**np.polyval(fit, np.log10(fit_wave)), label=f"PL fits" if i == 0 else None, ls="--",
                color="dimgrey", alpha=1, zorder=5)
    ax.axvline(WISEDataBase.band_wavelengths["W1"].to("AA").value, color="C3", label="WISE bands", ls=":")
    ax.axvline(WISEDataBase.band_wavelengths["W2"].to("AA").value, color="C3", ls=":")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim((3e3, 1e5))
    ax.set_ylim((1e-6, 1e1))
    ax.set_xlabel("Wavelength [AA]")
    ax.set_ylabel("Flux [a.u.]")
    zax = ax.twiny()
    zax.set_xscale("log")
    zax.set_xlim(ax.get_xlim())
    zs = np.array([0, 1, 2, 3, 4, 5])
    zax.set_xticks([wavelengthfromz(z) for z in zs])
    zax.set_xticks([wavelengthfromz(z) for z in zs + 0.5], minor=True)
    zax.set_xticklabels([f"{z:.0f}" for z in zs])
    zax.set_xticklabels([], minor=True)
    zax.set_xlabel("Redshift")
    ax.legend(ncols=1, loc="lower center", bbox_to_anchor=(0.5, 1.4))

    for a in [ax, zax]:
        a.tick_params(bottom=plt.rcParams["xtick.bottom"], top=plt.rcParams["xtick.top"],
                      left=plt.rcParams["ytick.left"], right=plt.rcParams["ytick.right"], which="both")

    return fig
