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


logger = logging.getLogger(__name__)


@Plotter.register("side")
def kcorrection():
    logger.info("plotting k-correction")

    k = Kcorrect(responses=["sdss_g0", "twomass_Ks", "wise_w1", "wise_w2"])
    wavelengths = k.templates.restframe_wave
    fluxes = [pd.Series(f / max(f)).rolling(100).median() for f in k.templates.restframe_flux]
    T = 2000
    bb = BlackBody(temperature=T * u.K)
    bb_flux = (bb(wavelengths * u.AA) * const.c / (wavelengths * u.AA)**2).to("erg s-1 cm-2 AA-1 sr-1")
    bb_flux = bb_flux / max(bb_flux)

    fit_wave = (1e4, 1e5)
    fit_mask = (wavelengths > fit_wave[0]) & (wavelengths < fit_wave[1])
    fit_wavelengths = np.log10(wavelengths[fit_mask])
    fits = [np.polyfit(fit_wavelengths, np.log10(f[fit_mask]), 1) for f in fluxes]
    logger.info(f"fitted spectral indices: {[fi[0] for fi in fits]}")

    fig, ax = plt.subplots()
    ax.plot(wavelengths, bb_flux, label=f"Blackbody, {T} K")
    for i, (f, fit) in enumerate(zip(fluxes, fits)):
        ax.plot(wavelengths, f, label=f"templates" if i == 0 else None, color="black")
        ax.plot(fit_wave, 10**np.polyval(fit, np.log10(fit_wave)), label=f"PL fits" if i == 0 else None, ls="--",
                color="black", alpha=0.7)
    ax.axvline(WISEDataBase.band_wavelengths["W1"].to("AA").value, color="red", label="WISE bands", ls=":")
    ax.axvline(WISEDataBase.band_wavelengths["W2"].to("AA").value, color="red", ls=":")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim((1e3, 1e5))
    ax.set_ylim((1e-6, 1e1))
    ax.set_xlabel("Wavelength [AA]")
    ax.set_ylabel("Flux [a.u.]")
    ax.legend()
    return fig
