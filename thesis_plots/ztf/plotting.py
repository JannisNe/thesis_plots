import logging
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u


logger = logging.getLogger(__name__)


def spectrum(data, window_width=20):
    flux = data["flux"]
    wave = data["wave_grid_mid"]

    fig, ax = plt.subplots()
    ax.set_xlabel(r'$\lambda$ [$\AA$]')
    ax.set_ylabel(r"$F_\lambda$ [$10^{-17}$ erg s$^{-1}$ cm$^{-2}$ $\AA ^{-1}$]")

    if window_width:
        cumsum_flux = np.cumsum(np.insert(flux, 0, 0))
        flux_smooth = (cumsum_flux[window_width:] - cumsum_flux[:-window_width]) / window_width
        ax.plot(wave, flux, lw=1.5, alpha=0.5, color="C0", zorder=5)
        ax.plot(wave[window_width//2:len(wave)-window_width//2+1], flux_smooth, lw=0.8, color="C3", zotrder=10)
    else:
        ax.plot(wave, flux, lw=0.5, color="C0", zorder=10)

    return fig, ax


spectral_lines = {
    "H": [r"Balmer series", "C4", "--", 656.279, 486.135, 434.0472, 410.1734, 397.0075, 388.9064, 383.5397],
    "OIII": [r"[O III]", "C2", ":", 500.7, 495.9, 436.3],
}


def add_lines(ax, lines, redshift=0, unit="AA"):
    assert all(l in spectral_lines for l in lines)
    for line_set, n_lines in lines.items():
        logger.debug(f"adding lines {line_set}")
        label = spectral_lines[line_set][0]
        color = spectral_lines[line_set][1]
        ls = spectral_lines[line_set][2]
        lines = (u.Quantity(spectral_lines[line_set][3:][:n_lines]) * u.nm * (1 + redshift)).to(unit).value
        for line in lines:
            ax.axvline(line, lw=.5, label=label, color=color, ls=ls, zorder=2)
            label = None


def lightcurve(data):
    pass
