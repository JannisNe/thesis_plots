import logging
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u


logger = logging.getLogger(__name__)


spectral_lines = {
    "H": [656.279, 486.135, 434.0472, 410.1734, 397.0075, 388.9064, 383.5397],
    "OIII": [500.7, 495.9, 436.3],
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
