import logging
from astropy.io import fits
import matplotlib.pyplot as plt
from thesis_plots.plotter import Plotter
from thesis_plots.ztf.data import data_dir, load_lines


logger = logging.getLogger(__name__)


@Plotter.register()
def spectrum():
    data_file = data_dir / "ZTF19aavnpjv_combined.fits"
    logger.debug(f"loading data from {data_file}")
    with fits.open(data_file) as h:
        coadd_sepc = h[1].data
    flux = coadd_sepc["flux"]
    wave = coadd_sepc["wave_grid_mid"]

    fig, ax = plt.subplots()
    ax.set_xlabel(r'$\lambda$ [$\AA$]')
    ax.set_ylabel(r"$F_\lambda$ [$10^{-17}$ erg s$^{-1}$ cm$^{-2}$ $\AA ^{-1}$]")
    ax.plot(wave, flux, lw=0.5, color="C0", zorder=10)
    ax.set_xlim(3500, 9680)
    ax.set_ylim(0, 20)
    return fig
