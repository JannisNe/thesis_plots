import logging
from astropy.io import fits
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from thesis_plots.plotter import Plotter
from thesis_plots.ztf.data import data_dir


logger = logging.getLogger(__name__)


z = 0.1503


@Plotter.register()
def spectrum():
    data_file_alfosc = data_dir / "coadd1d_ZTF23abidzvf_ALFOSC_20231017_20231017.fits"
    logger.debug(f"loading data from {data_file_alfosc}")
    with fits.open(data_file_alfosc) as h:
        coadd_alfosc = h[1].data

    data_file_lris = data_dir / "lris20231021_ZTF23abidzvf_o1.spec"
    columns = ["wavelen", "flux", "sky_flux", "flux_unc", "xpixel", "ypixel", "response", "flag"]
    data_lris = pd.read_csv(data_file_lris, comment='#', delim_whitespace=True, names=columns)
    lris_mask = data_lris.flux.notna()
    data_lris = data_lris[lris_mask]
    flux_lris = data_lris["flux"].values
    wave_lris = data_lris["wavelen"].values

    flux = coadd_alfosc["flux"]
    wave = coadd_alfosc["wave_grid_mid"]
    window_width_alfosc = 20
    cumsum_flux = np.cumsum(np.insert(flux, 0, 0))
    flux_smooth = (cumsum_flux[window_width_alfosc:] - cumsum_flux[:-window_width_alfosc]) / window_width_alfosc
    wave_smooth = wave[window_width_alfosc // 2:len(wave) - window_width_alfosc // 2 + 1]

    window_width_lris = 30
    cumsum_flux_lris = np.cumsum(np.insert(flux_lris, 0, 0))
    flux_smooth_lris = (cumsum_flux_lris[window_width_lris:] - cumsum_flux_lris[:-window_width_lris]) / window_width_lris
    wave_smooth_lris = wave_lris[window_width_lris // 2:len(wave_lris) - window_width_lris // 2 + 1]
    offset_lris = 4

    fig, ax = plt.subplots()
    ax.set_xlabel(r'$\lambda$ [$\AA$]')
    ax.set_ylabel(r"$F_\lambda$ [$10^{-17}$ erg s$^{-1}$ cm$^{-2}$ $\AA ^{-1}$]")
    ax.plot(wave, flux, lw=.5, alpha=0.5, color="C0", zorder=5)
    # ax.plot(wave_smooth, flux_smooth, lw=0.8, color="C3", zorder=10)
    ax.plot(wave_lris, flux_lris * 1e17 + offset_lris, lw=0.5, color="C1", zorder=10)
    # ax.plot(wave_smooth_lris, flux_smooth_lris * 1e17 + offset_lris, lw=0.8, color="C4", zorder=10)
    ax.set_xlim(3500, 9680)
    ax.set_ylim(0, 10)
    return fig
