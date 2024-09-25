import logging
from astropy.io import fits
import pandas as pd
import matplotlib.pyplot as plt
from thesis_plots.plotter import Plotter
from thesis_plots.ztf.data import data_dir, load_lines


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
    offset_lris = 4

    spectral_lines = load_lines()

    fig, ax = plt.subplots()
    ax.set_xlabel(r'$\lambda$ [$\AA$]')
    ax.set_ylabel(r"$F_\lambda$ [$10^{-17}$ erg s$^{-1}$ cm$^{-2}$ $\AA ^{-1}$]")
    ax.plot(wave, flux, lw=.5, alpha=1, color="C0", zorder=10, label="ALFOSC")
    ax.plot(wave_lris, flux_lris * 1e17 + offset_lris, lw=0.5, color="C1", zorder=10, label="LRIS + offset")
    for i, line in enumerate(spectral_lines["He"]):
        ax.axvline(line * (1+z), color="C3", lw=0.5, ls="--", zorder=5, label="He$I$" if i == 0 else None)
    ax.legend(bbox_to_anchor=(0.5, 1), loc='lower center', ncol=3)
    ax.set_xlim(3500, 9680)
    ax.set_ylim(0, 10)
    return fig


@Plotter.register()
def lightcurve():
    file = data_dir / "photometry.csv"
    logger.debug(f"loading data from {file}")
    data = pd.read_csv(file, parse_dates=["created_at", "UTC"])
    ul_mask = data.mag.isna()
    band_masks = {b: data["filter"] == b for b in data["filter"].unique()}

    fig, ax = plt.subplots()
    for band, mask in band_masks.items():
        band_data = data[mask]
        ax.errorbar(band_data.UTC, band_data.mag, yerr=band_data.magerr, fmt="o", label=band)
        ax.scatter(band_data.UTC[ul_mask[mask]], band_data.limiting_mag[ul_mask[mask]], marker="v", color="C1", alpha=0.5, s=10)
    ax.set_xlabel("UTC")
    ax.set_ylabel("magnitude")
    ax.invert_yaxis()
    ax.legend()
    return fig
