import logging
from astropy.io import fits
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from thesis_plots.plotter import Plotter
from thesis_plots.ztf.data import data_dir, load_lines, band_colors


logger = logging.getLogger(__name__)


z = 0.1503
neutrino_time_str = "2023-10-04 14:39:41.18"
neutrino_time = datetime.strptime(neutrino_time_str, "%Y-%m-%d %H:%M:%S.%f")
neutrino_name = "IC231004A"


@Plotter.register("half")
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


@Plotter.register("half")
def lightcurve():
    file = data_dir / "ZTF23abidzvf_photometry.csv"
    logger.debug(f"loading data from {file}")
    data = pd.read_csv(file, parse_dates=["created_at", "UTC"])
    ul_mask = data.mag.isna()
    band_masks = {b: data["filter"] == b for b in ["ztfg", "ztfi", "ztfr"]}

    fig, ax = plt.subplots()
    for band, mask in band_masks.items():
        color = band_colors[band[-1]]
        band_data = data[mask & ~ul_mask]
        if len(band_data) == 0:
            continue
        ax.errorbar(band_data.UTC, band_data.mag, yerr=band_data.magerr, fmt="o", label=band, color=color, ms=2, elinewidth=0.5)
    ax.axvline(neutrino_time, color="C0", ls="--", label=neutrino_name)
    ax.set_xlabel("Days in September 2023")
    ax.set_ylabel("magnitude")
    ax.xaxis.set_major_formatter(DateFormatter("%d"))
    ax.invert_yaxis()
    ax.legend(bbox_to_anchor=(0.5, 1), loc='lower center', ncol=3)
    return fig
