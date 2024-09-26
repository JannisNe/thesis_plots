import logging
from astropy.io import fits
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from thesis_plots.plotter import Plotter
from thesis_plots.ztf.data import data_dir, band_colors


logger = logging.getLogger(__name__)

neutrino_time_str = "2022-04-05 22:20:03.41"
neutrino_time = pd.to_datetime(neutrino_time_str)
neutrino_name = "IC220405A"


@Plotter.register("half")
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


@Plotter.register("half")
def lightcurve():
    file = data_dir / "ZTF19aavnpjv_photometry.csv"
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
    ax.set_xlabel("Days in April 2022")
    ax.set_xlim(right=pd.Timestamp("2022-04-30"))
    ax.set_ylabel("magnitude")
    ax.xaxis.set_major_formatter(DateFormatter("%d"))
    ax.invert_yaxis()
    ax.legend(bbox_to_anchor=(0.5, 1), loc='lower center', ncol=3)
    return fig
