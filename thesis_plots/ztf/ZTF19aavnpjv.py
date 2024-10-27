import logging
from astropy.io import fits
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from thesis_plots.plotter import Plotter
from thesis_plots.ztf.data import data_dir, band_colors, load_lines


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
    lines = load_lines()

    fig, ax = plt.subplots()
    ax.set_xlabel(r'$\lambda$ [$\AA$]')
    ax.set_ylabel(r"$F_\lambda$ [$10^{-17}$ erg s$^{-1}$ cm$^{-2}$ $\AA ^{-1}$]")
    ax.plot(wave, flux, lw=0.5, color="C0", zorder=10)
    for i, line in enumerate(lines["H"][:2]):
        greek = [r"$\alpha$", r"$\beta$"][i]
        ax.axvline(line, color="C1", ls="--", lw=0.5)
        ax.annotate(f"H{greek}", (line, 20), color="C1", ha="right", va="top", textcoords="offset points", xytext=(-2, -2))
    ax.axvline(lines["He"][4], color="C2", ls="--", lw=0.5)
    ax.annotate("HeI", (lines["He"][4], 20), color="C2", ha="right", va="top", textcoords="offset points", xytext=(-2, -2))
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
        ax.errorbar(band_data.UTC, band_data.mag, yerr=band_data.magerr, fmt="o", label=band[-1], color=color, ms=2, elinewidth=0.5)
    ax.axvline(neutrino_time, color="C0", ls="--", label=neutrino_name)
    ax.set_xlabel("Days in April 2022")
    ax.set_xlim(right=pd.Timestamp("2022-04-30"))
    ax.set_ylabel("magnitude")
    ax.xaxis.set_major_formatter(DateFormatter("%d"))
    ax.set_xlim(pd.Timestamp("2022-04-01"), pd.Timestamp("2022-04-30"))
    ax.invert_yaxis()
    ax.legend(bbox_to_anchor=(0.5, 1), loc='lower center', ncol=3)
    return fig
