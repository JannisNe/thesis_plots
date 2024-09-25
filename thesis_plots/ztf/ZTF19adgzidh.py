import logging
from astropy.io import fits
import matplotlib.pyplot as plt
import pandas as pd
from thesis_plots.plotter import Plotter
from thesis_plots.ztf.data import data_dir, load_lines, band_colors


logger = logging.getLogger(__name__)


z = .4760
neutrino_time_str = "2022-06-24 16:13:16.41"
neutrino_time = pd.to_datetime(neutrino_time_str)
neutrino_name = "IC220624A"


@Plotter.register("half")
def spectrum():
    data_file = data_dir / "ZTF19adgzidh_coadded1d_spec.fits"
    logger.debug(f"loading data from {data_file}")
    with fits.open(data_file) as h:
        coadd_sepc = h[1].data

    spectral_lines = load_lines()

    flux = coadd_sepc["flux"]
    wave = coadd_sepc["wave_grid_mid"]

    fig, ax = plt.subplots()
    ax.set_xlabel(r'$\lambda$ [$\AA$]')
    ax.set_ylabel(r"$F_\lambda$ [$10^{-17}$ erg s$^{-1}$ cm$^{-2}$ $\AA ^{-1}$]")
    ax.plot(wave, flux, lw=0.5, color="C0", zorder=10)

    for line, label in zip(spectral_lines["H"][:3], [r"$\alpha$", r"$\beta$", r"$\gamma$"]):
        x = line * (1 + z)
        ax.axvline(x, lw=.5, color="C1", zorder=2, ls="--")
        _label = "H" + label
        logger.debug(f"line {_label} at {x}")
        ax.annotate(_label, (x, 40), color="C1", ha="right", va="top", textcoords="offset points", xytext=(-2, -2))

    for i, line in enumerate(spectral_lines["OIII"][:2]):
        x = line * (1 + z)
        logger.debug(f"line [OIII] at {x}")
        ax.axvline(x, lw=.5, color="C2", zorder=2, ls="--")
        if i == 0:
            ax.annotate(r"[OIII]", (x, 40), color="C2", ha="left", va="top", textcoords="offset points", xytext=(2, -2))

    mgx = spectral_lines["MgII"][0] * (1 + z)
    logger.debug(f"line MgII at {mgx}")
    ax.axvline(mgx, lw=.5, color="C3", zorder=2, ls="--")
    ax.annotate(r"MgII", (mgx, 40), color="C3", ha="left", va="top", textcoords="offset points", xytext=(2, -2))

    ax.set_xlim(3500, 9800)
    ax.set_ylim(0, 40)
    return fig


@Plotter.register("half")
def lightcurve():
    file = data_dir / "ZTF19adgzidh_photometry.csv"
    logger.debug(f"loading data from {file}")
    data = pd.read_csv(file, parse_dates=["created_at", "UTC"])
    ul_mask = data.mag.isna()
    band_masks = {b: data["filter"] == b for b in ["ztfg", "ztfr"]}

    fig, ax = plt.subplots()
    for band, mask in band_masks.items():
        color = band_colors[band[-1]]
        band_data = data[mask & ~ul_mask]
        if len(band_data) == 0:
            continue
        ax.errorbar(band_data.UTC, band_data.mag, yerr=band_data.magerr, fmt="o", label=band, color=color, ms=2, elinewidth=0.5)
    ax.axvline(neutrino_time, color="C0", ls="--", label=neutrino_name)
    ax.set_xlabel("UTC")
    ax.set_ylabel("magnitude")
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_ha("right")
    ax.invert_yaxis()
    ax.set_xlim(pd.Timestamp("2022-03"), pd.Timestamp("2022-11"))
    ax.legend(bbox_to_anchor=(0.5, 1), loc='lower center', ncol=3)
    return fig
