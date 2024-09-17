import logging
import numpy as np
import pandas as pd
import requests
from astropy.table import Table
import io
import matplotlib.pyplot as plt

from thesis_plots.cache import DiskCache
from thesis_plots.plotter import Plotter


logger = logging.getLogger(__name__)


@DiskCache.cache
def get_filter(facility: str, instrument: str, band: str) -> Table:
    logger.debug(f"getting filter for {band}")
    logger.info(f"downloading {facility}/{instrument}.{band} transmission curve")
    url = f"http://svo2.cab.inta-csic.es/theory/fps/fps.php?ID={facility}/{instrument}.{band}"
    logger.debug(f"downloading {url}")
    r = requests.get(url)
    r.raise_for_status()
    logger.debug(f"code {r.status_code}")
    return Table.read(io.BytesIO(r.content), format="votable")


@Plotter.register("wide")
def bandpasses():
    bands = pd.DataFrame([
        ("Generic", "Bessell", "V",             "optical",  "C7", ":",                     "ASAS-SN", "V"),
        ("SLOAN",   "SDSS",    "gprime_filter", "optical",  "C2", "--",                    "ASAS-SN", "g"),
        ("Palomar", "ZTF",     "g_fil",         "optical",  "C2", "-",                     "ZTF",     "g"),
        ("Palomar", "ZTF",     "r_fil",         "optical",  "C8", "-.",                    "ZTF",     "r"),
        ("Palomar", "ZTF",     "i_fil",         "optical",  "C3", (0, (3, 1, 1, 1)),       "ZTF",     "i"),
        ("wise",    "wise",    "W1",            "infrared", "C4", "-",                     "WISE",    "W1"),
        ("wise",    "wise",    "W2",            "infrared", "C5", ":",                     "WISE",    "W2"),
        ("wise",    "wise",    "W3",            "infrared", "C6", "--",                    "WISE",    "W3"),
        ("wise",    "wise",    "W4",            "infrared", "C7", "-.",                    "WISE",    "W4")
    ],
        columns=["fac", "inst", "band", "regime", "color", "ls", "label_inst", "label_band"])

    fig, axs = plt.subplots(nrows=2, sharey=True, gridspec_kw={"hspace": .8})

    for ax, regime in zip(axs, ["optical", "infrared"]):
        m = bands.regime == regime
        insts = bands[m].label_inst.unique()
        for j, inst in enumerate(insts):
            inst_m = bands[m].label_inst == inst
            _h = []
            for i, r in bands[m & inst_m].iterrows():
                data = get_filter(r.fac, r.inst, r.band)
                t = data["Transmission"] / max(data["Transmission"])
                l, = ax.plot(data["Wavelength"] / 10, t, color=r.color, ls=r.ls)
                _h.append(l)
            if inst == "ASAS-SN":
                anchor = (0, 1)
                loc = "lower left"
            elif inst == "ZTF":
                anchor = (1, 1)
                loc = "lower right"
            elif inst == "WISE":
                anchor = (0.5, 1)
                loc = "lower center"
            leg = ax.legend(_h, bands[m & inst_m].label_band, title=inst, ncol=len(_h), loc=loc, bbox_to_anchor=anchor)
            fig.add_artist(leg)

        ax.set_xscale("log")
        ax.set_ylim(bottom=0)

    xticks_opt = np.array([4, 5, 6, 7, 8, 9]) * 1e2
    axs[0].set_xticks(xticks_opt, minor=True)
    axs[0].set_xticklabels([f"{i:.0f}" for i in xticks_opt], minor=True)

    xticks_ir = np.array([3, 5, 10, 30]) * 1e3
    axs[1].set_xticks(xticks_ir, minor=True)
    axs[1].set_xticks([], minor=False)
    axs[1].set_xticklabels([f"{i:.0f}" for i in xticks_ir], minor=True)

    # axs[1].set_yticks(np.array([4, 5, 6, 7, 8]) * 1e3)

    axs[-1].set_xlabel("Wavelength [nm]")
    fig.supylabel("relative Transmission")

    return fig
