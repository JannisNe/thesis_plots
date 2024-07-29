import logging
import matplotlib.pyplot as plt
import numpy as np
from astropy.modeling import models
from astropy import units as u
from thesis_plots.plotter import Plotter
from thesis_plots.instruments.bandpasses import get_filter


logger = logging.getLogger(__name__)


@Plotter.register("margin")
def wise_blackbody():
    temp = 1600 * u.K
    filters = [("wise", "wise", "W1"), ("wise", "wise", "W2")]
    bb = models.BlackBody(temperature=temp)
    tables = {f"{fac}/{inst} ({band})": get_filter(fac, inst, band) for fac, inst, band in filters}
    wl_range = np.array((1e4, 1e5)) * u.AA
    wls = np.logspace(np.log10(wl_range[0].value), np.log10(wl_range[1].value), 1000) * u.AA
    bb_flux = bb(wls)
    ls = ["--", ":"]

    fig, ax = plt.subplots()
    for ils, (f, table) in zip(ls, tables.items()):
        ax.plot(
            table["Wavelength"],
            table["Transmission"] * bb(table["Wavelength"].to("AA")).to("erg s-1 cm-2 Hz-1 sr-1").value,
            ls=ils,
            label=f.strip(")").replace("wise/wise (", ""),
            zorder=50
        )
    ax.plot(wls, bb_flux.to("erg s-1 cm-2 Hz-1 sr-1").value, label=f"blackbody")
    ax.legend(bbox_to_anchor=(0.5, 1), loc="lower center", ncols=2)
    ax.set(yticklabels=[])
    ax.set_xscale("log")
    ax.set_xlabel("Wavelength [AA]")
    ax.set_ylabel("Flux [a.u.]")

    return fig
