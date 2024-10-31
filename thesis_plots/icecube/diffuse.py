import logging
import matplotlib.pyplot as plt

from thesis_plots.icecube_diffuse_flux import load_spectrum
from thesis_plots.plotter import Plotter


logger = logging.getLogger(__name__)


@Plotter.register("wide")
def all_measurements():
    fig, ax = plt.subplots()
    for name in ["joint15", "joint23_spl", "nt22", "hese20"]:
        s = load_spectrum(name)
        srange = s.get_energy_range()
        supper = s.upper(68, srange) * srange ** 2
        slower = s.lower(68, srange) * srange ** 2
        label = f"{s.journal} ({s.year})"
        ax.fill_between(srange, slower, supper, alpha=0.3, label=label, fc="none")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Energy [GeV]")
    ax.set_ylabel(r"$E^2 \frac{dN}{dE}$ [GeV cm$^{-2}$ s$^{-1}$ sr$^{-1}$]")
    ax.legend()
    return fig
