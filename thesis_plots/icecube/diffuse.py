import logging
import matplotlib.pyplot as plt

from thesis_plots.icecube_diffuse_flux import load_spectrum
from thesis_plots.plotter import Plotter


logger = logging.getLogger(__name__)


@Plotter.register()
def all_measurements():
    ls = ["-", ":", "--", "-", "-."]
    fc = ["k", "none", "none", "none", "none"]
    ec = ["none", "C0", "C1", "C2", "C3"]
    alpha = [0.5, 1, 1, 1, 1]

    fig, ax = plt.subplots()
    for i, name in enumerate(["joint15", "nt22", "joint23_spl"]):
        s = load_spectrum(name)
        srange = s.get_energy_range()
        supper = s.upper(68, srange) * srange ** 2
        slower = s.lower(68, srange) * srange ** 2
        label = f"{s.journal} ({s.year})" if name != "joint23_spl" else s.journal
        ax.fill_between(srange, slower, supper, label=label, fc=fc[i], ec=ec[i], zorder=1, lw=2, alpha=alpha[i], ls=ls[i])
    sbpl = load_spectrum("joint23_bpl")
    sbplrange = sbpl.get_energy_range()
    bestbpl = sbpl.best(sbplrange) * sbplrange ** 2
    ax.plot(sbplrange, bestbpl, label="joint23_bpl", color="C3", zorder=2, lw=2, ls="-")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Energy [GeV]")
    ax.set_ylabel(r"$E^2 \Phi^{\nu + \bar{\nu}}_\mathrm{per \; flavor}$ [GeV cm$^{-2}$ s$^{-1}$ sr$^{-1}$]")
    ax.legend()
    return fig
