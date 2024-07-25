import logging
import matplotlib.pyplot as plt
import healpy as hp
from thesis_plots.flaires.data.load import load_data
from thesis_plots.plotter import Plotter


logger = logging.getLogger(__name__)


@Plotter.register()
def skymap():
    data = load_data()
    m = data["m"]
    a = data["a"]
    plot_props = {
        "figure_width": plt.rcParams["figure.figsize"][0],
        "figure_size_ratio": plt.rcParams["figure.figsize"][1] / plt.rcParams["figure.figsize"][0],
    }
    hp.projview(
        m / a,
        title="",
        norm="symlog2",
        coord="E",
        cmap="gist_heat",
        graticule=True,
        graticule_labels=True,
        graticule_color="grey",
        override_plot_properties=plot_props,
        max=2000,
        unit="density [deg$^{-2}$]"
    )
    return plt.gcf()
