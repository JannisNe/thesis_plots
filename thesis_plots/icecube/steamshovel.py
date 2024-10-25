import logging
import matplotlib.pyplot as plt
from matplotlib import cm, colors

from thesis_plots.plotter import Plotter


logger = logging.getLogger(__name__)


@Plotter.register("half")
def colorbar():
    plt.rcParams["figure.figsize"][1] = plt.rcParams["figure.figsize"][0] / 33
    fig = plt.figure()

    # Set up a ScalarMappable with the 'viridis' colormap
    cmap = cm.viridis_r
    norm = colors.Normalize(vmin=0, vmax=1)
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)

    # Create a horizontal colorbar
    cax = fig.add_axes([0.0, 0., 1, 1])
    cbar = fig.colorbar(mappable, cax=cax, orientation='horizontal', extend="max", ticks=[])
    cbar.set_label('time')

    # Make the colorbar pointed at the right end
    cbar.ax.xaxis.set_ticks_position('none')  # Remove any ticks
    cbar.outline.set_edgecolor('none')  # Set outline color for the bar

    return fig
