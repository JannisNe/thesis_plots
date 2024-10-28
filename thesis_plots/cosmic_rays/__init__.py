from thesis_plots.plotter import Plotter
from thesis_plots.cosmic_rays.PlotFuncs import TheCrSpectrum


# all credit goes to: https://github.com/carmeloevoli/The_CR_Spectrum/tree/master


@Plotter.register("wide")
def spectrum():
    # Initialize the plot class
    plot = TheCrSpectrum()

    # Set up the figure and axes
    fig, ax = plot.FigSetup()
    plot.plot_experiment_data(ax, 'protons')
    plot.plot_experiment_data(ax, 'allParticles')
    plot.gammas(ax)
    plot.neutrinos(ax)
    plot.experiment_legend(ax)
    plot.annotate(ax)

    return fig
