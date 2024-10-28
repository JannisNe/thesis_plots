import crdb
import logging
import matplotlib.pyplot as plt
import numpy as np
import math
from pathlib import Path

from thesis_plots.plotter import Plotter
from thesis_plots.cache import DiskCache


logger = logging.getLogger(__name__)


def print_column_names(tab):
    """Prints the column names of the table with indices."""
    for icol, col_name in enumerate(tab.dtype.fields):
        logger.info('%2i %s', icol, col_name)


def query_crdb(quantity, energyType, expName, combo_level=0):
    """Queries the CRDB and returns the result table."""
    try:
        return crdb.query(quantity, energy_type=energyType, combo_level=combo_level, energy_convert_level=2,
                          exp_dates=expName)
    except Exception as e:
        logger.error(f"Error querying CRDB: {e}")
        raise RuntimeError(f"Failed to query CRDB for {quantity} and {expName}") from e


def write_data_to_file(filename, header, data):
    """Writes header and data to the specified file."""
    try:
        with open(filename, 'w') as f:
            f.write(header)
            for line in data:
                f.write(line)
        logger.info(f'Data successfully written to {filename}')
    except IOError as e:
        logger.error(f"Failed to write to file {filename}: {e}")
        raise RuntimeError(f"Failed to write to file {filename}") from e


def dump_datafile(quantity, energyType, expName, subExpName, filename, combo_level=0):
    """Dumps data from CRDB into a specified file."""
    logger.info(f'Searching for {quantity} as a function of {energyType} measured by {expName}')

    # Query CRDB
    tab = query_crdb(quantity, energyType, expName, combo_level)
    if tab is None:
        return

    subExpNames = set(tab["sub_exp"])
    logger.info('Number of datasets found: %d', len(subExpNames))
    logger.info('Sub-experiments: %s', subExpNames)

    adsCodes = set(tab["ads"])
    logger.info('ADS codes: %s', adsCodes)

    # Select relevant items
    items = [i for i in range(len(tab["sub_exp"])) if tab["sub_exp"][i] == subExpName]
    logger.info('Number of data points: %d', len(items))

    if not items:
        logger.error(f"No data found for sub-experiment {subExpName}")
        raise ValueError(f"No data found for sub-experiment {subExpName}")

    # Prepare data for file writing
    header = (f'#source: CRDB\n'
              f'#Quantity: {quantity}\n'
              f'#EnergyType: {energyType}\n'
              f'#Experiment: {expName}\n'
              f'#ADS: {tab["ads"][items[0]]}\n'
              f'#E - y - errSta_lo - errSta_up - errSys_lo - errSys_up\n')

    data_lines = []
    for eBin, value, errSta, errSys in zip(tab["e_bin"][items], tab["value"][items], tab["err_sta"][items],
                                           tab["err_sys"][items]):
        eBinMean = np.sqrt(eBin[0] * eBin[1])
        data_lines.append(
            f'{eBinMean:10.5e} {value:10.5e} {errSta[0]:10.5e} {errSta[1]:10.5e} {errSys[0]:10.5e} {errSys[1]:10.5e}\n')

    # Write data to file
    write_data_to_file(filename, header, data_lines)


class TheCrSpectrum:
    """Class for plotting cosmic ray spectrum data."""

    # Experiment colors as a dictionary to avoid multiple attributes
    colors = {
        'KASCADE': 'darkgoldenrod',
        'KASCADE-Grande': 'goldenrod',
        'ICECUBE': 'salmon',
        'ICECUBE+ICETOP': 'c',
        'AMS-02': 'forestgreen',
        'AUGER': 'steelblue',
        'BESS': 'yellowgreen',
        'CALET': 'darkcyan',
        'CREAM': 'r',
        'DAMPE': 'm',
        'FERMI': 'b',
        'HAWC': 'slategray',
        'HESS': 'darkorchid',
        'NUCLEON': 'sienna',
        'PAMELA': 'darkorange',
        'TA': 'crimson',
        'TIBET': 'indianred',
        'TUNKA-133': 'hotpink',
        'VERITAS': 'seagreen',
    }
    colors = dict(reversed(list(colors.items())))

    experiments = [
        # AllParticles
        ('AllParticles', 'ETOT', 'AUGER', 'Auger SD750+SD1500 (2014/01-2018/08)'),
        ('AllParticles', 'ETOT', 'HAWC', 'HAWC (2018-2019) QGSJet-II-04'),
        ('AllParticles', 'ETOT', 'IceCube', 'IceCube+IceTop (2010/06-2013/05) SIBYLL2.1'),
        ('AllParticles', 'ETOT', 'KASCADE-Grande', 'KASCADE-Grande (2003/01-2009/03) QGSJet-II-04'),
        ('AllParticles', 'ETOT', 'KASCADE', 'KASCADE (1996/10-2002/01) SIBYLL 2.1'),
        ('AllParticles', 'ETOT', 'NUCLEON', 'NUCLEON-KLEM (2015/07-2017/06)'),
        ('AllParticles', 'ETOT', 'Telescope', 'Telescope Array Hybrid (2008/01-2015/05)'),
        ('AllParticles', 'ETOT', 'Tunka', 'TUNKA-133 Array (2009/10-2012/04) QGSJet01'),
    ]

    def __init__(self):
        logger.info("Initializing TheCrSpectrum")
        self.wd = Path(__file__).parent
        self.cache = DiskCache()

    def download(self):
        for exp in self.experiments:
            key = self.cache.get_key(dump_datafile, exp, {})
            value = self.cache.get(key)
            if value is None:
                fn = self.cache.get_cache_file(key)
                dump_datafile(*exp, filename=fn)
            dump_datafile(*exp)

    def FigSetup(self):
        """Sets up the figure based on the shape."""
        fig, ax = plt.subplots()
        self.SetAxes(ax)
        return fig, ax

    def SetAxes(self, ax):
        """Configures the x and y axes for the plot."""
        ax.set_xscale('log')
        ax.set_xlim([1, 1e12])
        ax.set_xlabel('Energy [GeV]')
        ax.set_yscale('log')
        ax.set_ylim([1e-7, 1e4])
        ax.set_ylabel(r'E$^{2}$ Intensity [GeV m$^{-2}$ s$^{-1}$ sr$^{-1}$]')

        # Twin axis for Joules
        ax2 = ax.twiny()
        ax2.set_xscale('log')
        ax2.set_xlabel('Energy [J]', color='tab:blue', labelpad=8)
        eV2Joule = 1.60218e-19
        ax2.set_xlim([1e9 * eV2Joule, 1e21 * eV2Joule])
        ax2.set_xticks([1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e0, 1e2])
        ax2.tick_params(axis='x', colors='tab:blue')

        for a in [ax, ax2]:
            a.tick_params(bottom=plt.rcParams["xtick.bottom"], top=plt.rcParams["xtick.top"],
                          left=plt.rcParams["ytick.left"], right=plt.rcParams["ytick.right"], which="both")

    def annotate(self, ax):
        """Annotates specific points on the plot."""
        s_LHC = 14e3 ** 2.0  # GeV2
        proton_mass = 0.938  # GeV
        E_LHC = 0.5 * s_LHC / proton_mass
        annotations = [
            ('LHC', E_LHC, 1e-7, E_LHC, 1e-6),
            ('Knee', 2.8e6, 1., 2.5e5, 1.3e-1),
            ('Ankle', 0.4e10, 2e-4, 1e9, 1e-5),
        ]
        for text, x, y, xtext, ytext in annotations:
            ax.annotate(text, xy=(x, y), xytext=(xtext, ytext), horizontalalignment="center",
                        arrowprops=dict(arrowstyle="-|>", color='tab:gray', lw=0.5, shrinkA=0, shrinkB=0.5)
                        )

        texts = [(r'$p$', 7, 0.7e3), (r'$\nu + \bar{\nu}$', 0.5e6, 2.5e-4), (r'$\gamma$ IGRB', 11, 3e-5)]
        for text, x, y in texts:
            ax.text(x, y, text)

        # Add fill between for the E2dNdEdOmega data
        E = np.logspace(0, 15)
        E2dNdEdOmega = E * 1. / 4. / math.pi  # m2/s
        ax.text(0.4e4, 0.8e3, r'1m$^2$/s', color='tab:gray', rotation=36)
        ax.fill_between(E, E2dNdEdOmega, 1e4, alpha=0.12, lw=0, facecolor='tab:gray', edgecolor='tab:gray')

        E2dNdEdOmega = E * 1. / 3.14e7 / 4. / math.pi  # m2/yr
        ax.text(4e10, 0.3e3, r'1/m$^2$/yr', color='tab:gray', rotation=36)
        ax.fill_between(E, E2dNdEdOmega, 1e4, alpha=0.12, lw=0, facecolor='tab:gray', edgecolor='tab:gray')

        E2dNdEdOmega = E * 1. / 3.14e7 / 1e6 / 4. / math.pi  # km2/yr
        ax.text(2.25e10, 1.6e-4, r'1/km$^2$/yr', color='tab:gray', rotation=36)
        ax.fill_between(E, E2dNdEdOmega, 1e4, alpha=0.12, lw=0, facecolor='tab:gray', edgecolor='tab:gray')

        ax.fill_between(E, E2dNdEdOmega, 1e-10, alpha=0.06, lw=0, facecolor='tab:gray', edgecolor='tab:gray')

    def experiment_legend(self, ax):
        """Adds legend for experiments."""
        for i, (exp, color) in enumerate(self.colors.items()):
            ax.text(*self.pos(i), exp, color=color)

    def pos(self, i):
        """Returns vertical position for experiment labels."""
        first_column = 7
        ypos = 2.5e3 * pow(3, -(i - 7 if i >= first_column else i))
        xpos = 1.e7 if i < first_column else 1.e9
        return xpos, ypos

    def plot_experiment_data(self, ax):
        """Plot function to handle different types of particles (positrons, antiprotons, etc.)"""
        for filename in data_files[experiment_type]:
            self.plot_data(ax, f'{pdir}{filename}', 'o', self.colors[filename.split('_')[0]], 1)
        self.plot_line(ax, f'{pdir}AMS-02_allParticles_energy.txt', self.colors['AMS-02'])
        self.plot_line(ax, f'{pdir}CREAM_allParticles_energy.txt', self.colors['CREAM'])

    def neutrinos(self, ax):
        """Plot neutrino measurements with error bars."""
        filename = self.wd / 'IceCube_nus_energy.txt'
        self.plot_data_diffuse(ax, filename, self.colors['ICECUBE'])

    def gammas(self, ax):
        """Plot ... with error bars."""
        filename = self.wd / 'FERMI_igrb_energy.txt'
        self.plot_data_diffuse(ax, filename, self.colors['FERMI'])

    def plot_data(self, ax, filename, fmt, color, zorder=1):
        """Plot data with error bars."""
        E, dJdE, errStatLo, errStatUp, errSysLo, errSysUp = np.loadtxt(
            filename, skiprows=8, usecols=(0, 1, 2, 3, 4, 5), unpack=True
        )
        E2 = E * E
        y = E2 * dJdE
        dyLo = E2 * np.sqrt(errStatLo ** 2 + errSysLo ** 2)
        dyUp = E2 * np.sqrt(errStatUp ** 2 + errSysUp ** 2)

        ind = dyLo < y
        ax.errorbar(E[ind], y[ind], yerr=[dyLo[ind], dyUp[ind]], fmt=fmt, markeredgecolor=color,
                    color=color, elinewidth=1.5, capthick=1.5, zorder=zorder)

        ind_upper = dyLo > y
        ax.errorbar(E[ind_upper], y[ind_upper], yerr=0.25 * y[ind_upper], uplims=True,
                    fmt=fmt, markeredgecolor=color, color=color, elinewidth=1.5, capthick=1.5, zorder=zorder)

    def plot_line(self, ax, filename, color):
        """Plot a line for the all-particle spectrum."""
        E, dJdE = np.loadtxt(filename, usecols=(0, 1), unpack=True)
        ax.plot(E, E ** 2 * dJdE, color=color)

    def plot_data_diffuse(self, ax, filename, color):
        """Plot diffuse data."""
        x, dxLo, dxUp, y, dyLo, dyUp = np.loadtxt(filename, skiprows=1, usecols=(0, 1, 2, 3, 4, 5), unpack=True)
        ind = dyLo < y
        ax.errorbar(x[ind], y[ind], yerr=[dyLo[ind], dyUp[ind]], xerr=[dxLo[ind], dxUp[ind]],
                    fmt='o', markeredgecolor=color, color=color, elinewidth=1.5, capthick=1.5, mfc='white')
        ind_upper = dyLo > y
        ax.errorbar(x[ind_upper], y[ind_upper], xerr=[dxLo[ind_upper], dxUp[ind_upper]], yerr=0.25 * y[ind_upper],
                    uplims=True,
                    fmt='o', markeredgecolor=color, color=color, elinewidth=1.5, capthick=1.5, mfc='white')


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
