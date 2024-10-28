import matplotlib.pyplot as plt
import numpy as np
import math
import logging
from pathlib import Path

from thesis_plots.cache import DiskCache
from thesis_plots.cosmic_rays.extract_crdb import main as download
from thesis_plots.cosmic_rays.sum_all_nuclei import main as sum_all_nuclei


logger = logging.getLogger(__name__)


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

    data_files = {
        'positrons': ['AMS-02_e+_energy.txt',
                      'FERMI_e+_energy.txt',
                      'PAMELA_e+_energy.txt'],
        'antiprotons': ['AMS-02_pbar_energy.txt',
                        'BESS_pbar_energy.txt',
                        'PAMELA_pbar_energy.txt'],
        'leptons': ['AMS-02_e-e+_energy.txt',
                    'CALET_e-e+_energy.txt',
                    'DAMPE_e-e+_energy.txt',
                    'FERMI_e-e+_energy.txt',
                    'HESS_e-e+_energy.txt'],
        'protons': ['AMS-02_H_energy.txt',
                    'BESS_H_energy.txt',
                    'CREAM_H_energy.txt',
                    'CALET_H_energy.txt',
                    'DAMPE_H_energy.txt',
                    'KASCADE_H_energy.txt',
                    'KASCADE-Grande_H_energy.txt',
                    'PAMELA_H_energy.txt'],
        'allParticles': ['AUGER_allParticles_energy.txt',
                         'HAWC_allParticles_energy.txt',
                         'KASCADE_allParticles_energy.txt',
                         'KASCADE-Grande_allParticles_energy.txt',
                         'NUCLEON_allParticles_energy.txt',
                         'ICECUBE+ICETOP_allParticles_energy.txt',
                         'TA_allParticles_energy.txt',
                         'TUNKA-133_allParticles_energy.txt'],
    }
    processed_files = {
        'ams_combined': "AMS-02_allParticles_energy.txt",
        'cream_combined': "CREAM_allParticles_energy.txt",
    }

    def __init__(self):
        logger.info("Initializing TheCrSpectrum")
        self.cache = DiskCache()
        self._check_files()
        self.data_dir = Path(__file__).parent / 'data'
    
    def FigSetup(self, shape='Rectangular'):
        """Sets up the figure based on the shape."""
        fig, ax = plt.subplots()
        self.SetAxes(ax)
        return fig, ax

    def SetAxes(self, ax):
        """Configures the x and y axes for the plot."""
        # ax.minorticks_off()
        ax.set_xscale('log')
        ax.set_xlim([1, 1e12])
        ax.set_xlabel('Energy [GeV]')
        ax.set_yscale('log')
        ax.set_ylim([1e-7, 1e4])
        ax.set_ylabel(r'E$^{2}$ Intensity [GeV m$^{-2}$ s$^{-1}$ sr$^{-1}$]')
        
        # Twin axis for Joules
        ax2 = ax.twiny()
        # ax2.minorticks_off()
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
        s_LHC = 14e3**2.0 # GeV2
        proton_mass = 0.938 # GeV
        E_LHC = 0.5 * s_LHC / proton_mass
        annotations = [
            ('LHC', E_LHC, 1e-7, E_LHC, 1e-6),
            ('1. Knee', 2.8e6, 1., 2.5e5, 1.3e-1),
            ('2. Knee', 8e7, 2e-2, 5e5, 5e-3),
            ('Ankle', 0.4e10, 2e-4, 1e9, 1e-5),
        ]
        for text, x, y, xtext, ytext in annotations:
            ax.annotate(text, xy=(x, y), xytext=(xtext, ytext), horizontalalignment="center",
                        arrowprops=dict(arrowstyle="-|>", color='k', lw=0.5, shrinkA=0, shrinkB=0.5)
                        )
        
        texts = [
            (r'$p$', 7, 0.7e3),
            (r'$\nu + \bar{\nu}$', 0.5e6, 2.5e-4),
            (r'$\gamma$ IGRB', 11, 3e-5),
        ]
        for text, x, y in texts:
            ax.text(x, y, text)

        # Add fill between for the E2dNdEdOmega data
        E = np.logspace(0, 15)
        E2dNdEdOmega = E * 1. / 4. / math.pi # m2/s
        ax.text(0.4e4, 0.8e3, r'1m$^2$/s', color='tab:gray', rotation=36)
        ax.fill_between(E, E2dNdEdOmega, 1e4, alpha=0.12, lw=0, facecolor='tab:gray', edgecolor='tab:gray')

        E2dNdEdOmega = E * 1. / 3.14e7 / 4. / math.pi # m2/yr
        ax.text(4e10, 0.3e3, r'1/m$^2$/yr', color='tab:gray', rotation=36)
        ax.fill_between(E, E2dNdEdOmega, 1e4, alpha=0.12, lw=0, facecolor='tab:gray', edgecolor='tab:gray')

        E2dNdEdOmega = E * 1. / 3.14e7 / 1e6 / 4. / math.pi # km2/yr
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

    def _check_files(self):
        """Check if data files are present, if not download them."""
        for _, files in self.data_files.items():
            for filename in files:
                if not (self.cache.cache_dir / filename).exists():
                    download(cache=self.cache)
                    break
        for _, filename in self.processed_files.items():
            if not (self.cache.cache_dir / filename).exists():
                sum_all_nuclei(cache=self.cache)
                break

    def plot_experiment_data(self, ax, experiment_type):
        """Plot function to handle different types of particles (positrons, antiprotons, etc.)"""

        if experiment_type in self.data_files:
            for filename in self.data_files[experiment_type]:
                self.plot_data(ax, self.cache.cache_dir / filename, 'o', self.colors[filename.split('_')[0]], 1)
        if experiment_type == 'allParticles':
            self.plot_line(ax, self.cache.cache_dir / 'AMS-02_allParticles_energy.txt', self.colors['AMS-02'])
            self.plot_line(ax, self.cache.cache_dir / 'CREAM_allParticles_energy.txt', self.colors['CREAM'])

    def neutrinos(self, ax):
        """Plot neutrino measurements with error bars."""
        filename = self.data_dir / 'IceCube_nus_energy.txt'
        self.plot_data_diffuse(ax, filename, self.colors['ICECUBE'])

    def gammas(self, ax):
        """Plot ... with error bars."""
        filename = self.data_dir / 'FERMI_igrb_energy.txt'
        self.plot_data_diffuse(ax, filename, self.colors['FERMI'])

    def plot_data(self, ax, filename, fmt, color, zorder=1):
        """Plot data with error bars."""
        E, dJdE, errStatLo, errStatUp, errSysLo, errSysUp = np.loadtxt(
            filename, skiprows=8, usecols=(0, 1, 2, 3, 4, 5), unpack=True
        )
        E2 = E * E
        y = E2 * dJdE
        dyLo = E2 * np.sqrt(errStatLo**2 + errSysLo**2)
        dyUp = E2 * np.sqrt(errStatUp**2 + errSysUp**2)

        ind = dyLo < y
        ax.errorbar(E[ind], y[ind], yerr=[dyLo[ind], dyUp[ind]], fmt=fmt, markeredgecolor=color,
                    color=color, elinewidth=1.5, capthick=1.5, zorder=zorder)

        ind_upper = dyLo > y
        ax.errorbar(E[ind_upper], y[ind_upper], yerr=0.25 * y[ind_upper], uplims=True,
                    fmt=fmt, markeredgecolor=color, color=color, elinewidth=1.5, capthick=1.5, zorder=zorder)

    def plot_line(self, ax, filename, color):
        """Plot a line for the all-particle spectrum."""
        E, dJdE = np.loadtxt(filename, usecols=(0, 1), unpack=True)
        ax.plot(E, E**2 * dJdE, color=color)

    def plot_data_diffuse(self, ax, filename, color):
        """Plot diffuse data."""
        x, dxLo, dxUp, y, dyLo, dyUp = np.loadtxt(filename, skiprows=1, usecols=(0, 1, 2, 3, 4, 5), unpack=True)
        ind = dyLo < y
        ax.errorbar(x[ind], y[ind], yerr=[dyLo[ind], dyUp[ind]], xerr=[dxLo[ind], dxUp[ind]],
                    fmt='o', markeredgecolor=color, color=color, elinewidth=1.5, capthick=1.5, mfc='white')
        ind_upper = dyLo > y
        ax.errorbar(x[ind_upper], y[ind_upper], xerr=[dxLo[ind_upper], dxUp[ind_upper]], yerr=0.25 * y[ind_upper], uplims=True,
                    fmt='o', markeredgecolor=color, color=color, elinewidth=1.5, capthick=1.5, mfc='white')
