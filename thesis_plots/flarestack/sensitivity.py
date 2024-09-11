import logging
from pydoc import visiblename

import matplotlib.pyplot as plt
import pickle
from pathlib import Path
import numpy as np
from scipy.optimize import curve_fit

from thesis_plots.plotter import Plotter


logger = logging.getLogger(__name__)


@Plotter.register("margin")
def example():
    data_file = Path(__file__).parent / "data" / "sens_example.pkl"
    with open(data_file, "rb") as f:
        data = pickle.load(f)
    ts_val = list(data.keys())[0]

    threshold = 0.9

    x, y, yerr = data[ts_val]
    yerr = np.array(yerr) * 3
    x_flux = x * 1e-9
    b = 1 - min(y)

    def f(x, a):
        value = 1 - b * np.exp(-a * x)
        return value

    res = curve_fit(f, x, y, sigma=yerr, absolute_sigma=True, p0=[1.0 / max(x)])
    popt, pcov = res

    perr = np.sqrt(np.diag(pcov))

    best_a = popt[0]

    def best_f(x, sd=0.0):
        a = best_a + perr * sd
        return f(x, a)

    fit = 1e-9 * ((1.0 / best_a) * np.log(b / (1 - threshold)))
    xrange = np.linspace(0.0, 1.1 * max(x), 1000)

    fitc = "C1"
    linec = "grey"

    fig, ax = plt.subplots()
    ax.errorbar(x_flux, y, yerr=yerr, color="black", fmt=" ", marker="o", label="Signal trials")
    ax.plot(xrange * 1e-9, best_f(xrange), color=fitc, label="Best Fit")
    ax.fill_between( xrange * 1e-9, best_f(xrange, 1), best_f(xrange, -1), color=fitc, alpha=0.1)
    ax.axhline(threshold, color=linec, linestyle=":", label="Threshold")
    ax.axvline(fit, color=linec, ls="--", label="Sensitivty")
    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(0.0, max(xrange) * 1e-9)
    ax.set_ylabel(r"$\beta$")
    ax.set_xlabel(r"$\phi$ [a.u.]")
    ax.set_xticks([])
    ax.legend(bbox_to_anchor=(0.5, 1.03), loc="lower center", ncol=1)
    return fig
