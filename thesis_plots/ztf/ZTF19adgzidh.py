import logging
from astropy.io import fits
from pathlib import Path
from thesis_plots.plotter import Plotter
from thesis_plots.ztf.plotting import spectrum as plot_spec, lightcurve as plot_lc, add_lines


logger = logging.getLogger(__name__)


z = .4760


@Plotter.register()
def spectrum():
    data_file = Path(__file__).parent / "data" / "ZTF19adgzidh_coadded1d_spec.fits"
    logger.debug(f"loading data from {data_file}")
    with fits.open(data_file) as h:
        coadd_sepc = h[1].data
    fig, ax = plot_spec(coadd_sepc, window_width=0)
    add_lines(ax, {"H": 3, "OIII": -1}, redshift=z)
    ax.legend()
    ax.set_xlim(3500, 10000)
    ax.set_ylim(0, 40)
    return fig
