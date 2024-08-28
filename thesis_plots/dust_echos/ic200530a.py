import logging
import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.visualization.wcsaxes import SphericalCircle, Quadrangle
from astropy import units as u
import ligo.skymap.plot

from thesis_plots.plotter import Plotter
from thesis_plots.resimulations.results import ic_gcn_circular_coords, em_counterpart


logger = logging.getLogger(__name__)


ic200530a_src_error = 160.19 / 60  # https://gcn.gsfc.nasa.gov/notices_amon_g_b/134139_35473338.amon
sdssJ1649 = {"ra": 252.4116, "dec": 26.4211, "name": "SDSS\nJ1649+2625"}


@Plotter.register("margin")
def coincidences():
    ic_center = (ic_gcn_circular_coords["tywin"][0][0], ic_gcn_circular_coords["tywin"][1][0])
    ic_center_coord = SkyCoord(*ic_center, unit="deg")
    logger.debug(ic_center_coord)
    ic_gcn_upper = (ic_gcn_circular_coords["tywin"][0][1], ic_gcn_circular_coords["tywin"][1][1])
    ic_gcn_lower = (ic_gcn_circular_coords["tywin"][0][2], ic_gcn_circular_coords["tywin"][1][2])
    ic_rect_corner = np.array(ic_center) + np.array(ic_gcn_lower)
    ic_rect_corner_coords = SkyCoord(*ic_rect_corner, unit="deg")
    dxdy = u.Quantity(np.array(ic_gcn_upper) - np.array(ic_gcn_lower)) * u.deg
    logger.debug(f"corner: {ic_rect_corner_coords}, dxdy: {dxdy}")

    ic_pos = SkyCoord(*ic_center, unit="deg")
    sdss_pos = SkyCoord(sdssJ1649["ra"], sdssJ1649["dec"], unit="deg")
    dist = ic_pos.separation(sdss_pos).arcmin
    logger.info(f"Distance between IC200530A and SDSS141249: {dist:.2f} arcmins")
    tywin_pos = SkyCoord(*em_counterpart["tywin"][1], unit="deg")
    dist = ic_pos.separation(tywin_pos).arcmin
    logger.info(f"Distance between IC200530A and AT2019fdr: {dist:.2f} arcmins")
    sdssJ1649_in_contour = sdssJ1649["ra"] > (ic_center[0] + ic_gcn_lower[0])
    logger.debug([sdssJ1649["ra"], ic_center[0] + ic_gcn_lower[0]])
    logger.info(f"SDSS141249 in the IC200530A contour: {sdssJ1649_in_contour}")

    ax = plt.axes(projection="astro degrees zoom", center=f"{ic_center[0]}d {ic_center[1]}d", radius="4 deg")
    _t = ax.get_transform('world')
    gcn_rect = Quadrangle([ic_rect_corner_coords.ra, ic_rect_corner_coords.dec], dxdy[0], dxdy[1],
                          edgecolor="C0", facecolor="none", transform=_t, label="Millipede")
    gcn_circ = SphericalCircle(ic_center_coord, ic200530a_src_error * u.deg, edgecolor="C0", facecolor="none",
                               transform=_t, ls="--", label="SRC_ERROR")
    ax.scatter(*ic_center, color="C0", s=10, alpha=1, marker="X", edgecolors="none", transform=_t)
    ax.add_patch(gcn_rect)
    ax.add_patch(gcn_circ)
    ax.scatter(sdssJ1649["ra"], sdssJ1649["dec"], color="C1", s=10, alpha=1, marker="s", edgecolors="none",
               transform=_t, label="SDSS J1649+2625")
    ax.scatter(*em_counterpart["tywin"][1], color="C2", s=10, alpha=1, marker="o", edgecolors="none", transform=_t,
               label="AT2019fdr")
    logger.debug(f"ax xticks: {ax.get_xticks()}")
    logger.debug(f"ax xticklabels: {ax.get_xticklabels()}")
    ax.set_xlabel("RA")
    ax.set_ylabel("Dec")
    for i, a in enumerate(ax.coords):
        comp_with = "xtick.bottom" if i == 0 else "ytick.left"
        a.set_ticks_visible(plt.rcParams[comp_with])
        a.set_ticks(spacing=3*u.deg)
    ax.legend(bbox_to_anchor=(0.5, 1.1), loc="lower center", borderaxespad=0.0, ncol=1)
    return plt.gcf()
