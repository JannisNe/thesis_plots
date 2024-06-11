import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import time
import logging
import json


from thesis_plots.plotter import Plotter


logger = logging.getLogger(__name__)


@Plotter.register("margin")
def circle_plane():
    r = 30
    rc = 20
    phic = 0 * np.pi

    xc = np.cos(phic) * rc
    yc = np.sin(phic) * rc

    alpha = np.arctan2(yc, xc)
    pos_prime = np.array([np.sqrt(xc ** 2 + yc ** 2), 0])

    D = pos_prime[0]
    x1 = D - r
    x2 = r

    y1 = -np.sqrt(r ** 2 - (D / 2) ** 2)
    y2 = -1 * y1

    rng = np.random.default_rng(seed=97865)
    draw_x = min(x1, x2), max(x1, x2)
    draw_y = min(y1, y2), max(y1, y2)
    x_draw_prime = rng.uniform(*draw_x, 100)
    y_draw_prime = rng.uniform(*draw_y, 100)
    m1 = x_draw_prime ** 2 + y_draw_prime ** 2 < r ** 2
    m2 = (x_draw_prime - D) ** 2 + y_draw_prime ** 2 < r ** 2
    m = m1 & m2
    draw_prime = np.array([x_draw_prime, y_draw_prime]).T[m]
    nraw_prime = np.array([x_draw_prime, y_draw_prime]).T[~m]
    ddraw_prime = rng.choice(draw_prime)

    rot_mat_prime = np.array([
        [np.cos(alpha), -np.sin(alpha)],
        [np.sin(alpha), np.cos(alpha)]
    ])

    draw = np.matmul(rot_mat_prime, draw_prime.T)
    nraw = np.matmul(rot_mat_prime, nraw_prime.T)
    ddraw = np.matmul(rot_mat_prime, ddraw_prime)

    fig, ax = plt.subplots()
    for i, (centerx, centery) in enumerate(zip([0, xc], [0, yc])):
        ax.scatter(centerx, centery, color=f"C{i}", s=4, alpha=1, marker="X", edgecolors="none")
        ax.add_patch(plt.Circle((centerx, centery), r, color=f"C{i}", fill=False, ls=["-", "--"][i], alpha=1))
    lim_factor = 2
    ax.set_xlim(-lim_factor * rc, lim_factor * rc)
    ax.set_ylim(-lim_factor * rc, lim_factor * rc)
    ax.scatter(*draw, alpha=0.7, s=2, color="C1", edgecolors="none")
    ax.scatter(*ddraw, color="C1", s=2, edgecolors="k", lw=0.2)
    ax.scatter(*nraw, alpha=0.7, s=2, color="grey", edgecolors="none")
    ax.add_patch(plt.Rectangle((draw_x[0], draw_y[0]), draw_x[1] - draw_x[0], draw_y[1] - draw_y[0],
                               color="grey", alpha=0.3, ls=":", fill=False))
    ax.set_aspect("equal")

    return fig


@Plotter.register("wide")
def toy_event():

    data_file = Path(__file__).parent / "data" / "toy_event.json"
    logger.debug(f"loading data from {data_file}")
    with open(data_file, "r") as f:
        data = json.load(f)
    geometry = data["geometry"]
    orig_line = data["orig_line"]

    fig = plt.figure()  # (8,6)
    ax = plt.subplot(projection='3d')
    ax.plot(*orig_line, lw=3)
    for line in data["new_lines"]:
        ax.plot(*line, color="r", alpha=0.2)
    ax.set_xlabel('pos.x [m]', fontsize=12, labelpad=-25)
    ax.set_ylabel('pos.y [m]', fontsize=12, labelpad=-25)
    ax.set_zlabel('pos.z [m]', fontsize=12, labelpad=-25)
    ax.scatter(geometry['x'], geometry['y'], geometry['z'], s=0.5, c='0.7', alpha=0.4)
    ax.set_aspect("equal")
    return fig
