import matplotlib.pyplot as plt
from matplotlib import patches, transforms
import numpy as np
from pathlib import Path
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
    n_points = 50
    x_draw_prime = rng.uniform(*draw_x, n_points)
    y_draw_prime = rng.uniform(*draw_y, n_points)
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
        ax.scatter(centerx, centery, color=f"C{i}", s=10, alpha=1, marker="X", edgecolors="none")
        ax.add_patch(plt.Circle((centerx, centery), r, color=f"C{i}", fill=False, ls=["-", "--"][i], alpha=1))
    lim_factor = 2.5
    ax.set_xlim(-lim_factor * rc, lim_factor * rc)
    ax.set_ylim(-lim_factor * rc, lim_factor * rc)
    ax.set_yticks(ax.get_yticks()[1:])
    ps = 5
    ax.scatter(*draw, alpha=1, s=ps, color="C1", edgecolors="none")
    ax.scatter(*ddraw, color="C1", s=ps, edgecolors="k", lw=0.4)
    ax.scatter(*nraw, alpha=1, s=ps, color="grey", edgecolors="none")
    ax.add_patch(plt.Rectangle((draw_x[0], draw_y[0]), draw_x[1] - draw_x[0], draw_y[1] - draw_y[0],
                               color="grey", alpha=0.3, ls=":", fill=False))
    ax.set_aspect("equal")
    ax.set_xlabel("$x''$ [m]")
    ax.set_ylabel("$y''$ [m]")

    xy_origin = (0., 0.)
    logger.debug(f"xy_origin: {xy_origin}")
    xy_beta = 40
    xy_l = 0.25
    t = (
            transforms.Affine2D().rotate_deg_around(*xy_origin, xy_beta)
            + transforms.Affine2D().translate(*xy_origin)
            + ax.transAxes
    )
    y_arrow = patches.FancyArrowPatch(
        (0, 0), (0, xy_l), arrowstyle="-|>", color="k", lw=1, mutation_scale=5,
        transform=t, shrinkB=0, shrinkA=0, clip_on=False
    )
    x_arrow = patches.FancyArrowPatch(
        (0, 0), (xy_l, 0), arrowstyle="-|>", color="k", lw=1, mutation_scale=5,
        transform=t, shrinkB=0, shrinkA=0, clip_on=False
    )
    ax.add_patch(x_arrow)
    ax.add_patch(y_arrow)
    ax.annotate("$x'$", xy=(xy_l, 0), ha="left", va="center", xycoords=t)
    ax.annotate("$y'$", xy=(0, xy_l), ha="center", va="bottom", xycoords=t)

    arc = patches.Arc((0, 0), xy_l*1.2, xy_l*1.2, theta1=0, theta2=xy_beta, color="k", lw=1, transform=ax.transAxes)
    ax.add_patch(arc)
    t2 = transforms.Affine2D().rotate_deg_around(*xy_origin, xy_beta / 2) + ax.transAxes
    ax.annotate(r"$\gamma$", (xy_l / 3, 0), xycoords=t2, xytext=(0.35, 0), textcoords=ax.transAxes,
                ha="center", va="bottom",
                arrowprops=dict(arrowstyle="-", lw=1, color="k", shrinkA=0, shrinkB=0))

    return fig


@Plotter.register("wide")
def toy_event():
    data_file = Path(__file__).parent / "data" / "toy_event.json"
    logger.debug(f"loading data from {data_file}")
    with open(data_file, "r") as f:
        data = json.load(f)
    geometry = np.array([data["geometry"]["x"], data["geometry"]["y"], data["geometry"]["z"]])
    orig_line = np.array(data["orig_line"])
    elev = 60
    azim = -60
    view_vector = np.array([np.sin(np.radians(elev)) * np.cos(np.radians(azim)),
                            np.sin(np.radians(elev)) * np.sin(np.radians(azim)),
                            np.cos(np.radians(elev))])

    # find geometry points behind and in front of the plane defined by view vector and orig_line
    m = np.dot(geometry.T - orig_line[:, 0], view_vector) > 0
    logger.debug(f"found {m.sum()} points in front of the plane")
    logger.debug(f"found {len(m) - m.sum()} points behind the plane")
    zorders = np.zeros(len(geometry.T))
    zorders[m] = 10

    fig = plt.figure()
    ax = plt.subplot(projection='3d', computed_zorder=False)
    ax.view_init(elev=elev, azim=azim)
    ax.plot(*orig_line, zorder=5)
    for line in data["new_lines"][:20]:
        ax.plot3D(*line, color="C3", alpha=0.2, zorder=5)
    ax.set_xlabel('x [m]', labelpad=-25)
    ax.set_ylabel('y [m]', labelpad=-25)
    ax.set_zlabel('z [m]', labelpad=-25)
    ax.scatter(*geometry[:, m], color="0.5", s=0.5, edgecolors="none", zorder=10)
    ax.scatter3D(*geometry[:, ~m], color="0.5", s=0.5, edgecolors="none", zorder=0)
    ax.set_aspect("equal")
    return fig


@Plotter.register("margin")
def diagram():
    fig, ax = plt.subplots()

    det_corner = np.array([0.25, 0.06])
    det_width = 0.6
    det_height = 0.8
    ax.add_patch(patches.Rectangle(det_corner, det_width, det_height, fill=False, color="grey", ls=":", zorder=1))
    ax.annotate("detector", xy=det_corner + np.array([det_width, det_height]), ha="center", va="bottom", color="grey")

    ax.add_patch(
        patches.FancyArrowPatch((1, 0), (0, 1), arrowstyle="-|>", color="C3", lw=1, mutation_scale=10,
                                zorder=5)
    )
    ax.annotate("resimulated", xy=(0, 1), ha="left", va="bottom", color="C3")

    arc_size = 0.6
    arc_deg = -45
    arc_rad = np.radians(arc_deg)
    ax.add_patch(
        patches.Arc((0.5, 0.5), arc_size, arc_size, theta1=arc_deg, theta2=0, color="k", lw=1, zorder=2)
    )
    alpha_r = 0.2
    alpha_xy = np.array([0.5, 0.5]) + np.array([np.cos(arc_rad/2) * alpha_r, np.sin(arc_rad/2) * alpha_r])
    logger.debug(f"alpha_xy: {alpha_xy}")
    ax.annotate(r"$\alpha$", xy=alpha_xy, ha="center", va="center")

    ax.add_patch(
        patches.FancyArrowPatch((1, 0.5), (0, 0.5), arrowstyle="-|>", color="C0", lw=1, mutation_scale=10
                                , zorder=6, ls="-")
    )
    ax.annotate("original", xy=(0, 0.5), ha="left", va="top", color="C0",
                textcoords="offset points", xytext=(-20, -2))

    ax.annotate(r"$d$", xy=(det_corner[0] + det_width, 0.325), xytext=(0.93, 0.325), ha="left", va="center",
                arrowprops=dict(
                    arrowstyle="-[, widthB=1.17, lengthB=0.", lw=1, color="k", zorder=3, shrinkB=0, shrinkA=0
                ))
    lx = det_corner[0] + det_width/2
    ax.annotate(r"L", xy=(lx, 0.52), xytext=(lx, 0.6), ha="center", va="bottom",
                arrowprops=dict(
                    arrowstyle="-[, widthB=2.1, lengthB=0.1", lw=1, color="k", zorder=3
                )
                )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_aspect("equal")
    return fig


@Plotter.register("margin")
def error():
    alpha = np.radians([0, 2, 5])
    beta = np.linspace(0, 84, 100)
    err = np.cos(alpha[:, np.newaxis]) / np.cos(alpha[:, np.newaxis] + np.radians(beta))

    fig, ax = plt.subplots()
    for ia, ierr, ls in zip(alpha, err, [":", "-", "--"]):
        ax.plot(beta, ierr, label=f"$\\alpha = {np.degrees(ia):.0f}^\\circ$", ls=ls)
    ax.set_yscale("log")
    ax.set_xticks([0, 30, 60, 90])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.legend()
    ax.set_xlabel(r"$\beta$ [deg]")
    ax.set_ylabel(r"$\frac{\cos(\alpha) }{ \cos(\alpha + \beta)}$")
    return fig
