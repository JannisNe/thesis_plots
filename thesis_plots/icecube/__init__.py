from thesis_plots.icecube import spice, steamshovel, diffuse, realtime

import logging
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch

from thesis_plots.plotter import Plotter


logger = logging.getLogger(__name__)


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))

        return np.min(zs)


@Plotter.register("half", arg_loop=["zenith_azimuth", "phi_theta"])
def coordinate_system(view):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Define axis ranges
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)

    # Plot axes
    ms = 4
    ax.add_artist(Arrow3D([-1, 1], [0, 0], [0, 0], mutation_scale=ms, lw=1, arrowstyle="-|>", color='k', shrinkA=0, shrinkB=0))
    ax.add_artist(Arrow3D([0, 0], [-1, 1], [0, 0], mutation_scale=ms, lw=1, arrowstyle="-|>", color='k', shrinkA=0, shrinkB=0))
    ax.add_artist(Arrow3D([0, 0], [0, 0], [-1, 1], mutation_scale=ms, lw=1, arrowstyle="-|>", color='k', shrinkA=0, shrinkB=0))

    ax.text(1.2, 0, -.1, 'X')
    ax.text(0, 1., 0., 'Y')
    ax.text(0.2, 0, 1.1, 'Z')

    # Plot the muon trajectory as a diagonal line
    muon_trajectory = np.array([[1, 1, 1], [-1, -1, -1]])
    # Calculate the direction vector of the muon trajectory
    muon_direction = muon_trajectory[1] - muon_trajectory[0]
    muon_direction = muon_direction / np.linalg.norm(muon_direction)  # Normalize the vector

    # Plot the muon trajectory
    ax.add_artist(Arrow3D(muon_trajectory[:, 0], muon_trajectory[:, 1], muon_trajectory[:, 2], mutation_scale=ms, lw=1, arrowstyle="-|>", color='C1', shrinkA=0))
    ax.text(*(muon_trajectory[0] - .1 * muon_direction), r'$\mu$', color="C1")

    f = 0.8
    lines = -1 * np.array([
            [[f, 0], [f, 0], [f, f]],
            [[f, 0], [f, 0], [0, 0]],
            [[f, 0], [f, f], [0, 0]],
            [[f, f], [f, 0], [0, 0]],
            [[f, f], [f, f], [f, 0]],
        ])

    angle2_coords = np.array([1.1, .3, 0]) * 1.2
    angle1_coords = np.array([-0.3, -0.3, .7]) * 1.2

    # Calculate angles
    if view == 'zenith_azimuth':
        muon_direction *= -1
        lines *= -1
        angle1_coords[0] *= -1
        angle1_coords[1] *= -1
        angle1 = r"$\Psi$"
        angle2 = r"$\alpha$"

    elif view == 'phi_theta':
        angle1 = r"$\theta$"
        angle2 = r"$\phi$"

    # Zenith angle (angle with the Z-axis)
    zenith = np.arccos(muon_direction[2])
    # Azimuth angle (angle with the X-axis in the XY-plane)
    azimuth = np.arctan2(muon_direction[1], muon_direction[0])
    if azimuth < 0:
        azimuth += 2 * np.pi
    logger.debug(f"zenith: {np.degrees(zenith)}, azimuth: {np.degrees(azimuth)}")

    # Plot zenith arc
    sin_zenith_angle = np.linspace(0, zenith, 100)
    z_zenith = f * np.cos(sin_zenith_angle)
    x_zenith = f * np.sin(sin_zenith_angle) * np.sin(azimuth)
    y_zenith = f * np.sin(sin_zenith_angle) * np.cos(azimuth)
    ax.plot(x_zenith, y_zenith, z_zenith, 'C0-', linewidth=1)

    # Plot azimuth arc
    azimuth_angle = np.linspace(0, azimuth, 100)
    x_azimuth = f * np.cos(azimuth_angle)
    y_azimuth = f * np.sin(azimuth_angle)
    z_azimuth = [0] * len(azimuth_angle)
    ax.plot(x_azimuth, y_azimuth, z_azimuth, 'C0-', linewidth=1)

    for xs, ys, zs in lines:
        ax.plot(xs, ys, zs, 'k--', linewidth=1)

    ax.text(*angle2_coords, angle2, color="C0")
    ax.text(*angle1_coords, angle1, color="C0")

    # Hide grid and axes for a clean look
    ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.view_init(azim=10, elev=20)
    ax.axis('off')

    RADIUS = 1.3  # Control this value.
    ax.set_xlim3d(-RADIUS / 2, RADIUS / 2)
    ax.set_zlim3d(-RADIUS / 2, RADIUS / 2)
    ax.set_ylim3d(-RADIUS / 2, RADIUS / 2)

    # Display the figure
    return fig
