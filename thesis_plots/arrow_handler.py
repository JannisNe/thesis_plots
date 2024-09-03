from matplotlib.legend_handler import HandlerPatch
from matplotlib import patches
import logging


logger = logging.getLogger(__name__)


# Define a custom handler to ensure the arrow is correctly scaled in the legend
class HandlerArrow(HandlerPatch):
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        pos_a = (xdescent + 0.5 * width, ydescent + height * 0.9)
        pos_b = (xdescent + 0.5 * width, ydescent + height * 0.1)
        logger.debug(f"Creating arrow from {pos_a} to {pos_b}")
        arrow = patches.FancyArrowPatch(
            pos_a, pos_b,
            arrowstyle=orig_handle.get_arrowstyle(),
            mutation_scale=orig_handle.get_mutation_scale(),
            color=orig_handle.get_facecolor(),
            shrinkA=orig_handle.shrinkA,
            shrinkB=orig_handle.shrinkB,
            alpha=orig_handle.get_alpha(),
        )
        arrow.set_transform(trans)
        return [arrow]
