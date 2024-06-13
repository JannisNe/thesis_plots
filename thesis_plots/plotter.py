import logging
import os
from typing import Any
from pathlib import Path

import numpy as np
from matplotlib import style
import matplotlib.pyplot as plt
from difflib import get_close_matches


formatter = logging.Formatter('%(levelname)s:%(name)s - %(asctime)s - %(message)s', "%H:%M:%S")
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logging.getLogger("thesis_plots").addHandler(handler)
logger = logging.getLogger(__name__)


class Plotter:

    registry = {}

    def __init__(self):
        self.dir = Path(os.environ.get("THESIS_PLOTS", "./thesis_plots")).expanduser().resolve()
        logger.debug(f"using directory {self.dir}")
        self.dir.mkdir(exist_ok=True)

    @classmethod
    def register(cls, style_name: str | list[str] = None, arg_loop: Any | list[Any] | None = None):
        if isinstance(style_name, str):
            style_name = [style_name]
        _add_styles = [f"thesis_plots.styles.{w}" for w in style_name] if style_name else []
        _styles = ["thesis_plots.styles.base"] + _add_styles

        def plot_function_with_style(f):

            fname = f.__module__.replace("thesis_plots.", "") + ":" + f.__name__

            if not arg_loop:
                def wrapper(*args, **kwargs):
                    logger.debug(f"using styles {_styles}")
                    style.use(_styles)
                    return f(*args, **kwargs)

                cls.registry[fname] = wrapper

            else:
                wrappers = []
                for a in np.atleast_1d(arg_loop):
                    def wrapper(*args, **kwargs):
                        logger.debug(f"using styles {_styles}")
                        style.use(_styles)
                        return f(a, *args, **kwargs)

                    cls.registry[f"{fname}_{a}"] = wrapper
                    wrappers.append(wrapper)

            return f

        return plot_function_with_style

    def get_filename(self, name: str):
        return self.dir / (name.replace(":", "_").replace(".", "_") + ".pdf")

    def plot(self, name: str | list[str] | None = None, save: bool = False, show: bool = False):
        all_names = self.registry.keys()
        if isinstance(name, str):
            name = [name]
        if name is None:
            name = self.registry.keys()
        _name = [n2 for n2 in all_names if any([n2.startswith(n1) for n1 in name])]
        logger.info(f"making {len(_name)} plots")
        logger.debug(", ".join(_name))
        for n in _name:
            if n not in self.registry:
                raise KeyError(f"plot {n} not found in registry! {_did_you_mean(n, list(self.registry.keys()))}")
            logger.info(f"plotting {n}")
            fig = self.registry[n]()
            if save:
                filename = self.get_filename(n)
                logger.info(f"saving to {filename}")
                fig.savefig(filename, bbox_inches="tight")
            if show:
                plt.show()
            plt.close()
        logger.info("done")


def _did_you_mean(name: str, available: list[str]) -> str:
    closest = get_close_matches(name, available, n=1, cutoff=0.8)
    if closest:
        return f"Did you mean {closest[0]}?"
    return f"Available are {', '.join(available)}"
