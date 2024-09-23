import logging
from matplotlib import pyplot as plt
import ipdb

from thesis_plots.plotter import Plotter


logger = logging.getLogger(__name__)


@Plotter.register("margin", orientation="portrait")
def piechart():

    labels = ['retracted', 'no uncertainties', 'observed', 'not observable', 'missed']
    sizes = [12, 1, 56, 14, 2]
    explode = [0, 0, 0.05, 0, 0]
    labels_with_numbers = [f"{l} ({n})" if n < 3 else l for l, n in zip(labels, sizes)]
    colors = [f"C{i}" for i in range(len(sizes))]# [6, 8, 5, 1, 7]]

    def make_autopct(values):
        def my_autopct(pct):
            total = sum(values)
            val = int(round(pct*total/100.0))
            return f"{val:.0f}" if val > 3 else ""
        return my_autopct

    fig, ax = plt.subplots()
    _, labels, autotxts = ax.pie(
        sizes,
        labels=labels_with_numbers,
        autopct=make_autopct(sizes),
        shadow=False,
        startangle=100,
        pctdistance=0.75,
        # textprops={'fontsize': big_fontsize},
        colors=colors,
        explode=explode
    )

    for autotext in autotxts:
        autotext.set_color('white')
    for text in labels:
        if text.get_text() != "observed":
            text.set_rotation(90)
            text.set_ha("center")
            text.set_va("bottom")
        else:
            text.set_ha("center")

    ax.axis('equal')

    return fig
