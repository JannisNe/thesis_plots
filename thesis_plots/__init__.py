import logging
from thesis_plots.flaires.kcorrection import kcorrection

formatter = logging.Formatter('%(levelname)s:%(name)s - %(asctime)s - %(message)s', "%H:%M:%S")
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logging.getLogger("thesis_plots").addHandler(handler)
