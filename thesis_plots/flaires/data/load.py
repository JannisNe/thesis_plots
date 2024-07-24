import logging
import pickle
from pathlib import Path


logger = logging.getLogger(__name__)


def load_data() -> dict:
    datafile = Path(__file__).parent / "plot_data.pkl"
    logger.debug(f"loading data from {datafile}")
    with datafile.open("rb") as f:
        data = pickle.load(f)
    return data
