from pathlib import Path
import logging
from thesis_plots.resimulations.utils import SCPDownloader

logger = logging.getLogger("thesis_plots.resimulations.data.create_checksum")

checksum_file = Path(__file__).parent / "checksum.json"


def create_checksum():
    SCPDownloader().create_checksum()


if __name__ == "__main__":
    logging.getLogger("thesis_plots").setLevel(logging.DEBUG)
    create_checksum()
