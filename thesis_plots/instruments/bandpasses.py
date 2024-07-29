import logging
import requests
from astropy.table import Table
import io

from thesis_plots.cache import DiskCache


logger = logging.getLogger(__name__)


@DiskCache.cache
def get_filter(facility: str, instrument: str, band: str) -> Table:
    logger.debug(f"getting filter for {band}")
    logger.info(f"downloading {facility}/{instrument}.{band} transmission curve")
    url = f"http://svo2.cab.inta-csic.es/theory/fps/fps.php?ID={facility}/{instrument}.{band}"
    logger.debug(f"downloading {url}")
    r = requests.get(url)
    r.raise_for_status()
    logger.debug(f"code {r.status_code}")
    return Table.read(io.BytesIO(r.content), format="votable")
