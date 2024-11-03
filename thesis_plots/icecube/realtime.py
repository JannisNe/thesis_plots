import logging
import tarfile
import requests
import io
import healpy as hp
import matplotlib.pyplot as plt
import ligo.skymap.plot
from astropy.io import fits
import numpy as np
from astropy.visualization.wcsaxes import SphericalCircle, Quadrangle

from thesis_plots.cache import DiskCache
from thesis_plots.plotter import Plotter


logger = logging.getLogger(__name__)


def get_alert():
    cache = DiskCache()
    key = cache.get_key(get_alert, [], {})
    out_fn = cache.get_cache_file(key)

    if not out_fn.exists():
        url = "https://dataverse.harvard.edu/api/access/datafile/6933695"
        logger.debug(f"Downloading {url}")
        run_id = 133119
        event_id = 22683750
        in_fn = f"Run{run_id}_{event_id}_nside1024.fits.gz"
        # Open a streaming request to the URL
        with requests.get(url, stream=True) as response:
            response.raise_for_status()  # Ensure the request was successful

            # Wrap the streaming content in a file-like object for tarfile
            file_stream = io.BytesIO(response.raw.read())

            # Open the tar file from the stream
            with tarfile.open(fileobj=file_stream, mode='r:*') as tar:
                # Search for the file with the specified filename in the tar archive
                for member in tar:
                    logger.debug(f"Found {member.name}")
                    if member.name == in_fn:
                        logger.debug(f"Extracting {member.name} from tar archive")
                        # Stream the file content to disk
                        with tar.extractfile(member) as file:
                            with open(out_fn, 'wb') as out_file:
                                # Read in chunks to avoid high memory usage
                                for chunk in iter(lambda: file.read(1024 * 1024), b""):
                                    out_file.write(chunk)
                        break
                    raise FileNotFoundError(f"File {in_fn} not found in tar archive")
        logger.debug(f"Saved {out_fn}")

    return out_fn


@Plotter.register("margin")
def example_alert():
    alert_fn = get_alert()
    logger.debug(f"Using alert file {alert_fn}")
    hp_map = hp.read_map(alert_fn, verbose=False)
    with fits.open(alert_fn) as h:
        header = h[1].header
    center_ra = header["RA"]
    center_dec = header["DEC"]
    comment = header["COMMENT"]
    dllh = float(comment.split("(")[-1].strip(")"))
    ra_err_minus = header["RA_ERR_MINUS"]
    ra_err_plus = header["RA_ERR_PLUS"]
    dec_err_minus = header["DEC_ERR_MINUS"]
    dec_err_plus = header["DEC_ERR_PLUS"]
    left_lower_corner_ra = center_ra - ra_err_minus
    left_lower_corner_dec = center_dec - dec_err_minus
    dra = ra_err_plus - ra_err_minus
    ddec = dec_err_plus - dec_err_minus

    pixels_above_threshold = np.where(hp_map > dllh)[0]
    theta, phi = hp.pix2ang(hp.npix2nside(len(hp_map)), pixels_above_threshold)
    lat = 90 - np.degrees(theta)
    lon = np.degrees(phi)

    fig = plt.figure()
    ax = plt.axes(projection="astro degrees zoom", center=f"{center_ra}d {center_dec}d", radius="4 deg")
    _t = ax.get_transform('world')
    ax.plot(lon, lat, 'o', transform=_t)
    gcn_rect = Quadrangle([left_lower_corner_ra, left_lower_corner_dec], dra, ddec,
                          edgecolor="C0", facecolor="none", transform=_t, label="Millipede")
    ax.add_patch(gcn_rect)
    ax.set_xlabel("RA")
    ax.set_ylabel("Dec")
    return fig
