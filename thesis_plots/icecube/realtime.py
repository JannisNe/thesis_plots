import logging
import tarfile
import requests
import io
import healpy as hp
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import ligo.skymap.plot
from astropy.io import fits
from astropy.visualization.wcsaxes import Quadrangle
from astropy import units as u

from thesis_plots.cache import DiskCache
from thesis_plots.plotter import Plotter


logger = logging.getLogger(__name__)


icecat_urls = {
    2019: "https://dataverse.harvard.edu/api/access/datafile/6933695",
    2020: "https://dataverse.harvard.edu/api/access/datafile/6933687"
}


def get_alert(year, run_id, event_id):
    cache = DiskCache()
    key = cache.get_key(get_alert, [year, run_id, event_id], {})
    out_fn = cache.get_cache_file(key)

    if not out_fn.exists():
        url = icecat_urls[year]
        logger.debug(f"Downloading {url}")
        in_fn = f"Run{run_id}_{event_id}_nside1024.fits.gz"
        # Open a streaming request to the URL
        with requests.get(url, stream=True) as response:
            response.raise_for_status()  # Ensure the request was successful

            # Wrap the streaming content in a file-like object for tarfile
            file_stream = io.BytesIO(response.raw.read())

            # Open the tar file from the stream
            found = False
            with tarfile.open(fileobj=file_stream, mode='r:*') as tar:
                # Search for the file with the specified filename in the tar archive
                for member in tar:
                    if member.name == in_fn:
                        logger.debug(f"Extracting {member.name} from tar archive")
                        # Stream the file content to disk
                        with tar.extractfile(member) as file:
                            with open(out_fn, 'wb') as out_file:
                                # Read in chunks to avoid high memory usage
                                for chunk in iter(lambda: file.read(1024 * 1024), b""):
                                    out_file.write(chunk)
                        found = True
                        break
            if not found:
                raise FileNotFoundError(f"File {in_fn} not found in tar archive")
        logger.debug(f"Saved {out_fn}")

    logger.debug(f"Using alert file {out_fn}")
    hp_map = hp.read_map(out_fn)
    with fits.open(out_fn) as h:
        header = h[1].header

    return hp_map, header


@Plotter.register("margin")
def example_alert():
    hp_map, header = get_alert(2019, 133119, 22683750)
    center_ra = header["RA"] * u.deg
    center_dec = header["DEC"] * u.deg
    comment = header["COMMENTS"]
    dllh = float(comment.split("(")[-1].strip(")"))
    ra_err_minus = header["RA_ERR_MINUS"] * u.deg
    ra_err_plus = header["RA_ERR_PLUS"] * u.deg
    dec_err_minus = header["DEC_ERR_MINUS"] * u.deg
    dec_err_plus = header["DEC_ERR_PLUS"] * u.deg
    left_lower_corner_ra = center_ra - ra_err_minus
    left_lower_corner_dec = center_dec - dec_err_minus
    dra = ra_err_plus + ra_err_minus
    ddec = dec_err_plus + dec_err_minus

    fig = plt.figure()
    ax = plt.axes(projection="astro degrees zoom", center=f"{center_ra.value + 2.5}d {center_dec.value}d", radius="6 deg")
    _t = ax.get_transform('world')
    ax.contour_hpx(hp_map, levels=[dllh], colors="C0")
    gcn_rect = Quadrangle([left_lower_corner_ra, left_lower_corner_dec], dra, ddec,
                          edgecolor="C1", facecolor="none", transform=_t, ls="--")
    ax.add_patch(gcn_rect)
    ax.set_xlabel("RA")
    ax.set_ylabel("Dec")
    handles = [
        Line2D([0], [0], color="C0", lw=1, label="Contour"),
        Line2D([0], [0], color="C1", lw=1, ls="--", label="Bounding rectangle"),
    ]
    ax.legend(handles=handles, loc="lower center", ncol=1, bbox_to_anchor=(0.5, 1.01))
    for i, a in enumerate(ax.coords):
        comp_with = "xtick.bottom" if i == 0 else "ytick.left"
        a.set_ticks_visible(plt.rcParams[comp_with])
        a.set_ticks(spacing=4*u.deg)
    return fig
