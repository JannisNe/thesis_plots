import logging
from pathlib import Path
import subprocess
import json
from pooch import retrieve, Pooch

from thesis_plots.plotter import Plotter


logger = logging.getLogger(__name__)


class SCPDownloader:

    checksum_file = Path(__file__).parent / "data" / "resim" / "checksum.json"
    remote_base_path = Path("/data/user/jnecker/tde_neutrinos/resim")
    local_base_path = Path(__file__).parent / "data" / "resim"
    event_simulations_paths = {
        "tywin": [
            "tywin/out_separate_selection2/out7_L2_M=0.80_E=0.20_*",
            "tywin/out_separate_selection2/out3_Select_M=0.80_E=0.20_*",
        ],
        "lancel": [
            "lancel/out_separate_selection_2m_posvar/out7_L2_M=0.60_E=0.20_*",
            "lancel/out_separate_selection_2m_posvar/out3_Select_M=0.60_E=0.20_*",
        ],
        "bran": [
            "bran/out_separate_selection2/out7_L2_M=0.60_E=0.20_*",
            "bran/out_separate_selection2/out3_Select_M=0.60_E=0.20_*",
        ],
        "txs": [
            "txs/out_separate_selection2/out7_L2_M=0.60_E=0.20_*",
            "txs/out_separate_selection2/out3_Select_M=0.60_E=0.20_*",
        ],
    }

    def __init__(self, ssh_name="cobalt08"):
        SCPDownloader.local_base_path.mkdir(exist_ok=True, parents=True)
        self.ssh_name = ssh_name

    def get_checksum(self, filename: str | Path):
        _fn = self.remote_base_path / str(filename).replace(str(self.local_base_path), "")
        with self.checksum_file.open("r") as f:
            checksums = json.load(f)
        return checksums[str(_fn)]

    def get_remote_filenames(self, event: str):
        filenames = []
        for paths in self.event_simulations_paths[event]:
            cmd = f"ssh {self.ssh_name} 'ls {self.remote_base_path / paths}'"
            logger.debug(f"running {cmd}")
            res = subprocess.check_output(cmd, shell=True)
            filenames += res.decode().strip().split("\n")
        logger.debug(f"found {len(filenames)} filenames for {event}")
        return filenames

    def create_checksum(self):
        logger.info("creating checksums")
        checksums = {}
        for event, paths in self.event_simulations_paths.items():
            for path in paths:
                remote_path = self.remote_base_path / path
                logger.info(f"calculating checksum for {remote_path}")
                cmd = f"ssh {self.ssh_name} 'sha256sum {remote_path}'"
                logger.debug(f"running {cmd}")
                _checksums = subprocess.check_output(cmd, shell=True).decode().strip().split("\n")
                logger.debug(f"got {len(_checksums)} checksums")
                for checksum in _checksums:
                    checksum, filename = checksum.split()
                    checksums[filename] = checksum
        with self.checksum_file.open("w") as f:
            json.dump(checksums, f, indent=4)
        logger.info(f"saved checksums to {self.checksum_file}")

    def __call__(self, url: str, output_file: Path, pooch: Pooch | None = None):
        logger.info(f"downloading {url} to {output_file}")
        cmd = f"scp {self.ssh_name}:{url} {output_file}"
        logger.debug(f"running {cmd}")
        res = subprocess.check_output(cmd, shell=True)
        logger.debug(res)


def retrieve_event_files(even_name: str):
    downloader = SCPDownloader()
    filenames = downloader.get_remote_filenames(even_name)
    for filename in filenames:
        output_file = downloader.local_base_path / str(filename).replace(str(downloader.remote_base_path) + "/", "")
        logger.debug(f"checking for {filename} in {output_file}")
        retrieve(
            url=filename,
            known_hash=downloader.get_checksum(filename),
            fname=output_file.name,
            path=output_file.parent,
            downloader=downloader,
            progressbar=True
        )
        logger.info(f"downloaded {filename} to {output_file}")
