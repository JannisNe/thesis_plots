import json
from pathlib import Path

from thesis_plots.icecube_diffuse_flux.spectrum import Spectrum


def load_spectrum(name: str) -> Spectrum:
    summary_file = Path(__file__).parent / "measurements.json"
    with summary_file.open("r") as f:
        data = json.load(f)
    assert name in data, f"{name} not in {summary_file}"
    return Spectrum.from_dict(data[name])
