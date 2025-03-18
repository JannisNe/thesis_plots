from thesis_plots.icecube_diffuse_flux.spectrum import Spectrum


def load_spectrum(name: str) -> Spectrum:
    return Spectrum.from_key(name)
