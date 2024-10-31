import logging
import abc
import numpy as np
from jedi.inference.value.iterable import Sequence
from mypy.server.objgraph import Iterable
from numpy import typing as npt
import pandas as pd
from pathlib import Path
import os


logger = logging.getLogger(__name__)


class Spectrum(abc.ABC):
    registry = {}

    def __init__(self,
                 best_fit_parameters: Iterable,
                 energy_range: tuple[float, float],
                 contour_file68: str | Path | None = None,
                 contour_file95: str | Path | None = None
                 ):
        self._best_fit_paramaters = best_fit_parameters
        self.energy_range = energy_range
        self.contour_files = {}
        for cl, fn in zip([68, 95], [contour_file68, contour_file95]):
            if fn:
                p = self.get_data_dir() / fn
                assert p.exists(), f"file {p} does not exist"
                self.contour_files[cl] = p
            else:
                logger.debug(f"no contour file for {cl} percent")

    @abc.abstractmethod
    def flux(self, e: npt.NDArray[float], *parameters) -> npt.NDArray[float]:
        pass

    @property
    @abc.abstractmethod
    def paramater_names(self) -> list[str]:
        pass

    @property
    def best_fit(self):
        return {k: v for k, v in zip(self.paramater_names, self._best_fit_paramaters)}

    def contour(self, cl: float) -> pd.DataFrame:
        assert cl in self.contour_files, f"no contour file for {cl} percent"
        return pd.read_csv(self.contour_files[cl], delimiter=";", names=self.paramater_names)

    def upper(self, cl: float, e: npt.NDArray[float]) -> npt.NDArray[float]:
        return np.max(self.flux(e, *self.contour(cl).values.T), axis=0)

    def lower(self, cl: float, e: npt.NDArray[float]) -> npt.NDArray[float]:
        return np.min(self.flux(e, *self.contour(cl).values.T), axis=0)

    def best(self, e: npt.NDArray[float]) -> npt.NDArray[float]:
        return self.flux(e, *self._best_fit_paramaters)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.registry[cls.__name__] = cls

    @classmethod
    def from_dict(cls, data):
        assert "class" in data, "data must contain a class key"
        _class = data.pop("class")
        assert _class in cls.registry, f"unknown class {_class}"
        logger.debug(f"creating class {_class} with data {data}")
        return cls.registry[_class](**data)

    @classmethod
    def get_data_dir(cls):
        env = os.environ.get("ICECUBE_DIFFUSE_RESULTS", None)
        if env:
            return Path(env).expanduser().resolve()
        else:
            return Path(__file__).parent / "data"


class SinglePowerLaw(Spectrum):

    @property
    def paramater_names(self):
        return ["norm", "gamma"]

    def flux(self, e: npt.NDArray[float], *parameters) -> npt.NDArray[float]:
        norm, gamma = parameters
        return norm * (e ** -gamma)
