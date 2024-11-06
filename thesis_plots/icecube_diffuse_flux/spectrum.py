import logging
import abc
import numpy as np
from typing import Iterable
from numpy import typing as npt
import pandas as pd
from pathlib import Path
import os


logger = logging.getLogger(__name__)


class Spectrum(abc.ABC):
    registry = {}

    def __init__(
            self,
            best_fit_parameters: Iterable,
            energy_range_gev: tuple[float, float],
            reference_energy_gev: float,
            contour_file68: str | Path | None = None,
            contour_file95: str | Path | None = None,
            csv_kwargs: dict | None = None,
            year: int | None = None,
            journal: str | None = None
    ):
        self._best_fit_parameters = best_fit_parameters
        self.reference_energy_gev = reference_energy_gev
        self.energy_range_gev = energy_range_gev
        self.contour_files = {}
        for cl, fn in zip([68, 95], [contour_file68, contour_file95]):
            self.set_contour_file(cl, fn)
        self.csv_kwargs = csv_kwargs if csv_kwargs else {}
        self.year = year
        self.journal = journal

    def set_contour_file(self, cl: float, fn: str | Path):
        if fn is None:
            logger.debug(f"no contour file for {cl} percent")
            return
        _fn = Path(fn)
        _fn_abs = _fn.expanduser() if _fn.is_absolute() else self.get_data_dir() / _fn
        assert _fn_abs.exists(), f"file {_fn_abs} does not exist"
        self.contour_files[cl] = _fn_abs

    @abc.abstractmethod
    def flux(self, e_gev: npt.NDArray[float], *parameters) -> npt.NDArray[float]:
        pass

    def _broadcast_flux(self, e_gev: npt.NDArray[float] | float, *parameters) -> npt.NDArray[float]:
        _e_gev = np.atleast_1d(e_gev)[..., np.newaxis]
        return self.flux(_e_gev, *parameters).squeeze()

    @property
    @abc.abstractmethod
    def paramater_names(self) -> list[str]:
        pass

    @property
    def best_fit(self):
        return {k: v for k, v in zip(self.paramater_names, self._best_fit_parameters)}

    def contour(self, cl: float) -> pd.DataFrame:
        assert cl in self.contour_files, f"no contour file for {cl} percent"
        df = pd.read_csv(self.contour_files[cl], **self.csv_kwargs)
        return df[self.paramater_names]

    def upper(self, cl: float, e_gev: npt.NDArray[float] | float) -> npt.NDArray[float]:
        f = self._broadcast_flux(e_gev, *self.contour(cl).values.T)
        return np.max(f, axis=f.ndim - 1)

    def lower(self, cl: float, e_gev: npt.NDArray[float] | float) -> npt.NDArray[float]:
        f = self._broadcast_flux(e_gev, *self.contour(cl).values.T)
        return np.min(f, axis=f.ndim - 1)

    def best(self, e_gev: npt.NDArray[float] | float) -> npt.NDArray[float]:
        return self._broadcast_flux(e_gev, *self.best_fit.values())

    def get_energy_range(self, log=True, n=100):
        return np.logspace(*np.log10(self.energy_range_gev), n) if log else np.linspace(*self.energy_range_gev, n)

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

    def plot_cl(self, cl: float, ax, log=True, energy_scaling=0, **kwargs):
        e = self.get_energy_range(log=log)
        scale = e ** energy_scaling
        return ax.fill_between(e, self.lower(cl, e) * scale, self.upper(cl, e) * scale, **kwargs)

    def plot(self, ax, log=True, energy_scaling=0, **kwargs):
        e = self.get_energy_range(log=log)
        scale = e ** energy_scaling
        return ax.plot(e, self.best(e) * scale, **kwargs)


class SinglePowerLaw(Spectrum):

    @property
    def paramater_names(self):
        return ["gamma", "norm"]

    def flux(self, e_gev: npt.NDArray[float], *parameters) -> npt.NDArray[float]:
        gamma, norm = parameters
        return norm * (e_gev / self.reference_energy_gev) ** -gamma


class BrokenPowerLaw(Spectrum):

    @property
    def paramater_names(self):
        return ["gamma1", "gamma2", "log10_break_energy_gev", "norm"]

    def flux(self, e_gev: npt.NDArray[float], *parameters) -> npt.NDArray[float]:
        gamma1, gamma2, log_break_energy, norm = parameters
        break_energy = 10 ** log_break_energy
        normg = gamma1 if break_energy > self.reference_energy_gev else gamma2
        normb = norm * (self.reference_energy_gev / break_energy) ** normg
        pl1 = normb * (e_gev / break_energy) ** -gamma1
        pl2 = normb * (e_gev / break_energy) ** -gamma2
        return np.where(e_gev < break_energy, pl1, pl2)


class LogParabola(Spectrum):

    @property
    def paramater_names(self):
        return ["gamma", "beta", "norm"]

    def flux(self, e_gev: npt.NDArray[float], *parameters) -> npt.NDArray[float]:
        gamma, beta, norm = parameters
        return norm * (e_gev / self.reference_energy_gev) ** (-gamma - beta * np.log10(e_gev / self.reference_energy_gev))
