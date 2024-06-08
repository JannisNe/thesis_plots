import inspect
import logging
from pathlib import Path
import os
import pickle
from typing import Any, Hashable, get_type_hints


logger = logging.getLogger(__name__)


class DiskCache:

    def __init__(self):
        self.cache_dir = Path(os.environ.get("THESIS_PLOTS_CACHE", "./thesis_plots_cache")).expanduser().resolve()
        logger.debug(f"using cache directory {self.cache_dir}")
        self.cache_dir.mkdir(exist_ok=True)

    @staticmethod
    def get_key(f, args, kwargs):
        logger.debug(f"getting key for {f} with args {args} and kwargs {kwargs}")
        _args = inspect.getfullargspec(f).args
        assert len(args) + len(kwargs) <= len(_args), "wrong number of arguments"
        assert all(a in _args for a in kwargs.keys()), "unknown keyword argument"

        _sorted_args = list(range(len(_args)))
        _found = [False] * len(_args)
        for n, a in enumerate(args):
            _sorted_args[n] = a
            _found[n] = True
        for k, v in kwargs.items():
            _sorted_args[_args.index(k)] = v
            _found[_args.index(k)] = True
        for k, v in inspect.signature(f).parameters.items():
            if not _found[_args.index(k)]:
                _sorted_args[_args.index(k)] = v.default

        logger.debug(f"sorted args are {_sorted_args}")
        key = f"{f.__module__}_{f.__name__}{hash(tuple(_sorted_args))}"
        logger.debug(f"key is {key}")
        return key

    def get_cache_file(self, key: str) -> Path:
        return self.cache_dir / (str(key) + ".cache")

    def get(self, key: str) -> Any | None:
        cache_file = self.get_cache_file(key)
        if cache_file.exists():
            logger.debug(f"loading {key} from cache")
            with cache_file.open("rb") as f:
                return pickle.load(f)
        else:
            logger.debug(f"{key} not in cache")
            return None

    def set(self, key: str, value: bytes):
        cache_file = self.get_cache_file(key)
        logger.debug(f"saving {key} to cache {cache_file}")
        with cache_file.open("wb") as f:
            pickle.dump(value, f)

    @staticmethod
    def cache(f):
        assert f.__module__.startswith("thesis_plots."), "cached function must be in thesis_plots"
        for n, t in get_type_hints(f).items():
            assert issubclass(t, Hashable), f"argument {n} of {f} is not hashable"

        def wrapper(*args, **kwargs):
            cache = DiskCache()
            key = cache.get_key(f, args, kwargs)
            value = cache.get(key)
            if value is None:
                value = f(*args, **kwargs)
                cache.set(key, value)
            return value

        return wrapper
