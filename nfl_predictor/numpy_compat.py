"""Work around Windows Application Control blocking numpy.random._generator.

When that DLL is blocked, numpy falls back to RandomState, which:
- only accepts 32-bit seeds (SciPy uses larger ones)
- is missing Generator methods such as permuted
"""

from __future__ import annotations


def patch_numpy_default_rng() -> None:
    import numpy as np

    original = getattr(np.random, "default_rng", None)
    if original is None or getattr(original, "_nfl_predictor_patched", False):
        return

    class GeneratorShim:
        def __init__(self, seed=None):
            if isinstance(seed, int):
                seed = int(seed) % (2**32)
            self._rs = np.random.RandomState(seed)

        def permuted(self, x, axis=0, out=None):
            arr = np.array(x, copy=True)
            if arr.ndim <= 1 or axis == 0:
                self._rs.shuffle(arr)
            else:
                arr = np.moveaxis(arr, axis, 0)
                for i in range(arr.shape[0]):
                    self._rs.shuffle(arr[i])
                arr = np.moveaxis(arr, 0, axis)
            if out is not None:
                out[...] = arr
                return out
            return arr

        def random(self, size=None, dtype=float, out=None):
            val = self._rs.random_sample(size)
            if out is not None:
                out[...] = val
                return out
            return val.astype(dtype, copy=False)

        def integers(self, low, high=None, size=None, dtype=int, endpoint=False):
            if high is None:
                high = low
                low = 0
            if endpoint:
                high = np.asarray(high) + 1
            return self._rs.randint(low, high, size=size, dtype=dtype)

        def normal(self, loc=0.0, scale=1.0, size=None):
            return self._rs.normal(loc, scale, size)

        def uniform(self, low=0.0, high=1.0, size=None):
            return self._rs.uniform(low, high, size)

        def standard_normal(self, size=None, dtype=float, out=None):
            val = self._rs.standard_normal(size)
            if out is not None:
                out[...] = val
                return out
            return val

        def choice(self, a, size=None, replace=True, p=None, axis=0, shuffle=True):
            return self._rs.choice(a, size=size, replace=replace, p=p)

        def shuffle(self, x, axis=0):
            if axis == 0 or getattr(x, "ndim", 1) <= 1:
                return self._rs.shuffle(x)
            arr = np.moveaxis(np.asarray(x), axis, 0)
            self._rs.shuffle(arr)
            return None

        def permutation(self, x, axis=0):
            if isinstance(x, int):
                return self._rs.permutation(x)
            arr = np.array(x, copy=True)
            if axis != 0:
                arr = np.moveaxis(arr, axis, 0)
                self._rs.shuffle(arr)
                return np.moveaxis(arr, 0, axis)
            return self._rs.permutation(arr)

        def spawn(self, n_children):
            seeds = self._rs.randint(0, 2**31 - 1, size=n_children)
            return [GeneratorShim(int(s)) for s in seeds]

        def __getattr__(self, name):
            return getattr(self._rs, name)

    def default_rng(seed=None):
        try:
            rng = original(seed)
        except (ValueError, TypeError, OverflowError):
            rng = original(int(seed) % (2**32) if isinstance(seed, int) else None)
        if hasattr(rng, "permuted"):
            return rng
        return GeneratorShim(seed)

    default_rng._nfl_predictor_patched = True  # type: ignore[attr-defined]
    np.random.default_rng = default_rng
    if not hasattr(getattr(np.random, "Generator", object), "permuted"):
        np.random.Generator = GeneratorShim


patch_numpy_default_rng()
