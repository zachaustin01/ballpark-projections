"""Rolling-window sums keyed by (entity, day) via cumulative sums + searchsorted."""
import numpy as np

STRIDE = 100_000


class WindowSum:
    def __init__(self, entity, dn, values):
        entity = np.asarray(entity, np.int64)
        values = np.asarray(values, np.float64)
        if values.ndim == 1:
            values = values[:, None]
        key = entity * STRIDE + np.asarray(dn, np.int64)
        order = np.argsort(key, kind="stable")
        self.order = order
        self.key = key[order]
        self.cum = np.vstack([np.zeros((1, values.shape[1])), np.cumsum(values[order], axis=0)])

    def query(self, entity, d0, d1):
        """Sum of values with entity == e and d0 <= dn < d1, per query row."""
        e = np.asarray(entity, np.int64) * STRIDE
        lo = np.searchsorted(self.key, e + np.asarray(d0, np.int64), "left")
        hi = np.searchsorted(self.key, e + np.asarray(d1, np.int64), "left")
        return self.cum[hi] - self.cum[lo]

    def span(self, entity, d0, d1):
        """(lo, hi) positions in the sorted order; rows are `self.order[lo:hi]`."""
        e = np.asarray(entity, np.int64) * STRIDE
        return (np.searchsorted(self.key, e + np.asarray(d0, np.int64), "left"),
                np.searchsorted(self.key, e + np.asarray(d1, np.int64), "left"))
