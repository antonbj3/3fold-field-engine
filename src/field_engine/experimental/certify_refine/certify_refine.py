from dataclasses import dataclass, field
from time import perf_counter
import math

INNE, UTE, OSAKER = "INNE", "UTE", "OSÄKER"


def decide(value, band, threshold=0.0, mode="below"):
    if mode == "below":
        if not (math.isfinite(value) and math.isfinite(band) and band >= 0.0):
            return OSAKER
        if value + band < threshold:
            return INNE
        if value - band > threshold:
            return UTE
        return OSAKER
    if mode == "all_above":
        states = [decide(-value, error, -threshold) for value, error in zip(value, band)]
        if UTE in states:
            return UTE
        return INNE if states and all(state == INNE for state in states) else OSAKER
    raise ValueError(mode)


@dataclass
class Statistics:
    queries: int = 0
    refined: int = 0
    cheap_s: float = 0.0
    refine_s: float = 0.0

    @property
    def fraction_refined(self):
        return self.refined / self.queries if self.queries else 0.0


@dataclass
class Filter:
    adapter: object
    stats: Statistics = field(default_factory=Statistics)

    def query(self, query):
        """Return INNE/UTE/GRÄNS after optional exact refinement.

        Coordinates use mm in a caller-supplied Frame. The adapter supplies
        a running arithmetic error band for represented binary inputs in its
        declared regime; no geometry or physical model error is enclosed.
        """
        started = perf_counter()
        value, band = self.adapter.cheap(query)
        self.stats.cheap_s += perf_counter() - started
        self.stats.queries += 1
        state = decide(value, band, self.adapter.threshold, self.adapter.mode)
        if state != OSAKER:
            return state
        started = perf_counter()
        fine = self.adapter.refine(query)
        self.stats.refine_s += perf_counter() - started
        self.stats.refined += 1
        return self.adapter.exact_decide(fine)


@dataclass(frozen=True)
class Frame:
    origin: tuple
    rows: tuple = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))

    def to_local(self, point):
        delta = [float(point[i]) - float(self.origin[i]) for i in range(3)]
        return tuple(sum(self.rows[j][i] * delta[i] for i in range(3)) for j in range(3))

    def to_world(self, point):
        return tuple(
            float(self.origin[i]) + sum(self.rows[j][i] * point[j] for j in range(3))
            for i in range(3)
        )
