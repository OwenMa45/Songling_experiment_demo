"""Calibratable displacement/pressure surrogate, not a CFD solver."""
from dataclasses import dataclass, field
import math


@dataclass
class Fluid:
    config: dict
    compression: float = 0.0
    pressure_volume: float = 0.0
    accumulator: float = 0.0
    emitted: int = 0
    received: int = 0
    spilled: int = 0
    pending: list = field(default_factory=list)

    def __post_init__(self):
        self.remaining = float(self.config["initial_volume_ml"])

    def step(self, dt, compression, tip, tube, radius, now):
        c = self.config
        compression = min(1.0, max(0.0, float(compression)))
        old = max(0, self.compression - c["compression_threshold"])
        new = max(0, compression - c["compression_threshold"])
        self.pressure_volume += max(0, new - old) * c["displacement_ml"]
        if compression < self.compression:
            self.pressure_volume *= math.exp(-dt / c["release_seconds"])
        self.compression = compression
        flow = min(self.remaining, self.pressure_volume * (1 - math.exp(-dt / c["relaxation_seconds"])))
        self.pressure_volume -= flow
        self.remaining -= flow
        self.accumulator += flow
        while self.accumulator + 1e-12 >= c["drop_volume_ml"]:
            self.accumulator -= c["drop_volume_ml"]
            self.emitted += 1
            height = tip[2] - tube[2]
            hit = height > 0 and math.hypot(tip[0] - tube[0], tip[1] - tube[1]) < radius
            self.pending.append((now + math.sqrt(2 * max(height, 0) / 9.81), hit, tuple(tip), now))
        waiting = []
        for arrival, hit, pos, start in self.pending:
            if now >= arrival:
                self.received += int(hit)
                self.spilled += int(not hit)
            else:
                waiting.append((arrival, hit, pos, start))
        self.pending = waiting


class Counter:
    """Interface also implementable by a real, independently validated vision counter."""
    def read(self):
        raise NotImplementedError


class SimulationCounter(Counter):
    def __init__(self, fluid):
        self.fluid = fluid

    def read(self):
        return self.fluid.received
