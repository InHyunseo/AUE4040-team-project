"""Deterministic conversion: road-center (x, y) -> steering."""


def center_to_steering(x_norm: float, k: float = 1.2) -> float:
    val = k * x_norm
    if val > 1.0:
        return 1.0
    if val < -1.0:
        return -1.0
    return val
