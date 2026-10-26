"""Power schedules for sampling from noise (1) to the image endpoint (0)."""

import math
from numbers import Real


def power_schedule(steps: int, rho: float = 1.0) -> tuple[float, ...]:
    """Return ``steps + 1`` strictly decreasing times, including 1 and 0.

    ``rho=1`` preserves the original uniform schedule. Reject exponents whose
    floating-point rounding would collapse any interval to zero length.
    """
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")
    if isinstance(rho, bool) or not isinstance(rho, Real):
        raise ValueError("rho must be a finite positive number")
    try:
        rho = float(rho)
    except (ValueError, OverflowError) as error:
        raise ValueError("rho must be a finite positive number") from error
    if not math.isfinite(rho) or rho <= 0:
        raise ValueError("rho must be a finite positive number")

    times = tuple((1 - i / steps) ** rho for i in range(steps + 1))
    if times[0] != 1.0 or times[-1] != 0.0 or not all(
        start > end for start, end in zip(times[:-1], times[1:])
    ):
        raise ValueError("rho must produce strictly decreasing sampling times")
    return times
