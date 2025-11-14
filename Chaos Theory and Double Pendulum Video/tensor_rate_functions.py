from __future__ import annotations
from functools import wraps
import typing
import torch

__all__ = [
    "linear",
    "smooth",
    "smoothstep",
    "smootherstep",
    "smoothererstep",
    "rush_into",
    "rush_from",
    "slow_into",
    "double_smooth",
    "there_and_back",
    "there_and_back_with_pause",
    "running_start",
    "not_quite_there",
    "wiggle",
    "squish_rate_func",
    "lingering",
    "exponential_decay",
]

from manim import bezier


# This is a decorator that makes sure any function it's used on will
# return 0 if t<0 and 1 if t>1.
def unit_interval(function):
    @wraps(function)
    def wrapper(t, *args, **kwargs):
        t = torch.clamp(t, 0, 1)
        return function(t, *args, **kwargs)
    return wrapper


def zero(function):
    @wraps(function)
    def wrapper(t, *args, **kwargs):
        t = torch.where((t >= 0) & (t <= 1), t, torch.tensor(0.0, device=t.device))
        return function(t, *args, **kwargs)
    return wrapper


@unit_interval
def linear(t: torch.Tensor) -> torch.Tensor:
    return t


@unit_interval
def smooth(t: torch.Tensor, inflection: float = 10.0) -> torch.Tensor:
    error = torch.sigmoid(torch.tensor(-inflection / 2))
    return torch.clamp((torch.sigmoid(inflection * (t - 0.5)) - error) / (1 - 2 * error), 0, 1)


def smoothstep(t: torch.Tensor) -> torch.Tensor:
    return torch.where(t <= 0, torch.tensor(0.0, device=t.device),
                       torch.where(t < 1, 3 * t**2 - 2 * t**3, torch.tensor(1.0, device=t.device)))


def smootherstep(t: torch.Tensor) -> torch.Tensor:
    return torch.where(t <= 0, torch.tensor(0.0, device=t.device),
                       torch.where(t < 1, 6 * t**5 - 15 * t**4 + 10 * t**3, torch.tensor(1.0, device=t.device)))


def smoothererstep(t: torch.Tensor) -> torch.Tensor:
    alpha = torch.where((t > 0) & (t < 1), 35 * t**4 - 84 * t**5 + 70 * t**6 - 20 * t**7, torch.tensor(0.0, device=t.device))
    alpha = torch.where(t >= 1, torch.tensor(1.0, device=t.device), alpha)
    return alpha


@unit_interval
def rush_into(t: torch.Tensor, inflection: float = 10.0) -> torch.Tensor:
    return 2 * smooth(t / 2.0, inflection)


@unit_interval
def rush_from(t: torch.Tensor, inflection: float = 10.0) -> torch.Tensor:
    return 2 * smooth(t / 2.0 + 0.5, inflection) - 1


@unit_interval
def slow_into(t: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(1 - (1 - t) * (1 - t))


@unit_interval
def double_smooth(t: torch.Tensor) -> torch.Tensor:
    return torch.where(t < 0.5, 0.5 * smooth(2 * t), 0.5 * (1 + smooth(2 * t - 1)))


@zero
def there_and_back(t: torch.Tensor, inflection: float = 10.0) -> torch.Tensor:
    new_t = torch.where(t < 0.5, 2 * t, 2 * (1 - t))
    return smooth(new_t, inflection)


@zero
def there_and_back_with_pause(t: torch.Tensor, pause_ratio: float = 1.0 / 3) -> torch.Tensor:
    a = 1.0 / pause_ratio
    return torch.where(t < 0.5 - pause_ratio / 2, smooth(a * t),
                       torch.where(t < 0.5 + pause_ratio / 2, torch.tensor(1.0, device=t.device), smooth(a - a * t)))


def not_quite_there(func: typing.Callable[[torch.Tensor], torch.Tensor] = smooth, proportion: float = 0.7) -> typing.Callable[[torch.Tensor], torch.Tensor]:
    def result(t):
        return proportion * func(t)
    return result


@zero
def wiggle(t: torch.Tensor, wiggles: float = 2) -> torch.Tensor:
    return there_and_back(t) * torch.sin(wiggles * torch.pi * t)


def squish_rate_func(func: typing.Callable[[torch.Tensor], torch.Tensor], a: float = 0.4, b: float = 0.6) -> typing.Callable[[torch.Tensor], torch.Tensor]:
    def result(t):
        return torch.where(t < a, func(torch.tensor(0.0, device=t.device)),
                           torch.where(t > b, func(torch.tensor(1.0, device=t.device)), func((t - a) / (b - a))))
    return result


@unit_interval
def lingering(t: torch.Tensor) -> torch.Tensor:
    return squish_rate_func(lambda t: t, 0, 0.8)(t)


@unit_interval
def exponential_decay(t: torch.Tensor, half_life: float = 0.1) -> torch.Tensor:
    return 1 - torch.exp(-t / half_life)


@unit_interval
def ease_in_sine(t: torch.Tensor) -> torch.Tensor:
    return 1 - torch.cos((t * torch.pi) / 2)


@unit_interval
def ease_out_sine(t: torch.Tensor) -> torch.Tensor:
    return torch.sin((t * torch.pi) / 2)


@unit_interval
def ease_in_out_sine(t: torch.Tensor) -> torch.Tensor:
    return -(torch.cos(torch.pi * t) - 1) / 2


@unit_interval
def ease_in_quad(t: torch.Tensor) -> torch.Tensor:
    return t * t


@unit_interval
def ease_out_quad(t: torch.Tensor) -> torch.Tensor:
    return 1 - (1 - t) * (1 - t)


@unit_interval
def ease_in_out_quad(t: torch.Tensor) -> torch.Tensor:
    return torch.where(t < 0.5, 2 * t * t, 1 - torch.pow(-2 * t + 2, 2) / 2)


@unit_interval
def ease_in_cubic(t: torch.Tensor) -> torch.Tensor:
    return t * t * t


@unit_interval
def ease_out_cubic(t: torch.Tensor) -> torch.Tensor:
    return 1 - torch.pow(1 - t, 3)


@unit_interval
def ease_in_out_cubic(t: torch.Tensor) -> torch.Tensor:
    return torch.where(t < 0.5, 4 * t * t * t, 1 - torch.pow(-2 * t + 2, 3) / 2)


@unit_interval
def ease_in_quart(t: torch.Tensor) -> torch.Tensor:
    return t * t * t * t


@unit_interval
def ease_out_quart(t: torch.Tensor) -> torch.Tensor:
    return 1 - torch.pow(1 - t, 4)


@unit_interval
def ease_in_out_quart(t: torch.Tensor) -> torch.Tensor:
    return torch.where(t < 0.5, 8 * t * t * t * t, 1 - torch.pow(-2 * t + 2, 4) / 2)


@unit_interval
def ease_in_quint(t: torch.Tensor) -> torch.Tensor:
    return t * t * t * t * t


@unit_interval
def ease_out_quint(t: torch.Tensor) -> torch.Tensor:
    return 1 - torch.pow(1 - t, 5)


@unit_interval
def ease_in_out_quint(t: torch.Tensor) -> torch.Tensor:
    return torch.where(t < 0.5, 16 * t * t * t * t * t, 1 - torch.pow(-2 * t + 2, 5) / 2)


@unit_interval
def ease_in_expo(t: torch.Tensor) -> torch.Tensor:
    return torch.where(t == 0, torch.tensor(0.0, device=t.device), torch.pow(2, 10 * t - 10))


@unit_interval
def ease_out_expo(t: torch.Tensor) -> torch.Tensor:
    return torch.where(t == 1, torch.tensor(1.0, device=t.device), 1 - torch.pow(2, -10 * t))


@unit_interval
def ease_in_out_expo(t: torch.Tensor) -> torch.Tensor:
    return torch.where(t == 0, torch.tensor(0.0, device=t.device),
                       torch.where(t == 1, torch.tensor(1.0, device=t.device),
                                   torch.where(t < 0.5, torch.pow(2, 20 * t - 10) / 2, (2 - torch.pow(2, -20 * t + 10)) / 2)))


@unit_interval
def ease_in_circ(t: torch.Tensor) -> torch.Tensor:
    return 1 - torch.sqrt(1 - torch.pow(t, 2))


@unit_interval
def ease_out_circ(t: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(1 - torch.pow(t - 1, 2))


@unit_interval
def ease_in_out_circ(t: torch.Tensor) -> torch.Tensor:
    return torch.where(t < 0.5, (1 - torch.sqrt(1 - torch.pow(2 * t, 2))) / 2,
                       (torch.sqrt(1 - torch.pow(-2 * t + 2, 2)) + 1) / 2)


@unit_interval
def ease_in_back(t: torch.Tensor) -> torch.Tensor:
    c1 = 1.70158
    c3 = c1 + 1
    return c3 * t * t * t - c1 * t * t


@unit_interval
def ease_out_back(t: torch.Tensor) -> torch.Tensor:
    c1 = 1.70158
    c3 = c1 + 1
    return 1 + c3 * torch.pow(t - 1, 3) + c1 * torch.pow(t - 1, 2)


@unit_interval
def ease_in_out_back(t: torch.Tensor) -> torch.Tensor:
    c1 = 1.70158
    c2 = c1 * 1.525
    return torch.where(t < 0.5, (torch.pow(2 * t, 2) * ((c2 + 1) * 2 * t - c2)) / 2,
                       (torch.pow(2 * t - 2, 2) * ((c2 + 1) * (t * 2 - 2) + c2) + 2) / 2)


@unit_interval
def ease_in_elastic(t: torch.Tensor) -> torch.Tensor:
    c4 = (2 * torch.pi) / 3
    return torch.where(t == 0, torch.tensor(0.0, device=t.device),
                       torch.where(t == 1, torch.tensor(1.0, device=t.device),
                                   -torch.pow(2, 10 * t - 10) * torch.sin((t * 10 - 10.75) * c4)))


@unit_interval
def ease_out_elastic(t: torch.Tensor) -> torch.Tensor:
    c4 = (2 * torch.pi) / 3
    return torch.where(t == 0, torch.tensor(0.0, device=t.device),
                       torch.where(t == 1, torch.tensor(1.0, device=t.device),
                                   torch.pow(2, -10 * t) * torch.sin((t * 10 - 0.75) * c4) + 1))


@unit_interval
def ease_in_out_elastic(t: torch.Tensor) -> torch.Tensor:
    c5 = (2 * torch.pi) / 4.5
    return torch.where(t == 0, torch.tensor(0.0, device=t.device),
                       torch.where(t == 1, torch.tensor(1.0, device=t.device),
                                   torch.where(t < 0.5, -(torch.pow(2, 20 * t - 10) * torch.sin((20 * t - 11.125) * c5)) / 2,
                                               (torch.pow(2, -20 * t + 10) * torch.sin((20 * t - 11.125) * c5)) / 2 + 1)))


@unit_interval
def ease_in_bounce(t: torch.Tensor) -> torch.Tensor:
    return 1 - ease_out_bounce(1 - t)


@unit_interval
def ease_out_bounce(t: torch.Tensor) -> torch.Tensor:
    n1 = 7.5625
    d1 = 2.75
    return torch.where(t < 1 / d1, n1 * t * t,
                       torch.where(t < 2 / d1, n1 * (t - 1.5 / d1) * (t - 1.5 / d1) + 0.75,
                                   torch.where(t < 2.5 / d1, n1 * (t - 2.25 / d1) * (t - 2.25 / d1) + 0.9375,
                                               n1 * (t - 2.625 / d1) * (t - 2.625 / d1) + 0.984375)))


@unit_interval
def ease_in_out_bounce(t: torch.Tensor) -> torch.Tensor:
    return torch.where(t < 0.5, (1 - ease_out_bounce(1 - 2 * t)) / 2,
                       (1 + ease_out_bounce(2 * t - 1)) / 2)
