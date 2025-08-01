"""Module for audio synthesis and effects



Convention: Standard SI units are used in parameters that represent quantities
unless specified otherwise
"""


from __future__ import annotations

import random
import struct
import wave
import collections
from collections.abc import Iterable
import itertools as it
from math import cos, pi, sin, log, ceil
from cmath import exp as cexp
from subprocess import PIPE, Popen
from typing import Callable, Union, Self, overload

SAMPRATE = 48_000.0

SignalLike = Union["Signal", Iterable[float], float]


class Signal:
    """Abstract representation of a signal

    In general, the `frames` instance variable is an iterator, which
    allows for signals of infinite length. This is useful if we don't
    know the length of of the final signal beforehand. See method
    `over` for the conversion to a finite sound.

    In the context of audio manipulation, we can use instances of this
    class both as audio signals and control signals e.g. to control
    volume or pitch.
    """

    frames: Iterable[float]

    def __init__(self, input: SignalLike) -> None:
        """Create a signal from a SignalLike input

        - if input is a float, a corresponding infinite constant signal
          gets created
        - if input is an Iterable[float], a signal with the
          corresponding frames gets created
        - if input is a signal, an identical signal gets created
        """
        if isinstance(input, Signal):
            self.frames = input.frames
        elif isinstance(input, (int, float)):
            self.frames = it.repeat(input)
        elif isinstance(input, Iterable):
            self.frames = input
        else:
            t = type(input).__qualname__
            raise TypeError(f"Cannot create a signal from type {t}")

    def __add__(self, other: SignalLike) -> Self:
        return self._bin_op(other, lambda x, y: x + y)

    def __radd__(self, other: SignalLike) -> Self:
        return self._bin_op(other, lambda x, y: y + x)

    def __neg__(self) -> Self:
        return self.map(lambda x: -x)

    def __sub__(self, other: SignalLike) -> Self:
        return self._bin_op(other, lambda x, y: x - y)

    def __rsub__(self, other: SignalLike) -> Self:
        return self._bin_op(other, lambda x, y: y - x)

    def __mul__(self, other: SignalLike) -> Self:
        return self._bin_op(other, lambda x, y: x * y)

    def __rmul__(self, other: SignalLike) -> Self:
        return self._bin_op(other, lambda x, y: y * x)

    def __truediv__(self, other: SignalLike) -> Self:
        return self._bin_op(other, lambda x, y: x / y)

    def __rtruediv__(self, other: SignalLike) -> Self:
        return self._bin_op(other, lambda x, y: y / x)

    def __pow__(self, other: SignalLike) -> Self:
        return self._bin_op(other, lambda x, y: x**y)

    def __rpow__(self, other: SignalLike) -> Self:
        return self._bin_op(other, lambda x, y: y**x)

    def _bin_op(
        self,
        other: SignalLike,
        op: Callable[[float, float], float],
    ) -> Self:
        if isinstance(other, FinSignal):
            self, other = other, self
        z = zip(self.frames, Signal(other).frames)
        return self.__class__(op(x, y) for x, y in z)

    def tee(self, n: int = 2) -> list[Self]:
        """Return n independent signals created using self."""
        return [self.__class__(frames) for frames in it.tee(self.frames, n)]

    def over(self, dur: float) -> FinSignal:
        """Return the a finite signal consisting of the start of
        the signal. The duration is specified in seconds."""
        return FinSignal(it.islice(self.frames, round(dur * SAMPRATE)))

    def extend(self) -> Signal:
        """Return an infinite signal with begins with self and
        continues as a const(0)"""
        return Signal(it.chain(self.frames, it.repeat(0.0)))

    @overload
    def repeat(self, times: None) -> Signal: ...
    @overload
    def repeat(self, times: int) -> Self: ...
    def repeat(self, times: int | None = None) -> Signal:
        """Repeat the signal. If times is None, repeat infinitely."""

        def frames():
            for _ in it.count() if times is None else range(times):
                for frame in self.frames:
                    yield frame

        if times is None:
            return Signal(frames())
        else:
            return self.__class__(frames())

    # For more info about the low-pass implementation, see
    # https://en.wikipedia.org/wiki/Low-pass_filter#Discrete-time_realization.

    def lowpass(self, cutoff: SignalLike) -> Self:
        """Apply the low-pass filter with a specified cutoff frequency"""
        return self.lowpasst(1 / ((2 * pi) * Signal(cutoff)))

    def lowpasst(self, tau: SignalLike) -> Self:
        """Apply the low-pass filter with a specified time constant"""
        a = 1 / (SAMPRATE * Signal(tau) + 1)

        def frames() -> Iterable[float]:
            y_fr = 0.0
            for a_fr, x_fr in zip(a.frames, self.frames):
                y_fr = a_fr * x_fr + (1 - a_fr) * y_fr
                yield y_fr

        return self.__class__(frames())

    # For more info about the high-pass implementation, see
    # https://en.wikipedia.org/wiki/High-pass_filter#Discrete-time_realization.

    def highpasst(self, tau: SignalLike) -> Self:
        """Apply the high-pass filter with a specified time constant"""
        return self.highpass(1 / ((2 * pi) * Signal(tau)))

    def highpass(self, cutoff: SignalLike) -> Self:
        """Apply the high-pass filter with a specified cutoff frequency"""
        a = 1 / ((2 * pi / SAMPRATE) * Signal(cutoff) + 1)

        def frames() -> Iterable[float]:
            y_fr = prev_x_fr = 0.0
            for a_fr, x_fr in zip(a.frames, self.frames):
                y_fr = a_fr * (y_fr + x_fr - prev_x_fr)
                yield y_fr
                prev_x_fr = x_fr

        return self.__class__(frames())

    def distort(self, f: Callable[[float], float]) -> Self:
        """Distort the signal by applying f to each frame. All frames
        are scaled to (-1, 1)"""
        m = max(self.map(abs).frames)
        return self.map(lambda x: f(x / m))

    def delay(self, period, coeff) -> Signal:
        """Apply a delay scaled by coeff to the signal with the
        specified period."""
        shift = round(period * SAMPRATE)
        history = collections.deque()

        def frames() -> Iterable[float]:
            for x in self.frames:
                echo = history.popleft() if len(history) == shift else 0
                history.append(x + coeff * echo)
                yield history[-1]

        return self.__class__(frames())

    def map(self, func: Callable[[float], float]) -> Self:
        """Apply func to each frame of the signal"""
        return self.__class__(map(func, self.frames))

    def convolve(self, kernel: FinSignal) -> Self:
        k = len(kernel.frames)
        window = collections.deque([0.0] * k)

        def frames() -> Iterable[float]:
            for x in it.chain(self.frames, [0.0] * (k - 1)):
                window.popleft()
                window.append(x)
                yield sum(a * b for a, b in zip(window, kernel.frames))

        return self.__class__(frames())


class FinSignal(Signal):
    """A subclass of Signal for Finite signals"""

    frames: list[float]

    def __init__(self, input: SignalLike) -> None:
        self.frames = list(Signal(input).frames)

    def __repr__(self) -> str:
        self.play()
        return Signal.__repr__(self)

    @property
    def dur(self):
        """The duration of the signal"""
        return len(self.frames) / SAMPRATE

    def play(self) -> None:
        """Play the peak-normalised signal"""
        sound = self.normalised() * 32767
        cmd = ["aplay", "-f", "S16_LE", "-r", str(round(SAMPRATE))]
        with Popen(cmd, stdin=PIPE) as player:
            if player.stdin is None:
                raise ValueError
            for fr in sound.frames:
                player.stdin.write(struct.pack("h", round(fr)))
                player.stdin.flush()

    def export_wav(self, filename: str = "out.wav") -> None:
        """Export the peak-normalised signal to a wav file"""
        sound = self.normalised() * 32767
        with wave.open(filename, "wb") as file:
            file.setsampwidth(2)
            file.setnchannels(1)
            file.setframerate(SAMPRATE)
            for frame in sound.frames:
                file.writeframes(struct.pack("h", round(frame)))

    def normalised(self) -> FinSignal:
        """Peak-normalise the sound to (-1, 1)"""
        if not self.frames:
            return self
        maxi = max(self.frames)
        mini = min(self.frames)
        amp = (maxi - mini) / 2
        shift = (maxi + mini) / 2
        return (self - shift) / amp

    def reversed(self) -> FinSignal:
        return FinSignal(reversed(self.frames))

    def reverb(self, imp_resp: FinSignal) -> FinSignal:
        return FinSignal(_convolve(self.frames, imp_resp.frames))

def _convolve(a: list[float], b: list[float]):
    """Return the convolution of a and b, where the orientation of
    the inputs is opposite. (Reverse one of the inputs to get
    a traditional convolution.)"""
    # This function works using the fact that circular convolution in the time
    # domain corresponds to multiplication in the frequency domain. With
    # the use of FFT, the algorithm therefore has a O(n log n) time complexity.
    # We pad with enough zeros (>= len(a) + len(b)), such that the end to start
    # wrapping of the circular convolution doesn't affect the result.
    length = 2 ** ceil(log(len(a) + len(b), 2))
    a = a + [0.0] * (length - len(a))
    b = b + [0.0] * (length - len(b))
    result_f = [x * y for x, y in zip(_fft(a), _fft(b))]
    return _ifft(result_f)[: len(a) + len(b) - 1]


def _fft(a: list[float]) -> list[complex]:
    """A fast (O(n log n)) algorithm for the discrete fourier transform
    of a, where len(a) is a power of two."""
    return _fft_helper(a, False)


def _ifft(a: list[complex]) -> list[float]:
    """A fast (O(n log n)) algorithm for the inverse discrete fourier
    transform of a, where len(a) is a power of two."""
    return [x.real / len(a) for x in _fft_helper(a, True)]


def _fft_helper(a: list[complex], inverse: bool) -> list[complex]:
    if len(a) == 1:
        return a
    if len(a) % 2 != 0:
        raise ValueError("Input length isn't a power of two")
    even = _fft(a[::2], inverse)
    odd = _fft(a[1::2], inverse)
    mid = len(a) // 2
    result = [0] * len(a)
    for i in range(mid):
        e = even[i]
        o = cexp((1 if inverse else -1) * 2j * pi * i / len(a)) * odd[i]
        result[i] = e + o
        result[mid + i] = e - o
    return result


def load(filename: str) -> FinSignal:
    """Load a signal from a wave file with the correct parameters.

    To convert a general audio file to the desired format, one can use
    ffmpeg -i input_file -ac 1 -ar SAMPRATE output_file.wav"""
    with wave.open(filename, "rb") as f:
        if (
            f.getnchannels() == 1
            and f.getsampwidth() == 2
            and f.getframerate() == SAMPRATE
            and f.getcomptype() == "NONE"
        ):
            n = f.getnframes()
            frames = [x for (x,) in struct.iter_unpack("h", f.readframes(n))]
            return FinSignal(frames)
        else:
            print(f.getparams())
            raise ValueError(
                f"{filename} must have 1 channel, 2 byte samples,"
                f"sample rate of {SAMPRATE} and no compression."
            )


def white() -> Signal:
    """Returns white noise"""
    return Signal(random.uniform(-1.0, 1.0) for _ in it.repeat(None))


def saw(freq: SignalLike) -> Signal:
    """Returns the signal of a sawtooth wave form with peaks at 1 and -1"""

    def frames():
        out = 0
        for f in Signal(freq).frames:
            yield out
            step = f / SAMPRATE
            if out + step >= 1:
                out = -2 + out + step
            out += f / SAMPRATE

    return Signal(frames())


def triangle(freq: SignalLike) -> Signal:
    """Returns the signal of a triangle wave form with peaks at 1 and -1"""

    def frames():
        out = 0
        d = 1
        for f in Signal(freq).frames:
            yield out
            step = 2 * f / SAMPRATE
            if d == 1 and out + step >= 1:
                out = 2 - out - step
                d = -d
            elif d == -1 and out - step <= -1:
                out = -2 - (out - step)
                d = -d
            else:
                out += d * step

    return Signal(frames())


def step() -> Signal:
    """Returns the signal of a unit step function"""
    return Signal(it.chain([0.0], it.repeat(1.0)))


def stepdown() -> Signal:
    """Returns the signal of a unit step down function (starting at 1
    and immediately going to 0)"""
    return Signal(it.chain([1.0], it.repeat(0.0)))


def decay(tau: SignalLike) -> Signal:
    """Return an exponential decay signal starting at 1 and approaching
    0 with time constant tau."""
    return stepdown().lowpasst(tau)


def rise(tau: SignalLike) -> Signal:
    """Return an exponential rise signal starting at 0 and approaching
    1 with time constant tau."""
    return step().lowpasst(tau)


def linear() -> Signal:
    """Return a linear signal starting at 0 with a unit slope in
    seconds."""
    return Signal(it.count(step=1 / SAMPRATE))


def exp() -> Signal:
    """Return an exponential signal starting at one and doubling every
    second"""
    return 2 ** linear()


def sine(freq: SignalLike) -> Signal:
    """Return a unit sine signal with the specified frequency"""
    freq = Signal(freq)

    def frames() -> Iterable[float]:
        x, y = 1.0, 0.0
        for f in freq.frames:
            yield y
            d = 2 * pi * f / SAMPRATE
            # Apply the rotation matrix
            x, y = x * cos(d) - y * sin(d), x * sin(d) + y * cos(d)

    return Signal(frames())


def rect(freq: SignalLike) -> Signal:
    """Returns the signal of a rectangular wave form with peaks at 1 and -1"""
    return sine(freq).map(lambda x: 1 if x >= 0 else -1)


def compose(*table: tuple[SignalLike, float] | FinSignal) -> FinSignal:
    """Compose multiple signals in series

    Each argument can either be a (signal, duration) tuple or a finite,
    finite whose duration is determined directly. The first type of
    argument allows overlapping signals (if duration is less than the
    duration of the signal)"""
    tot_frames = round(
        sum(
            len(item.frames) if isinstance(item, FinSignal) else item[1] * SAMPRATE
            for item in table
        )
    )
    result = FinSignal([0] * tot_frames)
    start = 0
    for item in table:
        if isinstance(item, tuple):
            sig, dur = item
            dur_frames = round(dur * SAMPRATE)
            sig = Signal(sig).extend()
        elif isinstance(item, FinSignal):
            sig = item
            dur_frames = len(item.frames)
        for i, x in enumerate(sig.frames):
            if start + i >= tot_frames:
                break
            result.frames[start + i] += x
        start += dur_frames
    return FinSignal(result)
