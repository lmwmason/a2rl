"""Betaflight MSP v1 RC boundary for the simulator's Mode 2 controller."""
from functools import reduce
from operator import xor
from struct import Struct

import numpy as np

HEADER = b"$M<"
SET_RAW_RC = 200
CHANNELS = Struct("<4H")


def timed_mean_throttle(actions, sample_rate, start, end):
    """Average the held Mode 2 throttle over the timed race interval."""
    if not sample_rate > 0 or not 0 <= start < end <= len(actions) / sample_rate:
        return None
    boundaries = np.arange(len(actions) + 1) / sample_rate
    weights = np.diff(np.clip(boundaries, start, end))
    throttle = (np.asarray(actions)[:, 1] + 1.) / 2.
    return float(np.dot(weights, throttle) / (end - start))


def encode_rc(sticks):
    """Convert Mode 2 [yaw, throttle, roll, pitch] to a four-channel AETR frame."""
    sticks = np.asarray(sticks)
    if sticks.shape != (4,) or not np.isfinite(sticks).all():
        raise ValueError("Expected four finite Mode 2 sticks.")
    pwm = np.rint(1500 + 500 * np.clip(sticks[[2, 3, 1, 0]], -1, 1)).astype(int)
    body = bytes((CHANNELS.size, SET_RAW_RC)) + CHANNELS.pack(*pwm)
    return HEADER + body + bytes((reduce(xor, body),))


def decode_rc(packet):
    """Read the AETR channels a receiver presents to the simulated Acro controller."""
    if (len(packet) != 6 + CHANNELS.size or packet[:3] != HEADER
            or packet[3:5] != bytes((CHANNELS.size, SET_RAW_RC))
            or reduce(xor, packet[3:]) != 0):
        raise ValueError("Invalid MSP_SET_RAW_RC frame.")
    roll, pitch, throttle, yaw = CHANNELS.unpack(packet[5:-1])
    pwm = np.array([yaw, throttle, roll, pitch], dtype=np.float32)
    if np.any((pwm < 1000) | (pwm > 2000)):
        raise ValueError("RC channels must be in the configured 1000–2000 µs range.")
    return (pwm - 1500.) / 500.
