"""Compare the simulator's axial force with 4TU MonoRace Log 8 IMU and EKF data."""
import argparse
import ctypes as ct

import numpy as np

from training.sim import load_sim


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("imu")
    parser.add_argument("ekf")
    args = parser.parse_args()
    imu = np.loadtxt(args.imu, delimiter=",", skiprows=1, usecols=range(13))
    ekf = np.loadtxt(args.ekf, delimiter=",", skiprows=1, usecols=range(18))
    time, rotor, acceleration = imu[:, 0], imu[:, 8:12], -imu[:, 7]

    # Flight segment before the crash. Reject implausible rotor readings and IMU spikes.
    bins = np.linspace(55., 69.8, 741)
    slots = np.digitize(time, bins)
    good = np.all((rotor > 500) & (rotor < 4500), axis=1) & (abs(acceleration) < 70)
    good &= (slots > 0) & (slots < len(bins))
    selected = slots[good]
    count = np.bincount(selected, minlength=len(bins))
    valid = count >= 12

    def average(values):
        sums = np.bincount(selected, values[good], minlength=len(bins))
        return sums[valid] / count[valid]

    t = average(time)
    omega_squared = average((rotor ** 2).sum(axis=1))
    omega_mean = average(rotor.mean(axis=1))
    measured = average(acceleration)
    vx, vy, vz, roll, pitch, yaw = [
        np.interp(t, ekf[:, 0], ekf[:, i]) for i in range(5, 11)
    ]
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    airspeed_up = -(vx * (cy * sp * cr + sy * sr) + vy * (sy * sp * cr - cy * sr) + vz * cp * cr)
    horizontal = np.sqrt(np.maximum(vx**2 + vy**2 + vz**2 - airspeed_up**2, 0))

    lib = load_sim()
    model = lib.sim_axial_specific_force
    model.argtypes = [ct.c_float] * 4
    model.restype = ct.c_float
    predicted = np.array([model(*inputs)
                          for inputs in zip(omega_squared, omega_mean, airspeed_up, horizontal)])
    next_half = t >= 62.4
    prior = (4.8e-7 * omega_squared - (0.16 + 0.075 * abs(airspeed_up)) * airspeed_up) / 0.966
    for name, estimate in (("previous nominal", prior), ("calibrated Rust", predicted)):
        error = (estimate - measured)[next_half]
        print(f"{name}: {len(error)} windows, RMSE {np.sqrt((error**2).mean()):.2f} m/s², "
              f"bias {error.mean():+.2f} m/s², peak error {abs(error).max():.2f} m/s²")
    print(f"Reference EKF peak speed in selected windows: {np.hypot(np.hypot(vx, vy), vz).max():.2f} m/s")


if __name__ == "__main__":
    main()
