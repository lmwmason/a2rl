"""Reproduce the horizontal drag fit and evaluate independent MonoRace flights."""
import argparse
import ctypes as ct

import numpy as np

from training.sim import FLOAT_POINTER, load_sim


def windowed_samples(imu_path, ekf_path, start, stop):
    imu = np.loadtxt(imu_path, delimiter=",", skiprows=1, usecols=range(13))
    ekf = np.loadtxt(ekf_path, delimiter=",", skiprows=1, usecols=range(11))
    ekf = ekf[np.argsort(ekf[:, 0])]
    bins = start + .02 * np.arange(round((stop - start) / .02) + 1)
    clean = (imu[:, 0] >= bins[0]) & (imu[:, 0] < bins[-1])
    clean &= np.all((imu[:, 8:12] > 500) & (imu[:, 8:12] < 4500), axis=1)
    clean &= np.all(abs(imu[:, 5:8]) < 70, axis=1)
    slots = np.digitize(imu[clean, 0], bins)
    count = np.bincount(slots, minlength=len(bins))
    valid = count >= 12

    def average(values):
        return np.bincount(slots, values[clean], minlength=len(bins))[valid] / count[valid]

    t = average(imu[:, 0])
    if len(t) < 3:
        raise ValueError("Not enough synchronized data in the requested time window.")
    vx, vy, vz = [np.interp(t, ekf[:, 0], ekf[:, column]) for column in range(5, 8)]
    roll, pitch, yaw = [np.interp(t, ekf[:, 0], np.unwrap(ekf[:, column])) for column in range(8, 11)]
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    fore = vx * cy * cp + vy * sy * cp - vz * sp
    right = vx * (cy * sp * sr - sy * cr) + vy * (sy * sp * sr + cy * cr) + vz * cp * sr
    # Logged body axes are forward/right/down. The simulator uses forward/left/up.
    return np.column_stack((fore, -right)), np.column_stack((average(imu[:, 5]), -average(imu[:, 6])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", nargs=2, required=True, metavar=("IMU", "EKF"))
    parser.add_argument("--flight", action="append", nargs=5, required=True,
                        metavar=("NAME", "IMU", "EKF", "START", "STOP"))
    args = parser.parse_args()
    lib = load_sim()
    parameters = (ct.c_float * 6)()
    lib.sim_rigid_parameters(parameters)
    mass = parameters[1]
    force = lib.sim_horizontal_specific_force
    force.argtypes = [ct.c_float, ct.c_float, FLOAT_POINTER]
    force.restype = None
    output = (ct.c_float * 2)()

    speed, accel = windowed_samples(*args.calibration, 55., 62.4)
    x_linear, x_quadratic = np.linalg.lstsq(
        -np.column_stack((speed[:, 0], speed[:, 0] * abs(speed[:, 0]))) / mass,
        accel[:, 0], rcond=None,
    )[0]
    lateral_basis = -speed[:, 1] / mass
    y_linear = (lateral_basis @ accel[:, 1]) / (lateral_basis @ lateral_basis)
    print(f"Log 8 fit ({len(speed)} windows): {x_linear=:.7f}, {x_quadratic=:.7f}, {y_linear=:.7f}")
    for name, imu, ekf, start, stop in args.flight:
        speed, accel = windowed_samples(imu, ekf, float(start), float(stop))
        baseline = -(np.array([.09, .13]) + np.array([.045, .065]) * abs(speed)) * speed / mass
        prediction = np.empty_like(speed)
        for index, (x, y) in enumerate(speed):
            force(x, y, output)
            prediction[index] = output[:]
        before = np.sqrt(np.mean((baseline - accel)**2, axis=0))
        after = np.sqrt(np.mean((prediction - accel)**2, axis=0))
        print(f"{name} ({len(speed)} windows), RMSE in m/s²:"
              f" forward {before[0]:.2f} -> {after[0]:.2f}, lateral {before[1]:.2f} -> {after[1]:.2f}")


if __name__ == "__main__":
    main()
