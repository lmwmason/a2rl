"""Compare Rust and MuJoCo with identical Mode 2 sticks and vehicle parameters."""
import argparse
import json
import math

import numpy as np

from training.betaflight_link import decode_rc, encode_rc
from training.sim import Environment, load_sim


def compare(lib, name, commands):
    rust = Environment(lib, 16, 12, physics="rust")
    mujoco = Environment(lib, 16, 12, physics="mujoco")
    position, velocity, angle, rate = [], [], [], []
    status = (0, 0)
    try:
        for env in (rust, mujoco):
            env.reset()
        for action in commands:
            status = (rust.advance(action)[2], mujoco.advance(action)[2])
            a, b = rust.rigid_state(), mujoco.rigid_state()
            position.append(float(np.linalg.norm(a[:3] - b[:3])))
            velocity.append(float(np.linalg.norm(a[3:6] - b[3:6])))
            rate.append(float(np.linalg.norm(a[10:] - b[10:])))
            p = a[6:10].astype(np.float64)
            q = b[6:10].astype(np.float64)
            cosine = abs(float(p @ q / (np.linalg.norm(p) * np.linalg.norm(q))))
            angle.append(math.degrees(2 * math.acos(min(1., cosine))))
            if any(status):
                break
        return {
            "profile": name, "steps": len(position), "status": status,
            "max_position_error_m": max(position),
            "max_speed_error_m_s": max(velocity),
            "max_attitude_error_deg": max(angle),
            "max_rate_error_rad_s": max(rate),
        }
    finally:
        rust.close()
        mujoco.close()


def course(lib, engine, msp=False):
    env = Environment(lib, 16, 12, physics=engine)
    try:
        env.official_start()
        while True:
            action = env.expert_action()
            if msp:
                action = decode_rc(encode_rc(action))
            _, _, status = env.advance(action)
            if status:
                return {"engine": engine, "status": status, "gates": env.lap_stats()[0],
                        "official_s": env.official_seconds() if status == 4 else None}
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--course", action="store_true", help="Run the state-driven teacher on both backends.")
    parser.add_argument("--msp", action="store_true", help="Send each command through the MSP v1 RC adapter.")
    args = parser.parse_args()
    lib = load_sim()
    hover = lib.sim_hover_stick()
    profiles = {
        "hover": [[0., hover, 0., 0.]] * 60,
        "throttle pulse": [[0., hover + .1, 0., 0.]] * 12 + [[0., hover, 0., 0.]] * 18,
        "roll reversal": [[0., hover, .2, 0.]] * 12 + [[0., hover, -.2, 0.]] * 12,
        "pitch reversal": [[0., hover, 0., -.2]] * 12 + [[0., hover, 0., .2]] * 12,
        "combined turn": [[.25, hover + .05, -.2, .18]] * 10 + [[-.15, hover, .1, -.21]] * 14,
        "free fall": [[0., -1., 0., 0.]] * 8,
    }
    if args.msp:
        profiles = {name: [decode_rc(encode_rc(action)) for action in commands]
                    for name, commands in profiles.items()}
    results = [compare(lib, name, commands) for name, commands in profiles.items()]
    print(json.dumps(results, indent=2), flush=True)
    if any(result["status"][0] != result["status"][1]
           or result["max_position_error_m"] > .001
           or result["max_speed_error_m_s"] > .005
           or result["max_attitude_error_deg"] > .02 for result in results):
        raise SystemExit("Rust and MuJoCo exceeded the shared-input comparison tolerances.")
    if args.course:
        races = [course(lib, engine, args.msp) for engine in ("rust", "mujoco")]
        print(json.dumps(races, indent=2))
        if any(race["status"] != 4 or race["gates"] != 2 * lib.sim_gate_count() for race in races):
            raise SystemExit("The teacher did not complete both laps on both engines.")


if __name__ == "__main__":
    main()
