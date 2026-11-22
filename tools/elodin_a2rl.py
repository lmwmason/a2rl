"""Run the A2RL reference course and airframe in the pinned Elodin/Betaflight harness."""
from dataclasses import replace
import ctypes as ct
from math import cos, radians, sin
import os
from pathlib import Path
import runpy
import sys
import tomllib


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def configure_plant(physics, profile):
    """Use the A2RL plant coefficients with native Betaflight motor commands."""
    import elodin as el
    import jax.numpy as jnp

    def motor_system(config):
        maximum = config.motor_max_thrust
        alpha = 1. - jnp.exp(-config.dt / config.motor_time_constant)

        @el.map
        def actuate(command: physics.MotorCommand, previous: physics.MotorThrust) -> physics.MotorThrust:
            # Apply first-order lag in rotor speed, then use the calibrated squared-speed law.
            speed = jnp.sqrt(jnp.maximum(previous / maximum, 0.))
            target = jnp.sqrt(jnp.clip(command, 0., 1.))
            return maximum * (speed + alpha * (target - speed)) ** 2

        return actuate

    def thrust_system(config):
        torque_arms = jnp.array(config.motor_torque_axes)
        spin = jnp.array(config.motor_spin_directions)
        coefficient = profile["motor_thrust_coefficient"]
        rotor_limit = profile["motor_limit_rad_s"]

        @el.map
        def thrust(nominal: physics.MotorThrust, pose: el.WorldPos, motion: el.WorldVel) -> physics.BodyThrust:
            v = pose.angular().inverse() @ motion.linear()
            speed = rotor_limit * jnp.sqrt(jnp.maximum(nominal / config.motor_max_thrust, 0.)).mean()
            tip = jnp.maximum(.0485775 * speed, 1.)
            axial = jnp.arctan(-v[2] / tip)
            crossflow = jnp.arctan((v[0] ** 2 + v[1] ** 2) / tip ** 2)
            kt = jnp.maximum(coefficient + 4.492359e-6 * axial + 3.180793e-6 * crossflow, .05 * coefficient)
            rotor_thrust = nominal * (kt / coefficient)
            torque = jnp.sum(rotor_thrust[:, None] * torque_arms, axis=0)
            torque += jnp.array([0., 0., config.motor_torque_coeff * (rotor_thrust @ spin)])
            return el.SpatialForce(linear=jnp.array([0., 0., rotor_thrust.sum()]), torque=torque)

        return thrust

    def drag_system(config):
        linear = jnp.array(config.linear_drag)
        quadratic = jnp.array(profile["quadratic_drag"])

        @el.map
        def drag(pose: el.WorldPos, motion: el.WorldVel) -> physics.BodyDrag:
            v = pose.angular().inverse() @ motion.linear()
            return pose.angular() @ (-(linear + quadratic * jnp.abs(v)) * v)

        return drag

    def forces_system(config):
        angular = jnp.array(config.angular_drag)
        gravity = jnp.array([0., 0., -config.gravity])

        @el.map
        def forces(thrust: physics.BodyThrust, drag: physics.BodyDrag, pose: el.WorldPos,
                   motion: el.WorldVel, inertia: el.Inertia, force: el.Force) -> el.Force:
            quat = pose.angular()
            damping = quat @ (-angular * (quat.inverse() @ motion.angular()))
            return force + quat @ thrust + el.SpatialForce(
                linear=gravity * inertia.mass() + drag, torque=damping)

        return forces

    physics.create_motor_dynamics = motor_system
    physics.create_body_thrust_system = thrust_system
    physics.create_drag_system = drag_system
    physics.create_apply_forces_system = forces_system


def configure_course(course, lib, season2, asset, finish_capture):
    import numpy as np

    pointer = ct.POINTER(ct.c_float)
    lib.sim_course_gate_pose.argtypes = [ct.c_bool, ct.c_size_t, pointer]
    lib.sim_course_gate_pose.restype = ct.c_bool
    lib.sim_course_segment.argtypes = [ct.c_bool, pointer, pointer, ct.c_size_t]
    lib.sim_course_segment.restype = ct.c_int
    lib.sim_course_environment.argtypes = [ct.c_bool, pointer]
    lib.sim_course_environment.restype = None
    geometry = (ct.c_float * 10)()
    lib.sim_course_environment(season2, geometry)
    start, heading = geometry[:3], geometry[3]
    low, high = np.asarray(geometry[4:7]), np.asarray(geometry[7:10])
    low[:2] += .205
    high -= .205
    n = lib.sim_course_gate_count(season2)
    frames = []
    for i in range(n + 1):
        pose = (ct.c_float * 4)()
        if lib.sim_course_gate_pose(season2, i, pose):
            frames.append(course.Gate(i, tuple(pose[:3]), pose[3]))
        elif i < n:
            raise RuntimeError(f"Cannot read A2RL gate {i}")
    course.EASY_COURSE = tuple(replace(frame, index=lap * n + frame.index)
                               for lap in range(2) for frame in frames[:n])
    spawn = course.spawn_gates
    course.spawn_gates = lambda world, path: spawn(world, tuple(frames))
    pitch = 0.0 if season2 else 90.0
    course.schematic_for = lambda path: "\n".join(
        f'object_3d gate_{gate.index}.world_pos {{\n'
        f'    glb path="{asset}" rotate="(0.0, {pitch}, {gate.yaw_deg})"\n'
        f'}}' for gate in frames)
    previous, current = (ct.c_float * 3)(), (ct.c_float * 3)()
    contact = False

    def detect(path, last, start, end):
        nonlocal contact
        target = last + 1
        if contact or target >= len(path):
            return None
        previous[:], current[:] = start, end
        if not np.isfinite(end).all() or not np.isfinite(start).all():
            contact = True
            print("[A2RL] invalid rigid-body state", flush=True)
            return None
        hit = lib.sim_course_segment(season2, previous, current, target % n)
        outside = np.any(np.asarray(end) < low) or np.any(np.asarray(end) > high)
        if hit < 0 or outside:
            contact = True
            print(f"[A2RL] contact after {target} gates", flush=True)
        return target if hit == 1 and not contact else None

    def summary(path, last, times, end):
        finished = last + 1 == len(path)
        official = times[last] - times[0] if finished else None
        print(f"[A2RL] gates={last + 1}/{len(path)} contact={contact} official_seconds={official}", flush=True)
        finish_capture()

    course.detect_gate_pass = detect
    course.print_summary = summary
    return (lambda: contact), start, heading


def main():
    if not os.environ.get("A2RL_ELODIN_ROOT"):
        raise SystemExit("Set A2RL_ELODIN_ROOT to the pinned Elodin practice harness checkout.")
    root = Path(os.environ["A2RL_ELODIN_ROOT"]).expanduser().resolve()
    source = root / "sim/main.py"
    if not source.is_file():
        raise SystemExit(f"Elodin harness not found at {source}.")
    with Path(os.environ.get("A2RL_AIRFRAME", ROOT / "assets/a2rl-2025-airframe.toml")).open("rb") as file:
        profile = tomllib.load(file)
    sys.path.insert(0, str(root))

    import elodin as el
    import numpy as np
    from sim import camera, config, course, physics
    from training.sim import course_profile, load_sim
    from tools import elodin_policy

    cfg = config.DEFAULT_CONFIG
    course_name = course_profile()
    season2 = course_name == "2026-practice"
    asset = "gate.glb"
    if season2:
        from tools.elodin_gate import write_gate

        asset = "a2rl-season2-gate.glb"
        assets = Path(os.environ.get("ELODIN_ASSETS_DIR", root / "assets")).resolve()
        os.environ["ELODIN_ASSETS_DIR"] = str(assets)
        write_gate(assets / asset)
    lib = load_sim()
    path = os.environ.get("A2RL_ELODIN_FPV")
    if path or os.environ.get("A2RL_ELODIN_POLICY"):
        if not hasattr(lib, "sim_render_pose"):
            raise RuntimeError("Rebuild the simulator library for the shared FPV camera.")
        elodin_policy.camera = elodin_policy.CameraFeed(
            lib, Path(path).expanduser().resolve() if path else None, int(cfg.fpv_rate),
            float(os.environ.get("A2RL_FPV_BREAKUP", "0")),
            float(os.environ.get("A2RL_FPV_EXPOSURE", "1")),
            int(os.environ.get("A2RL_FPV_SEED", "1")),
        )
    finish_capture = elodin_policy.camera.close if elodin_policy.camera else lambda: None
    elodin_policy.contact, start, heading = configure_course(course, lib, season2, asset, finish_capture)
    if os.environ.get("A2RL_ELODIN_FLOW_TEACHER") == "1":
        if os.environ.get("A2RL_ELODIN_POLICY"):
            raise ValueError("Select one pilot for Elodin validation.")
        elodin_policy.configure_expert(lib, season2)
    heading = radians(heading)
    cfg.mass = profile["mass_kg"]
    cfg.inertia_diagonal = np.array(profile["inertia_kg_m2"])
    cfg.arm_length = profile["motor_arm_m"]
    cfg.motor_max_thrust = profile["motor_thrust_coefficient"] * profile["motor_limit_rad_s"] ** 2
    cfg.motor_time_constant = profile["motor_time_constant_s"]
    cfg.motor_torque_coeff = profile["motor_yaw_torque_m"]
    cfg.linear_drag = np.array(profile["linear_drag"])
    cfg.angular_drag = np.array(profile["angular_drag"])
    cfg.gravity = 9.80665
    cfg.initial_position = np.array([*start[:2], .205])
    cfg.initial_quaternion = np.array([0., 0., sin(heading / 2), cos(heading / 2)])
    cfg.ground_level = .205
    cfg.simulation_time = float(os.environ.get("A2RL_ELODIN_SECONDS", "120"))
    if cfg.simulation_time < 1.:
        raise ValueError("A2RL_ELODIN_SECONDS must be at least 1 second.")
    configure_plant(physics, profile)
    elodin_policy.hover = cfg.hover_throttle
    camera.CAM_WIDTH, camera.CAM_HEIGHT = profile["camera_width"], profile["camera_height"]
    def register_camera(world, drone):
        world.sensor_camera(entity=drone, name=camera.CAMERA_NAME, width=camera.CAM_WIDTH,
                            height=camera.CAM_HEIGHT, fov=profile["camera_vertical_fov_deg"],
                            near=.02, far=120., pos_offset=[.10, 0., 0.],
                            rot_offset=[0., -profile["camera_up_tilt_deg"], 0.], format="rgba",
                            create_frustum=True, fps=cfg.fpv_rate)
        return camera.DB_NAME

    if elodin_policy.camera is not None and os.environ.get("A2RL_ELODIN_PREVIEW") != "1":
        camera.register = lambda world, drone: camera.DB_NAME
        camera.collect_frame = lambda ctx, name: None
    else:
        camera.register = register_camera
    # Elodin 0.17.3's render server publishes cameras at their configured FPS.
    camera.request_render = lambda ctx, name: None
    print(f"[A2RL] {course_name}; {profile['name']}: {cfg.mass:.3f} kg, "
          f"{cfg.motor_max_thrust:.3f} N/motor", flush=True)
    os.environ["RACE_SOLVER"] = "tools.elodin_policy"
    if policy := os.environ.get("A2RL_ELODIN_POLICY"):
        os.environ["A2RL_ELODIN_POLICY"] = str(Path(policy).expanduser().resolve(strict=True))
    # Upstream still owns its SITL recipe. Skip only its broad process cleanup.
    world_run = el.World.run
    world_schematic = el.World.schematic

    def schematic(world, content, name):
        # The upstream Crazyflie preview mesh would block this aircraft's FPV image.
        first = content.index("    object_3d drone.world_pos {")
        last = content.index("    vector_arrow", first)
        content = (content[:first] + content[last:]).replace(
            "FPV Camera (640x360)", "Elodin preview")
        return world_schematic(world, content, name)

    def run(world, *args, **kwargs):
        sys.argv.remove("--no-s10")
        sys.argv[0] = str(Path(__file__).resolve())
        if database := os.environ.get("A2RL_ELODIN_DB"):
            kwargs["db_path"] = str(Path(database).expanduser().resolve())
        kwargs["backend"] = "jax-cpu"
        return world_run(world, *args, **kwargs)

    el.World.run = run
    el.World.schematic = schematic
    sys.argv.append("--no-s10")
    os.chdir(root)
    runpy.run_path(str(source), run_name="__main__")


if __name__ == "__main__":
    main()
