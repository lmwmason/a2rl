"""Run the A2RL reference course and airframe in the pinned Elodin/Betaflight harness."""
from dataclasses import replace
import ctypes as ct
from functools import partial
import json
from math import cos, radians, sin
import os
from pathlib import Path
import runpy
import subprocess
import sys
from time import monotonic, sleep
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


def configure_course(course, lib, season2, asset, camera_feed, report_path):
    import numpy as np

    from tools import elodin_policy

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
    variant = json.loads(os.environ.get("A2RL_ELODIN_COURSE_VARIANT", "null"))
    check_segment = partial(lib.sim_course_segment, season2)
    if variant is not None:
        from training.course_layout import varied_gate_poses

        if not season2 or camera_feed is None or len(frames) != n:
            raise ValueError("Gate installation variation requires the 2026 RGB course.")
        poses = varied_gate_poses([(*gate.center, gate.yaw_deg) for gate in frames], **variant).astype(np.float32)
        frames = [course.Gate(i, tuple(map(float, pose[:3])), float(pose[3])) for i, pose in enumerate(poses)]
        camera_feed.env.place_gates(poses)
        if camera_feed.chase:
            camera_feed.chase.place_gates(poses)
        lib.sim_scene_segment.argtypes = [ct.c_void_p, pointer, pointer, ct.c_size_t]
        lib.sim_scene_segment.restype = ct.c_int
        check_segment = partial(lib.sim_scene_segment, camera_feed.env.handle)
    course.EASY_COURSE = tuple(replace(frame, index=lap * n + frame.index)
                               for lap in range(2) for frame in frames[:n])
    if camera_feed:
        camera_feed.course_centers = tuple(frame.center for frame in course.EASY_COURSE)
    spawn = course.spawn_gates
    course.spawn_gates = lambda world, path: spawn(world, tuple(frames))
    pitch = 0.0 if season2 else 90.0
    course.schematic_for = lambda path: "\n".join(
        f'object_3d gate_{gate.index}.world_pos {{\n'
        f'    glb path="{asset}" rotate="(0.0, {pitch}, {gate.yaw_deg})"\n'
        f'}}' for gate in frames)
    previous, current = (ct.c_float * 3)(), (ct.c_float * 3)()
    contact = False
    contact_kind = None
    stalled = False
    max_gate_gap = float(os.environ.get("A2RL_ELODIN_MAX_GATE_GAP", "0"))
    if not np.isfinite(max_gate_gap) or max_gate_gap < 0:
        raise ValueError("Maximum gate gap must be finite and nonnegative.")
    elapsed = 0.
    passed_times = []
    trace = None
    if os.environ.get("A2RL_ELODIN_REWARD_TRACE") == "1":
        from training.race_reward import GateTrace

        trace = GateTrace(start, course.EASY_COURSE)

    def detect(path, last, start, end):
        nonlocal contact, contact_kind
        target = last + 1
        if contact or elodin_policy.peer_contact or target >= len(path):
            return None
        previous[:], current[:] = start, end
        if not np.isfinite(end).all() or not np.isfinite(start).all():
            contact = True
            contact_kind = "invalid_state"
            print("[A2RL] invalid rigid-body state", flush=True)
            return None
        hit = check_segment(previous, current, target % n)
        outside = np.any(np.asarray(end) < low) or np.any(np.asarray(end) > high)
        if trace is not None:
            trace.observe(target, start, end, elapsed, crossed=hit == 1 and not outside)
        if hit < 0 or outside:
            contact = True
            contact_kind = "gate_frame" if hit < 0 else "arena_boundary"
            print(f"[A2RL] {contact_kind} contact after {target} gates", flush=True)
        if hit == 1 and not contact:
            passed_times.append(elapsed)
            return target
        return None

    def summary(path, last, times, end):
        crashed = contact or elodin_policy.peer_contact
        finished = last + 1 == len(path)
        official = times[last] - times[0] if finished else None
        print(f"[A2RL] gates={last + 1}/{len(path)} contact={crashed} official_seconds={official}", flush=True)
        if camera_feed:
            camera_feed.close()
            if (finished and not crashed) or (elodin_policy.shadow_expert and len(camera_feed.samples) >= 8):
                camera_feed.save_demonstration(last + 1, elodin_policy.shadow_expert and not elodin_policy.assisted_rollout)
        if report_path:
            from training.betaflight_link import timed_mean_throttle

            mean_throttle = timed_mean_throttle(camera_feed.actions, camera_feed.fps, times[0], times[last]) \
                if finished and camera_feed else None
            details = {
                "physics_engine": "elodin",
                "execution": "jax-in-process" if os.environ.get("A2RL_ELODIN_IN_PROCESS") == "1" else "jax-server",
                "course": "2026-practice" if season2 else "2025",
                "gates": last + 1,
                "gates_per_race": len(path),
                "contact": crashed,
                "contact_kind": contact_kind,
                "stalled": stalled,
                "finished": finished,
                "official_seconds": official,
                "mean_throttle_fraction": mean_throttle,
                "gate_times": times[:last + 1],
                "end_seconds": float(end),
                "video_start_seconds": camera_feed.first_frame_seconds if camera_feed else None,
                "policy_steps": elodin_policy.policy_steps,
                "camera_mount_pitch_deg": camera_feed.env.mount_pitch if camera_feed else None,
            }
            if trace is not None:
                details["reward_trace"] = trace.report()
            if variant is not None:
                details["course_variant"] = variant
            if elodin_policy.heat is not None:
                details.update(took_off=bool(elodin_policy.heat_airborne),
                               aircraft_contact=elodin_policy.peer_contact)
            if elodin_policy.assisted_rollout:
                details.update(assisted=True, student_influence_frames=elodin_policy.student_influence_frames,
                               reduced_assistance_frames=elodin_policy.reduced_assistance_frames)
            if elodin_policy.handoff is not None:
                details.update(handoff=elodin_policy.handoff,
                               handoff_frames=elodin_policy.handoff_frames)
            if elodin_policy.completion_profile:
                details["completion_profile"] = elodin_policy.completion_profile
            temporary = report_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(details, indent=2) + "\n")
            temporary.replace(report_path)

    def set_elapsed(seconds):
        nonlocal elapsed
        elapsed = seconds

    def finish_early():
        nonlocal stalled
        handoff = elodin_policy.handoff
        target_reached = handoff is not None and len(passed_times) >= handoff["target_gates"]
        scored_finish = trace is not None and len(passed_times) >= len(course.EASY_COURSE)
        last_crossing = passed_times[-1] if passed_times else 0.
        stalled = (max_gate_gap > 0 and not contact and len(passed_times) < len(course.EASY_COURSE)
                   and elapsed - last_crossing >= max_gate_gap)
        if not (contact or target_reached or scored_finish or stalled) or elodin_policy.heat is not None:
            return False
        summary(course.EASY_COURSE, len(passed_times) - 1, passed_times, elapsed)
        return True

    course.detect_gate_pass = detect
    course.print_summary = summary
    return (lambda: contact), start, heading, set_elapsed, finish_early


class InProcessContext:
    """Bridge the upstream lockstep callback to the in-process JAX state."""

    def __init__(self, simulation):
        self.simulation = simulation
        self.components = {name: index for _, index, name in simulation.map}
        self.slots = {}

    def indexes(self, pair):
        if pair not in self.slots:
            entity, component = pair.split(".", 1)
            try:
                slot = self.components[component]
                ident = self.simulation.entity_dict[entity]
                entities = self.simulation.component_entity_dict[component]
                row = entities.index(ident)
                if len(entities) == 1 and self.simulation.state[slot].ndim <= 1:
                    row = None
            except (KeyError, ValueError) as error:
                raise RuntimeError(f"Unknown simulation component {pair}") from error
            self.slots[pair] = slot, row
        return self.slots[pair]

    def read_component(self, pair):
        slot, row = self.indexes(pair)
        value = self.simulation.state[slot]
        return value if row is None else value[row]

    def component_batch_operation(self, reads):
        return {pair: self.read_component(pair) for pair in reads}

    def write_component(self, pair, value):
        import numpy as np

        slot, row = self.indexes(pair)
        component = np.array(self.simulation.state[slot], copy=True)
        if row is None:
            component[...] = value
        else:
            component[row] = value
        self.simulation.state[slot] = component


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

    offset = int(os.environ.get("A2RL_SITL_PORT_OFFSET", "0"))
    if not 0 <= offset <= 50000:
        raise ValueError("A2RL_SITL_PORT_OFFSET must be between 0 and 50000.")
    if offset:
        from sim import betaflight_bridge as bridge

        bridge.BetaflightSyncBridge = partial(bridge.BetaflightSyncBridge,
                                             state_port=bridge.PORT_STATE + offset,
                                             rc_port=bridge.PORT_RC + offset,
                                             pwm_port=bridge.PORT_PWM + offset)
    if socket_path := os.environ.get("A2RL_HEAT_SOCKET"):
        elodin_policy.connect_heat(socket_path, int(os.environ["A2RL_HEAT_PILOT"]))

    cfg = config.DEFAULT_CONFIG
    course_name = course_profile()
    season2 = course_name == "2026-practice"
    asset = "gate.glb"
    lib = load_sim()
    if season2:
        from tools.elodin_gate import write_gate

        asset = "a2rl-season2-gate.glb"
        assets = Path(os.environ.get("ELODIN_ASSETS_DIR", root / "assets")).resolve()
        assets.mkdir(parents=True, exist_ok=True)
        os.environ["ELODIN_ASSETS_DIR"] = str(assets)
        lib.sim_course_gate_half_depth.argtypes = [ct.c_bool]
        lib.sim_course_gate_half_depth.restype = ct.c_float
        write_gate(assets / asset, lib.sim_course_gate_half_depth(season2))
    path = os.environ.get("A2RL_ELODIN_FPV")
    teacher = os.environ.get("A2RL_ELODIN_FLOW_TEACHER") == "1"
    dataset = os.environ.get("A2RL_ELODIN_DEMONSTRATION")
    policy = os.environ.get("A2RL_ELODIN_POLICY")
    if (dataset or teacher) and json.loads(os.environ.get("A2RL_ELODIN_COURSE_VARIANT", "null")) is not None:
        raise ValueError("The fixed-route teacher cannot label shifted gate installations.")
    if dataset and (not (teacher or policy) or Path(dataset).suffix != ".npz"):
        raise ValueError("A2RL_ELODIN_DEMONSTRATION requires a pilot and a .npz path.")
    if path or dataset or os.environ.get("A2RL_ELODIN_POLICY"):
        if not hasattr(lib, "sim_render_pose"):
            raise RuntimeError("Rebuild the simulator library for the shared FPV camera.")
        pitch_override = os.environ.get("A2RL_FPV_MOUNT_PITCH")
        telemetry = os.environ.get("A2RL_ELODIN_TELEMETRY")
        elodin_policy.camera = elodin_policy.CameraFeed(
            lib, Path(path).expanduser().resolve() if path else None, int(cfg.fpv_rate),
            float(os.environ.get("A2RL_FPV_BREAKUP", "0")),
            float(os.environ.get("A2RL_FPV_EXPOSURE", "1")),
            int(os.environ.get("A2RL_FPV_SEED", "1")),
            Path(dataset).expanduser().resolve() if dataset else None,
            mount_pitch=float(pitch_override) if pitch_override is not None else None,
            chase=os.environ.get("A2RL_ELODIN_CHASE") == "1",
            telemetry=Path(telemetry) if telemetry else None,
            record_memory=os.environ.get("A2RL_ELODIN_MEMORY_OBSERVATIONS") == "1",
        )
    explicit_report = os.environ.get("A2RL_ELODIN_REPORT")
    report = (Path(explicit_report).expanduser().resolve() if explicit_report else
              Path(path).expanduser().resolve().with_suffix(".result.json") if path else None)
    elodin_policy.contact, start, heading, set_elapsed, finish_early = configure_course(
        course, lib, season2, asset, elodin_policy.camera, report,
    )
    if teacher and policy:
        raise ValueError("Select one pilot for Elodin validation.")
    elodin_policy.shadow_expert = bool(policy and dataset)
    elodin_policy.record_full_rollout = os.environ.get("A2RL_ELODIN_CORRECTION_SCOPE") == "rollout"
    elodin_policy.assisted_rollout = os.environ.get("A2RL_ELODIN_ASSISTED") == "1"
    elodin_policy.completion_profile = json.loads(os.environ.get("A2RL_ELODIN_COMPLETION_PROFILE", "{}"))
    if elodin_policy.assisted_rollout and not (elodin_policy.shadow_expert and elodin_policy.record_full_rollout):
        raise ValueError("Assisted rollouts require a policy and full-rollout correction recording.")
    handoff = elodin_policy.completion_profile.get("handoff")
    if handoff is not None:
        if (not elodin_policy.assisted_rollout or not 0 <= handoff["after_gates"] < handoff["target_gates"] <= len(course.EASY_COURSE)
                or os.environ.get("A2RL_ELODIN_IN_PROCESS") != "1" or not 0. <= handoff.get("delay_s", 0.) <= 1.):
            raise ValueError("Gate handoffs require an in-process assisted rollout with an ordered segment.")
        elodin_policy.handoff = handoff
    if teacher or elodin_policy.shadow_expert:
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
    side = float(os.environ.get("A2RL_ELODIN_START_SIDE", "0"))
    if not np.isfinite(side):
        raise ValueError("A2RL_ELODIN_START_SIDE must be finite.")
    cfg.initial_position += side * np.array([-sin(heading), cos(heading), 0.])
    if elodin_policy.heat:
        forward = float(os.environ.get("A2RL_HEAT_START_FORWARD", "0"))
        cfg.initial_position += forward * np.array([cos(heading), sin(heading), 0.])
    cfg.initial_quaternion = np.array([0., 0., sin(heading / 2), cos(heading / 2)])
    cfg.ground_level = .205
    cfg.simulation_time = float(os.environ.get("A2RL_ELODIN_SECONDS", "120"))
    if cfg.simulation_time < 1.:
        raise ValueError("A2RL_ELODIN_SECONDS must be at least 1 second.")
    configure_plant(physics, profile)
    elodin_policy.hover = cfg.hover_throttle
    elodin_policy.control_rate = cfg.fpv_rate
    camera.CAM_WIDTH, camera.CAM_HEIGHT = profile["camera_width"], profile["camera_height"]
    tilt = profile["camera_up_tilt_deg"]
    if season2:
        lib.sim_course_camera_tilt.argtypes = [ct.c_bool]
        lib.sim_course_camera_tilt.restype = ct.c_float
        tilt = lib.sim_course_camera_tilt(True)

    def register_camera(world, drone):
        world.sensor_camera(entity=drone, name=camera.CAMERA_NAME, width=camera.CAM_WIDTH,
                            height=camera.CAM_HEIGHT, fov=profile["camera_vertical_fov_deg"],
                            near=.02, far=120., pos_offset=[.10, 0., 0.],
                            rot_offset=[0., -tilt, 0.], format="rgba",
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
    world_recipe = el.World.recipe

    firmware = Path(os.environ.get("A2RL_SITL_EXECUTABLE", root / "betaflight/obj/main/betaflight_SITL.elf"))
    firmware = firmware.expanduser().resolve(strict=True)
    cli = ROOT / "assets/a2rl-betaflight-sitl.cli"
    work = Path(os.environ.get("A2RL_ELODIN_DB", ROOT / "runs/elodin/flight")).expanduser().resolve().parent
    work.mkdir(parents=True, exist_ok=True)
    launch = """dir=$(mktemp -d "$3/.betaflight-XXXXXX")
cd "$dir"
if ! "$1" --config "$2" > configuration.log 2>&1 ||
   grep -q '###ERROR IN' configuration.log ||
   ! test -s eeprom.bin; then
    cat configuration.log >&2
    exit 1
fi
exec "$1"
"""
    firmware_args = ["-eu", "-c", launch, "a2rl-sitl", str(firmware), str(cli), str(work)]

    def recipe(world, _original):
        return world_recipe(world, el.s10.PyRecipe.process(
            name="Betaflight SITL", cmd="sh", args=firmware_args, cwd=str(work),
        ))

    def in_process(world, system, *, simulation_rate, max_ticks, post_step, **_kwargs):
        simulation = world.to_jax(system, simulation_rate=simulation_rate, max_ticks=max_ticks)
        context = InProcessContext(simulation)
        pace = float(os.environ["A2RL_HEAT_PACE"]) if elodin_policy.heat else None
        epoch = None
        with subprocess.Popen(["sh", *firmware_args], cwd=work) as flight_controller:
            try:
                for tick in range(max_ticks):
                    if epoch is not None:
                        remaining = epoch + tick / (simulation_rate * pace) - monotonic()
                        if remaining > 0:
                            sleep(remaining)
                    if flight_controller.poll() is not None:
                        raise RuntimeError("Betaflight SITL exited during the rollout.")
                    simulation.step(1)
                    set_elapsed(tick / simulation_rate)
                    post_step(tick, context)
                    if tick < max_ticks - 1 and finish_early():
                        break
                    if pace is not None:
                        epoch = elodin_policy.heat_epoch
                    if elodin_policy.heat_failure is not None:
                        raise RuntimeError("The shared heat disconnected.") from elodin_policy.heat_failure
            finally:
                if flight_controller.poll() is None:
                    flight_controller.terminate()
                    try:
                        flight_controller.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        flight_controller.kill()

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
        if os.environ.get("A2RL_ELODIN_IN_PROCESS") == "1":
            return in_process(world, *args, **kwargs)
        kwargs["backend"] = "jax-cpu"
        return world_run(world, *args, **kwargs)

    el.World.run = run
    el.World.schematic = schematic
    el.World.recipe = recipe
    sys.argv.append("--no-s10")
    os.chdir(root)
    runpy.run_path(str(source), run_name="__main__")


if __name__ == "__main__":
    main()
