// Python owns the image buffer and one exclusive handle per environment.
use crate::{
    axial_specific_force, camera::Camera, mode2_acro, teacher::Teacher, Race, Status, DT, GRAVITY,
    HOVER_STICK, INERTIA, LAYOUT, MASS_KG, PHYSICS_MODEL_VERSION,
};

pub struct Environment {
    camera: Camera,
    race: Race,
    teacher: Teacher,
}

#[no_mangle]
pub extern "C" fn sim_gate_count() -> usize {
    LAYOUT.len()
}

#[no_mangle]
pub extern "C" fn sim_hover_stick() -> f32 {
    HOVER_STICK
}

#[no_mangle]
pub extern "C" fn sim_physics_model_version() -> u32 {
    PHYSICS_MODEL_VERSION
}

#[no_mangle]
pub unsafe extern "C" fn sim_horizontal_specific_force(x: f32, y: f32, result: *mut [f32; 2]) {
    let force = crate::drag(glam::Vec3::new(x, y, 0.)) / MASS_KG;
    *result = [force.x, force.y];
}

#[no_mangle]
pub extern "C" fn sim_axial_specific_force(
    squared_sum: f32,
    mean: f32,
    axial: f32,
    horizontal: f32,
) -> f32 {
    axial_specific_force(squared_sum, mean, axial, horizontal)
}

#[no_mangle]
pub extern "C" fn sim_create(width: usize, height: usize) -> *mut Environment {
    if !(1..=820).contains(&width) || !(1..=616).contains(&height) {
        return std::ptr::null_mut();
    }
    let race = Race::default();
    let teacher = Teacher::new(&race, false);
    Box::into_raw(Box::new(Environment {
        camera: Camera::new(width, height),
        race,
        teacher,
    }))
}

#[no_mangle]
pub unsafe extern "C" fn sim_destroy(env: *mut Environment) {
    if !env.is_null() {
        drop(Box::from_raw(env));
    }
}

#[no_mangle]
pub unsafe extern "C" fn sim_reset(
    env: *mut Environment,
    gate: i32,
    side: f32,
    height: f32,
    exposure: f32,
    video_breakup: f32,
    video_seed: u32,
    image: *mut u8,
) {
    let env = &mut *env;
    env.race = if gate == -3 {
        let mut race = Race::grand_challenge();
        race.drone.position.y += side;
        race.exposure = exposure;
        race
    } else if gate == -2 {
        Race::ground_start(side, exposure)
    } else if gate < 0 {
        Race::from_start(side, height, exposure)
    } else {
        Race::practice(gate as usize, side, height, exposure)
    };
    env.teacher = Teacher::new(&env.race, false);
    env.race.video_breakup = video_breakup;
    env.race.video_seed = video_seed;
    env.camera.render(
        &env.race,
        std::slice::from_raw_parts_mut(image, env.camera.byte_len()),
    );
}

#[no_mangle]
pub unsafe extern "C" fn sim_reset_after_gate(
    env: *mut Environment,
    gate: usize,
    side: f32,
    height: f32,
    speed: f32,
    exposure: f32,
    video_breakup: f32,
    video_seed: u32,
    image: *mut u8,
) {
    let env = &mut *env;
    env.race = Race::after_gate(gate, side, height, speed, exposure);
    env.teacher = Teacher::new(&env.race, true);
    env.race.video_breakup = video_breakup;
    env.race.video_seed = video_seed;
    env.camera.render(
        &env.race,
        std::slice::from_raw_parts_mut(image, env.camera.byte_len()),
    );
}

#[no_mangle]
pub unsafe extern "C" fn sim_expert_action(env: *mut Environment, out: *mut [f32; 4]) {
    let env = &mut *env;
    *out = env.teacher.action(&env.race);
}

#[no_mangle]
pub unsafe extern "C" fn sim_expert_features(env: *const Environment, out: *mut [f32; 12]) {
    let env = &*env;
    let drone = &env.race.drone;
    let inverse = drone.rotation.conjugate();
    let target = env.race.gates[env.race.next_gate()].center - drone.position;
    let vectors = [
        (inverse * target).normalize_or_zero(),
        inverse * glam::Vec3::Z,
        (inverse * drone.velocity / 10.).clamp_length_max(2.),
        (drone.omega / 5.).clamp_length_max(2.),
    ];
    for (i, value) in vectors.into_iter().enumerate() {
        (&mut *out)[i * 3..i * 3 + 3].copy_from_slice(&value.to_array());
    }
}

#[no_mangle]
pub unsafe extern "C" fn sim_step(
    env: *mut Environment,
    action: *const [f32; 4],
    image: *mut u8,
    reward: *mut f32,
) -> i32 {
    let env = &mut *env;
    let Some(value) = env.race.step(mode2_acro(*action)) else {
        return -1;
    };
    write_step(env, image, reward, value)
}

type RigidBodyStep = unsafe extern "C" fn(*const [f32; 13], *const [f32; 6], *mut [f32; 13]) -> i32;

#[no_mangle]
pub unsafe extern "C" fn sim_rigid_parameters(output: *mut [f32; 6]) {
    *output = [DT, MASS_KG, INERTIA.x, INERTIA.y, INERTIA.z, GRAVITY];
}

#[no_mangle]
pub unsafe extern "C" fn sim_rigid_state(env: *const Environment, output: *mut [f32; 13]) {
    *output = (&*env).race.drone.rigid_state();
}

#[no_mangle]
pub unsafe extern "C" fn sim_step_external(
    env: *mut Environment,
    action: *const [f32; 4],
    image: *mut u8,
    reward: *mut f32,
    integrate: RigidBodyStep,
) -> i32 {
    let env = &mut *env;
    let Some(value) = env.race.step_with(mode2_acro(*action), |drone, input| {
        let state = drone.rigid_state();
        let (force, torque) = drone.body_wrench(input);
        let torque = torque - drone.angular_damping();
        let wrench = [force.x, force.y, force.z, torque.x, torque.y, torque.z];
        let mut result = [0.; 13];
        integrate(&state, &wrench, &mut result) == 0 && drone.set_rigid_state(result)
    }) else {
        return -1;
    };
    write_step(env, image, reward, value)
}

unsafe fn write_step(env: &mut Environment, image: *mut u8, reward: *mut f32, value: f32) -> i32 {
    *reward = value;
    env.camera.render(
        &env.race,
        std::slice::from_raw_parts_mut(image, env.camera.byte_len()),
    );
    match env.race.status {
        Status::Racing => 0,
        Status::Timeout => 2,
        Status::Stalled => 3,
        Status::Finished => 4,
        _ => 1,
    }
}

#[no_mangle]
pub unsafe extern "C" fn sim_official_seconds(env: *const Environment) -> f32 {
    (&*env).race.official_seconds().unwrap_or(f32::NAN)
}

#[no_mangle]
pub unsafe extern "C" fn sim_lap_stats(env: *const Environment, gates: *mut u32, best: *mut f32) {
    let env = &*env;
    *gates = env.race.passed as u32;
    *best = env.race.best_lap.unwrap_or(f32::NAN);
}
