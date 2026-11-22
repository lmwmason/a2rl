// Python owns the image buffer and one exclusive handle per environment.
use crate::{
    axial_specific_force, camera::Camera, mode2_acro, teacher::Teacher, Course, Race,
    RewardProfile, Status, HOVER_STICK, LAYOUT, MASS_KG, PACE_REWARD_VERSION,
    PHYSICS_MODEL_VERSION, REWARD_MODEL_VERSION,
};

pub struct Environment {
    camera: Camera,
    race: Race,
    teacher: Teacher,
    flow_teacher: bool,
    course: Course,
}

fn course_from_flag(season2: bool) -> Course {
    if season2 {
        Course::Season2Practice2026
    } else {
        Course::default()
    }
}
#[no_mangle]
pub extern "C" fn sim_gate_count() -> usize {
    LAYOUT.len()
}

#[no_mangle]
pub extern "C" fn sim_course_gate_count(season2: bool) -> usize {
    course_from_flag(season2).layout().len()
}

#[no_mangle]
pub extern "C" fn sim_course_gate_half_depth(season2: bool) -> f32 {
    course_from_flag(season2).gate_half_depth()
}

#[no_mangle]
pub extern "C" fn sim_course_camera_tilt(season2: bool) -> f32 {
    crate::camera::mount_angle(course_from_flag(season2))
}

#[no_mangle]
pub unsafe extern "C" fn sim_course_gate_pose(
    season2: bool,
    index: usize,
    out: *mut [f32; 4],
) -> bool {
    let course = course_from_flag(season2);
    if let Some(pose) = course.layout().get(index) {
        *out = *pose;
        true
    } else if let Some(frame) = course.extra_frames().get(index - course.layout().len()) {
        let [x, y, z] = frame.center.to_array();
        *out = [x, y, z, frame.normal.y.atan2(frame.normal.x).to_degrees()];
        true
    } else {
        false
    }
}

#[no_mangle]
pub unsafe extern "C" fn sim_course_segment(
    season2: bool,
    previous: *const [f32; 3],
    current: *const [f32; 3],
    target: usize,
) -> i32 {
    let a = glam::Vec3::from_array(*previous);
    let b = glam::Vec3::from_array(*current);
    let mut passed = false;
    let course = course_from_flag(season2);
    for (index, gate) in crate::Gate::on_course(course)
        .chain(course.extra_frames().iter().copied())
        .enumerate()
    {
        let (collision, crossed) = gate.progress(a, b, course);
        if collision {
            return -1;
        }
        passed |= crossed && index == target;
    }
    i32::from(passed)
}

#[no_mangle]
pub unsafe extern "C" fn sim_course_environment(season2: bool, out: *mut [f32; 10]) {
    let course = course_from_flag(season2);
    let (start, heading) = course.start();
    let (min, max) = course.bounds();
    *out = [
        start.x, start.y, start.z, heading, min.x, min.y, min.z, max.x, max.y, max.z,
    ];
}

#[no_mangle]
pub unsafe extern "C" fn sim_set_season2_course(env: *mut Environment, season2: bool) {
    (*env).course = course_from_flag(season2);
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
pub extern "C" fn sim_reward_model_version() -> u32 {
    REWARD_MODEL_VERSION
}

#[no_mangle]
pub extern "C" fn sim_pace_reward_version() -> u32 {
    PACE_REWARD_VERSION
}

#[no_mangle]
pub unsafe extern "C" fn sim_set_reward_profile(env: *mut Environment, pace: bool) {
    (&mut *env).race.reward_profile = if pace {
        RewardProfile::Pace
    } else {
        RewardProfile::Standard
    };
}

#[no_mangle]
pub unsafe extern "C" fn sim_flow_stats(env: *const Environment, output: *mut [f32; 2]) {
    let race = &(&*env).race;
    *output = [race.approach_delay_seconds, race.gate_speed_sum];
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
        flow_teacher: false,
        course: Course::default(),
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
    let reward_profile = env.race.reward_profile;
    env.race = if gate == -3 {
        let mut race = Race::grand_challenge_on(env.course);
        race.drone.position.y += side;
        race.exposure = exposure;
        race
    } else if gate == -2 {
        Race::ground_start_on(env.course, side, exposure)
    } else if gate < 0 {
        Race::from_start_on(env.course, side, height, exposure)
    } else {
        Race::practice_on(env.course, gate as usize, side, height, exposure)
    };
    env.race.reward_profile = reward_profile;
    env.teacher = Teacher::new(&env.race, false).with_flow(env.flow_teacher);
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
    let reward_profile = env.race.reward_profile;
    env.race = Race::after_gate_on(env.course, gate, side, height, speed, exposure);
    env.race.reward_profile = reward_profile;
    env.teacher = Teacher::new(&env.race, true).with_flow(env.flow_teacher);
    env.race.video_breakup = video_breakup;
    env.race.video_seed = video_seed;
    env.camera.render(
        &env.race,
        std::slice::from_raw_parts_mut(image, env.camera.byte_len()),
    );
}

#[no_mangle]
pub unsafe extern "C" fn sim_set_flow_teacher(env: *mut Environment, enabled: bool) {
    let env = &mut *env;
    env.flow_teacher = enabled;
}

#[no_mangle]
pub unsafe extern "C" fn sim_expert_action(env: *mut Environment, out: *mut [f32; 4]) {
    let env = &mut *env;
    *out = env.teacher.action(&env.race);
}

#[no_mangle]
pub unsafe extern "C" fn sim_sprint_window(env: *const Environment) -> bool {
    (&*env).race.sprint_window()
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

#[no_mangle]
pub unsafe extern "C" fn sim_rigid_state(env: *const Environment, output: *mut [f32; 13]) {
    *output = (&*env).race.drone.rigid_state();
}

#[no_mangle]
pub unsafe extern "C" fn sim_render_pose(
    env: *mut Environment,
    state: *const [f32; 13],
    elapsed: f64,
    image: *mut u8,
) -> bool {
    let env = &mut *env;
    if !elapsed.is_finite() || elapsed < 0. || !env.race.drone.set_rigid_state(*state) {
        return false;
    }
    env.race.ticks = (elapsed * crate::PHYSICS_HZ as f64).round() as usize;
    env.camera.render(
        &env.race,
        std::slice::from_raw_parts_mut(image, env.camera.byte_len()),
    );
    true
}

#[no_mangle]
pub unsafe extern "C" fn sim_external_expert_action(
    env: *mut Environment,
    state: *const [f32; 13],
    passed: usize,
    output: *mut [f32; 4],
) -> bool {
    let env = &mut *env;
    if passed >= env.race.gates.len() * 2 || !env.race.drone.set_rigid_state(*state) {
        return false;
    }
    env.race.passed = passed;
    *output = env.teacher.action(&env.race);
    true
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
