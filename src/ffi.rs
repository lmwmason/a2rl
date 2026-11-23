// Python owns the image buffer and one exclusive handle per environment.
use crate::{
    axial_specific_force, camera::Camera, teacher::Teacher, Course, Gate, Race, LAYOUT, MASS_KG,
};
use glam::Vec3;

pub struct RenderScene {
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
    let course = course_from_flag(season2);
    segment(
        course,
        Gate::on_course(course).chain(course.extra_frames().iter().copied()),
        Vec3::from_array(*previous),
        Vec3::from_array(*current),
        target,
    )
}

#[no_mangle]
pub unsafe extern "C" fn sim_set_gate_poses(
    env: *mut RenderScene,
    poses: *const [f32; 4],
    count: usize,
) -> bool {
    let race = &mut (*env).race;
    if poses.is_null() || count != race.gates.len() {
        return false;
    }
    let poses = std::slice::from_raw_parts(poses, count);
    if !poses.iter().flatten().all(|value| value.is_finite()) {
        return false;
    }
    race.gates = poses.iter().copied().map(Gate::from_pose).collect();
    true
}

#[no_mangle]
pub unsafe extern "C" fn sim_scene_segment(
    env: *const RenderScene,
    previous: *const [f32; 3],
    current: *const [f32; 3],
    target: usize,
) -> i32 {
    let race = &(*env).race;
    segment(
        race.course,
        race.gate_frames().copied(),
        Vec3::from_array(*previous),
        Vec3::from_array(*current),
        target,
    )
}

fn segment(
    course: Course,
    gates: impl Iterator<Item = Gate>,
    a: Vec3,
    b: Vec3,
    target: usize,
) -> i32 {
    let mut passed = false;
    for (index, gate) in gates.enumerate() {
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
pub unsafe extern "C" fn sim_set_season2_course(env: *mut RenderScene, season2: bool) {
    (*env).course = course_from_flag(season2);
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
pub extern "C" fn sim_create(width: usize, height: usize) -> *mut RenderScene {
    if !(1..=820).contains(&width) || !(1..=616).contains(&height) {
        return std::ptr::null_mut();
    }
    let race = Race::default();
    let teacher = Teacher::new(&race, false);
    Box::into_raw(Box::new(RenderScene {
        camera: Camera::new(width, height),
        race,
        teacher,
        flow_teacher: false,
        course: Course::default(),
    }))
}

#[no_mangle]
pub unsafe extern "C" fn sim_destroy(env: *mut RenderScene) {
    if !env.is_null() {
        drop(Box::from_raw(env));
    }
}

#[no_mangle]
pub unsafe extern "C" fn sim_configure_scene(
    env: *mut RenderScene,
    camera_tilt: f32,
    exposure: f32,
    video_breakup: f32,
    video_seed: u32,
) {
    let env = &mut *env;
    env.race = Race::grand_challenge_on(env.course);
    env.race.camera_tilt = camera_tilt;
    env.race.exposure = exposure;
    env.race.video_breakup = video_breakup;
    env.race.video_seed = video_seed;
    env.teacher = Teacher::new(&env.race, false).with_flow(env.flow_teacher);
}

#[no_mangle]
pub unsafe extern "C" fn sim_set_flow_teacher(env: *mut RenderScene, enabled: bool) {
    let env = &mut *env;
    env.flow_teacher = enabled;
}

#[no_mangle]
pub unsafe extern "C" fn sim_set_teacher_pace(env: *mut RenderScene, pace: f32) -> bool {
    (*env).teacher.set_pace(pace)
}

#[no_mangle]
pub unsafe extern "C" fn sim_render_pose(
    env: *mut RenderScene,
    state: *const [f32; 13],
    elapsed: f64,
    image: *mut u8,
) -> bool {
    sim_render_heat_pose(env, state, elapsed, std::ptr::null(), 0, image)
}

#[no_mangle]
pub unsafe extern "C" fn sim_render_heat_pose(
    env: *mut RenderScene,
    state: *const [f32; 13],
    elapsed: f64,
    opponents: *const [f32; 3],
    opponent_count: usize,
    image: *mut u8,
) -> bool {
    render_pose(env, state, elapsed, opponents, opponent_count, false, image)
}

#[no_mangle]
pub unsafe extern "C" fn sim_render_chase_pose(
    env: *mut RenderScene,
    state: *const [f32; 13],
    elapsed: f64,
    opponents: *const [f32; 3],
    opponent_count: usize,
    image: *mut u8,
) -> bool {
    render_pose(env, state, elapsed, opponents, opponent_count, true, image)
}

unsafe fn render_pose(
    env: *mut RenderScene,
    state: *const [f32; 13],
    elapsed: f64,
    opponents: *const [f32; 3],
    opponent_count: usize,
    chase: bool,
    image: *mut u8,
) -> bool {
    let env = &mut *env;
    if !elapsed.is_finite()
        || elapsed < 0.
        || opponent_count > 2
        || !env.race.drone.set_rigid_state(*state)
    {
        return false;
    }
    let mut peers = [Vec3::ZERO; 2];
    if opponent_count > 0 {
        for (peer, position) in peers
            .iter_mut()
            .zip(std::slice::from_raw_parts(opponents, opponent_count))
        {
            *peer = Vec3::from_array(*position);
            if !peer.is_finite() {
                return false;
            }
        }
    }
    env.race.ticks = (elapsed * crate::VIDEO_CLOCK_HZ as f64).round() as usize;
    env.camera.render_with_peers(
        &env.race,
        &peers[..opponent_count],
        chase,
        std::slice::from_raw_parts_mut(image, env.camera.byte_len()),
    );
    true
}

#[no_mangle]
pub unsafe extern "C" fn sim_external_expert_action(
    env: *mut RenderScene,
    state: *const [f32; 13],
    passed: usize,
    output: *mut [f32; 4],
) -> bool {
    let env = &mut *env;
    if passed >= env.race.gates.len() * 2 || !env.race.drone.set_rigid_state(*state) {
        return false;
    }
    env.race.passed = passed;
    *output = env.teacher.external_action(&env.race);
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shifted_gate_scene_updates_image_and_contact() {
        let env = sim_create(96, 72);
        let state = [15.7, 2., 1.47, 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.];
        let previous = [13.5, 2.65, 1.47];
        let current = [11.5, 2.65, 1.47];
        let mut nominal = vec![0; 96 * 72 * 3];
        let mut shifted = nominal.clone();
        let mut poses = crate::LAYOUT_2026;
        poses[0][1] += 0.25;
        unsafe {
            sim_set_season2_course(env, true);
            sim_configure_scene(env, 0., 1., 0., 1);
            assert_eq!(sim_scene_segment(env, &previous, &current, 0), -1);
            assert!(sim_render_pose(env, &state, 1., nominal.as_mut_ptr()));
            assert!(sim_set_gate_poses(env, poses.as_ptr(), poses.len()));
            assert_eq!(sim_scene_segment(env, &previous, &current, 0), 1);
            assert_eq!(sim_course_segment(true, &previous, &current, 0), -1);
            assert!(sim_render_pose(env, &state, 1., shifted.as_mut_ptr()));
            sim_destroy(env);
        }
        assert_ne!(nominal, shifted);
    }
}
