// Python owns the image buffer and one exclusive handle per environment.
use crate::{camera::Camera, Race, Status};

pub struct Environment {
    camera: Camera,
    race: Race,
}

#[no_mangle]
pub extern "C" fn sim_create(width: usize, height: usize) -> *mut Environment {
    if !(1..=820).contains(&width) || !(1..=616).contains(&height) {
        return std::ptr::null_mut();
    }
    Box::into_raw(Box::new(Environment {
        camera: Camera::new(width, height),
        race: Race::default(),
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
    image: *mut u8,
) {
    let env = &mut *env;
    env.race = if gate < 0 {
        Race::from_start(side, height, exposure)
    } else {
        Race::practice(gate as usize, side, height, exposure)
    };
    env.camera.render(
        &env.race,
        std::slice::from_raw_parts_mut(image, env.camera.byte_len()),
    );
}

#[no_mangle]
pub unsafe extern "C" fn sim_step(
    env: *mut Environment,
    action: *const [f32; 4],
    image: *mut u8,
    reward: *mut f32,
) -> i32 {
    let env = &mut *env;
    let Some(value) = env.race.step(*action) else {
        return -1;
    };
    *reward = value;
    env.camera.render(
        &env.race,
        std::slice::from_raw_parts_mut(image, env.camera.byte_len()),
    );
    match env.race.status {
        Status::Racing => 0,
        Status::Timeout => 2,
        Status::Stalled => 3,
        _ => 1,
    }
}

#[no_mangle]
pub unsafe extern "C" fn sim_lap_stats(env: *const Environment, gates: *mut u32, best: *mut f32) {
    let env = &*env;
    *gates = env.race.passed as u32;
    *best = env.race.best_lap.unwrap_or(f32::NAN);
}
