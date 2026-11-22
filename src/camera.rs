use crate::{ray_box, Course, Race, ARENA_MAX, ARENA_MIN, BARS};
use glam::Vec3;

pub const H_FOV: f32 = 155.;
pub const V_FOV: f32 = 115.;
pub const MOUNT_ANGLE: f32 = 51.3328;
// Intrinsics of IMX219 in 4TU MonoRace Log 8, normalized to image dimensions.
pub const LENS: [f32; 8] = [
    578.258_7 / 1640.,
    578.5696 / 1232.,
    801.66187 / 1640.,
    618.0977 / 1232.,
    0.04393058,
    -0.00824081,
    0.00355597,
    -0.00106689,
];
const GATE_TEXTURE: &[u8; 256 * 256 * 3] = include_bytes!("../assets/grand-challenge-gate.rgb");
const SEASON2_GATE_INK: &[u8; 256 * 256] = include_bytes!("../assets/season2-gate-ink.a8");
const HALL_CEILING: f32 = 8.6;

pub fn camera_pose(race: &Race) -> [Vec3; 4] {
    let drone = &race.drone;
    let (sin, cos) = MOUNT_ANGLE.to_radians().sin_cos();
    [
        drone.position + drone.rotation * Vec3::new(0.12, 0., 0.02),
        drone.rotation * -Vec3::Y,
        drone.rotation * Vec3::new(-sin, 0., cos),
        drone.rotation * Vec3::new(cos, 0., sin),
    ]
}

pub fn hall_bounds() -> [Vec3; 2] {
    [ARENA_MIN, Vec3::new(ARENA_MAX.x, ARENA_MAX.y, HALL_CEILING)]
}

pub struct Camera {
    rays: Vec<Vec3>,
    width: usize,
}

fn hash(mut n: u32) -> u32 {
    n ^= n >> 16;
    n = n.wrapping_mul(0x7feb_352d);
    n ^= n >> 15;
    n = n.wrapping_mul(0x846c_a68b);
    n ^ (n >> 16)
}

fn hdzero_breakup(image: &mut [u8], width: usize, tick: usize, seed: u32, strength: f32) {
    let frame = (tick / (crate::PHYSICS_HZ / crate::CONTROL_HZ)) as u32;
    let phase = hash(seed ^ (frame / 8));
    let noise = strength.clamp(0., 1.) * (0.15 + 0.85 * (phase & 255) as f32 / 255.);
    let frame_seed = hash(seed ^ frame.wrapping_mul(0x9e37_79b9));
    for (index, pixel) in image.as_chunks_mut::<3>().0.iter_mut().enumerate() {
        let x = (index % width) as u32;
        let y = (index / width) as u32;
        let cluster =
            hash(phase ^ (x / 13).wrapping_mul(0x8da6_b343) ^ (y / 3).wrapping_mul(0xd816_3841));
        let intensity = if cluster & 255 < (10. * noise) as u32 {
            0.78
        } else {
            0.035
        };
        let sample = hash(frame_seed ^ index as u32);
        if (sample & 0xffff) < (65535. * intensity * noise) as u32 {
            *pixel = match (sample >> 16) & 7 {
                0 | 1 => [15, 14, 21],
                2 | 3 => [235, 241, 232],
                4 => [195, 54, 196],
                5 => [53, 226, 104],
                _ => [112, 121, 134],
            };
        }
    }
}

fn near_grid(value: f32, spacing: f32, tolerance: f32) -> bool {
    (value - (value / spacing).round() * spacing).abs() < tolerance
}

fn floor(x: f32, y: f32) -> [u8; 3] {
    let fiber = ((x * 151.7).floor() as i32 ^ (y * 171.9).floor() as i32) & 15;
    let row = (x * 1.1).cos() * 4. + (y * 0.8).cos() * 4.;
    let level = (158. + row + fiber as f32 * 0.53) as u8;
    [level + 2, level + 1, level + 2]
}

fn gate_uv(side: f32, height: f32, rear: bool) -> (f32, f32) {
    let side = if rear { side } else { -side };
    (
        ((side / 2.7 + 0.5) * 256.).clamp(0., 255.),
        ((0.5 - height / 2.7) * 256.).clamp(0., 255.),
    )
}

fn gate_texture(side: f32, height: f32, rear: bool) -> [u8; 3] {
    let (u, v) = gate_uv(side, height, rear);
    let i = (v as usize * 256 + u as usize) * 3;
    [GATE_TEXTURE[i], GATE_TEXTURE[i + 1], GATE_TEXTURE[i + 2]]
}

fn season2_ink(contact: Vec3) -> f32 {
    let (u, v) = gate_uv(contact.y, contact.z, contact.x > 0.);
    let (x, y) = (u as usize, v as usize);
    let (next_x, next_y) = ((x + 1).min(255), (y + 1).min(255));
    let ink = |col, row| f32::from(SEASON2_GATE_INK[row * 256 + col]);
    let top = ink(x, y) + (ink(next_x, y) - ink(x, y)) * u.fract();
    let bottom = ink(x, next_y) + (ink(next_x, next_y) - ink(x, next_y)) * u.fract();
    (top + (bottom - top) * v.fract()) / 255.
}

fn season2_gate_color(contact: Vec3) -> [u8; 3] {
    // Season 2 used continuous light strips along the aperture and the vertical outer edges.
    let edge = contact.y.abs();
    let aperture = edge.max(contact.z.abs());
    let face = contact.x.abs() > 0.38;
    if face && (aperture < 0.79 || edge > 1.325) {
        [236, 221, 255]
    } else if face {
        let grain = ((contact.y * 147.).floor() as i32 ^ (contact.z * 183.).floor() as i32) & 7;
        let lit = (5. * (contact.z + 1.35) + grain as f32 * 0.6) as u8;
        let ink = season2_ink(contact);
        let panel = [74 + lit, 25 + lit, 155 + lit];
        std::array::from_fn(|i| {
            let base = f32::from(panel[i]);
            (base + (f32::from([236_u8, 230, 245][i]) - base) * ink) as u8
        })
    } else {
        [51, 23, 105]
    }
}

impl Camera {
    pub fn new(width: usize, height: usize) -> Self {
        let mut rays = Vec::with_capacity(width * height);
        let [fx, fy, cx, cy, k1, k2, k3, k4] = LENS;
        for row in 0..height {
            let v = (cy - (row as f32 + 0.5) / height as f32) / fy;
            for col in 0..width {
                let u = ((col as f32 + 0.5) / width as f32 - cx) / fx;
                let radius = u.hypot(v);
                let mut angle = radius;
                for _ in 0..5 {
                    let t = angle * angle;
                    let poly = 1. + t * (k1 + t * (k2 + t * (k3 + t * k4)));
                    let slope = 1. + t * (3. * k1 + t * (5. * k2 + t * (7. * k3 + t * 9. * k4)));
                    angle -= (angle * poly - radius) / slope;
                }
                let k = if radius > 0. {
                    angle.sin() / radius
                } else {
                    1.
                };
                rays.push(Vec3::new(u * k, v * k, angle.cos()));
            }
        }
        Self { rays, width }
    }

    pub fn byte_len(&self) -> usize {
        self.rays.len() * 3
    }

    pub fn render(&self, race: &Race, image: &mut [u8]) {
        let [sensor, right, up, forward] = camera_pose(race);
        let [px, py, pz] = sensor.to_array();
        let (arena_min, arena_max) = race.course.bounds();
        let arena_min = arena_min - sensor;
        let arena_max = Vec3::new(arena_max.x, arena_max.y, HALL_CEILING) - sensor;
        let gates: Vec<_> = race
            .gate_frames()
            .map(|&gate| {
                let delta = gate.center - sensor;
                (
                    gate,
                    delta,
                    gate.basis(-delta),
                    (delta.length_squared() - 4.).max(0.).sqrt(),
                )
            })
            .collect();

        for (pixel, ray) in image.as_chunks_mut::<3>().0.iter_mut().zip(&self.rays) {
            let direction = right * ray.x + up * ray.y + forward * ray.z;
            let [dx, dy, dz] = direction.to_array();
            let tx = if dx > 0. {
                arena_max.x / dx
            } else {
                arena_min.x / dx.min(-1e-12)
            };
            let ty = if dy > 0. {
                arena_max.y / dy
            } else {
                arena_min.y / dy.min(-1e-12)
            };
            let tz = if dz > 0. {
                arena_max.z / dz
            } else {
                arena_min.z / dz.min(-1e-12)
            };
            let mut distance = tx.min(ty).min(tz);
            let x = px + distance * dx;
            let y = py + distance * dy;
            let z = pz + distance * dz;

            let mut color = if distance == tz && dz < 0. {
                floor(x, y)
            } else if distance == tz {
                if near_grid(x, 4.8, 0.22) && near_grid(y, 4.65, 0.16) {
                    [253, 249, 224]
                } else {
                    let lit = (x * 1.3).cos() * 16.;
                    let level = (80. + lit) as u8;
                    [level + 9, level + 10, level]
                }
            } else {
                let along = if distance == tx { y } else { x };
                let fold = (along * 10. + (z * 0.72).sin() * 0.7).cos() * 9.;
                let spill = ((along * 0.62).cos() + 1.) * (3. + z * 1.4);
                let level = (36. + fold + spill) as u8;
                [level + 1, level + 1, level.saturating_sub(6)]
            };

            for &(gate, delta, origin, threshold) in &gates {
                if delta.dot(direction) < threshold {
                    continue;
                }
                let local = gate.basis(direction);
                for (min, max) in BARS {
                    let Some(hit) = ray_box(origin, local, min, max, 0.015, distance) else {
                        continue;
                    };
                    let contact = origin + local * hit;
                    color = if race.course == Course::Season2Practice2026 {
                        season2_gate_color(contact)
                    } else if contact.x.abs() >= 0.395 {
                        gate_texture(contact.y, contact.z, contact.x > 0.)
                    } else {
                        [53, 30, 93]
                    };
                    distance = hit;
                }
            }
            let fog = 1. / (1. + 0.005 * distance.max(0.));
            for (i, backdrop) in [27., 25., 34.].into_iter().enumerate() {
                pixel[i] = ((color[i] as f32 * fog + backdrop * (1. - fog)) * race.exposure)
                    .clamp(0., 255.) as u8;
            }
        }
        if race.video_breakup > 0. {
            hdzero_breakup(
                image,
                self.width,
                race.ticks,
                race.video_seed,
                race.video_breakup,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn raw_rgb_changes_with_camera_attitude() {
        let camera = Camera::new(96, 72);
        let mut race = Race::default();
        let mut a = vec![0; camera.byte_len()];
        let mut b = a.clone();
        camera.render(&race, &mut a);
        camera.render(&race, &mut b);
        assert_eq!(a, b);
        assert!(a.as_chunks::<3>().0.iter().any(|pixel| pixel[0] > 180));
        race.drone.rotation *= glam::Quat::from_rotation_z(0.5);
        camera.render(&race, &mut b);
        assert!(a.iter().zip(b).filter(|(x, y)| **x != *y).count() > a.len() / 4);
    }

    #[test]
    fn hdzero_errors_are_sparse_and_deterministic() {
        let camera = Camera::new(160, 120);
        let mut race = Race::default();
        let mut clean = vec![0; camera.byte_len()];
        let mut first = clean.clone();
        let mut second = clean.clone();
        camera.render(&race, &mut clean);
        race.video_breakup = 0.8;
        camera.render(&race, &mut first);
        camera.render(&race, &mut second);
        assert_eq!(first, second);
        let changes = first.iter().zip(&clean).filter(|(a, b)| a != b).count();
        assert!(changes > 0 && changes < clean.len() / 8);
    }
}
