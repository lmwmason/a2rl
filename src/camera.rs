use crate::{ray_box, Race, ARENA_MAX, ARENA_MIN, BARS, LAYOUT};
use glam::Vec3;

pub const H_FOV: f32 = 155.;
pub const V_FOV: f32 = 115.;

pub struct Camera {
    rays: Vec<Vec3>,
}

fn near_grid(value: f32, spacing: f32, tolerance: f32) -> bool {
    (value - (value / spacing).round() * spacing).abs() < tolerance
}

fn floor(x: f32, y: f32) -> [u8; 3] {
    if (x - 3.).abs() < 0.4 && (y + 10.).abs() < 2.1 {
        return if ((x * 5.).floor() as i32 + (y * 5.).floor() as i32) % 2 == 0 {
            [222, 239, 246]
        } else {
            [16, 25, 41]
        };
    }
    let mut best = f32::INFINITY;
    for i in 0..LAYOUT.len() {
        let [ax, ay, ..] = LAYOUT[i];
        let [bx, by, ..] = LAYOUT[(i + 1) % LAYOUT.len()];
        if x < ax.min(bx) - 2.2
            || x > ax.max(bx) + 2.2
            || y < ay.min(by) - 2.2
            || y > ay.max(by) + 2.2
        {
            continue;
        }
        let dx = bx - ax;
        let dy = by - ay;
        let t = (((x - ax) * dx + (y - ay) * dy) / (dx * dx + dy * dy)).clamp(0., 1.);
        best = best.min((x - ax - t * dx).powi(2) + (y - ay - t * dy).powi(2));
    }
    if best < 3.9 {
        [39, 63, 76]
    } else if best < 4.5 {
        [61, 156, 150]
    } else if near_grid(x, 2., 0.043) || near_grid(y, 2., 0.043) {
        [62, 85, 101]
    } else {
        [47, 64, 81]
    }
}

impl Camera {
    pub fn new(width: usize, height: usize) -> Self {
        let mut rays = Vec::with_capacity(width * height);
        for row in 0..height {
            let v = (1. - 2. * (row as f32 + 0.5) / height as f32) * V_FOV.to_radians() / 2.;
            for col in 0..width {
                let u = (2. * (col as f32 + 0.5) / width as f32 - 1.) * H_FOV.to_radians() / 2.;
                let angle = u.hypot(v);
                let k = if angle > 0. { angle.sin() / angle } else { 1. };
                rays.push(Vec3::new(u * k, v * k, angle.cos()));
            }
        }
        Self { rays }
    }

    pub fn byte_len(&self) -> usize {
        self.rays.len() * 3
    }

    pub fn render(&self, race: &Race, image: &mut [u8]) {
        let drone = &race.drone;
        let sensor = drone.position + drone.rotation * Vec3::new(0.12, 0., 0.02);
        let [px, py, pz] = sensor.to_array();
        let (sin, cos) = 12_f32.to_radians().sin_cos();
        let right = drone.rotation * -Vec3::Y;
        let up = drone.rotation * Vec3::new(-sin, 0., cos);
        let forward = drone.rotation * Vec3::new(cos, 0., sin);
        let arena_min = ARENA_MIN - sensor;
        let arena_max = ARENA_MAX - sensor;
        let gates = race.gates.map(|gate| {
            let delta = gate.center - sensor;
            (
                gate,
                delta,
                gate.basis(-delta),
                (delta.length_squared() - 6.).max(0.).sqrt(),
            )
        });

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
                if near_grid(x, 6., 1.80) && near_grid(y, 7., 0.17) {
                    [186, 220, 240]
                } else if near_grid(x, 6., 0.18) {
                    [16, 25, 41]
                } else {
                    [27, 42, 61]
                }
            } else if (2.15..2.6).contains(&z) {
                [33, 125, 139]
            } else if near_grid(if distance == tx { y } else { x }, 3., 0.13)
                || (z - 8.5).abs() < 0.10
            {
                [56, 79, 105]
            } else {
                [38, 55, 79]
            };

            for (index, &(gate, delta, origin, threshold)) in gates.iter().enumerate() {
                if delta.dot(direction) < threshold {
                    continue;
                }
                let local = gate.basis(direction);
                for (min, max) in BARS {
                    let Some(hit) = ray_box(origin, local, min, max, 0.015, distance) else {
                        continue;
                    };
                    let contact = origin + local * hit;
                    color = if contact.x.abs() < 0.195 {
                        [91, 60, 177]
                    } else if contact.z.abs() > 1.32 && contact.y.abs() > 1.02 {
                        if index + 1 == gates.len() {
                            [94, 240, 190]
                        } else {
                            [153, 116, 255]
                        }
                    } else {
                        [222, 239, 246]
                    };
                    distance = hit;
                }
            }
            let fog = 1. / (1. + 0.014 * distance.max(0.));
            for (i, backdrop) in [20., 33., 53.].into_iter().enumerate() {
                pixel[i] = ((color[i] as f32 * fog + backdrop * (1. - fog)) * race.exposure)
                    .clamp(0., 255.) as u8;
            }
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
        race.drone.rotation = glam::Quat::from_rotation_z(0.5);
        camera.render(&race, &mut b);
        assert!(a.iter().zip(b).filter(|(x, y)| **x != *y).count() > a.len() / 4);
    }
}
