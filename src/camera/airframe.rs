//! Third-person model of the A2RL × DCL reference quadrotor.
use crate::ray_box;
use glam::{Vec2, Vec3};

const CARBON: [u8; 3] = [28, 30, 34];
const ORANGE: [u8; 3] = [237, 143, 35];
const PURPLE: [u8; 3] = [64, 37, 111];
const LED: [u8; 3] = [255, 227, 118];
const MOTOR_X: f32 = 0.114;
const MOTOR_Y: f32 = 0.105;
const PROP_RADIUS: f32 = 0.0635;

// X station, bottom and top Z, and half-width. Based on the published
// reference-quad photograph, with a pointed tail and a raised camera hood.
const CANOPY: [[f32; 4]; 6] = [
    [-0.148, -0.002, 0.011, 0.011],
    [-0.103, -0.016, 0.027, 0.031],
    [-0.028, -0.014, 0.049, 0.038],
    [0.035, -0.011, 0.066, 0.034],
    [0.093, -0.010, 0.035, 0.027],
    [0.133, -0.009, 0.009, 0.014],
];

fn canopy_ring([x, bottom, top, width]: [f32; 4]) -> [Vec3; 4] {
    [
        Vec3::new(x, -width, bottom),
        Vec3::new(x, width, bottom),
        Vec3::new(x, width * 0.62, top),
        Vec3::new(x, -width * 0.62, top),
    ]
}

fn triangle(origin: Vec3, ray: Vec3, [a, b, c]: [Vec3; 3], far: f32) -> Option<(f32, Vec3)> {
    let ab = b - a;
    let ac = c - a;
    let across = ray.cross(ac);
    let det = ab.dot(across);
    if det.abs() < 1e-9 {
        return None;
    }
    let offset = origin - a;
    let u = offset.dot(across) / det;
    let cross = offset.cross(ab);
    let v = ray.dot(cross) / det;
    let t = ac.dot(cross) / det;
    (u >= 0. && v >= 0. && u + v <= 1. && t >= 0.015 && t < far)
        .then(|| (t, ab.cross(ac).normalize()))
}

fn shade(color: [u8; 3], normal: Vec3, light: Vec3) -> [u8; 3] {
    let factor = 0.72 + 0.28 * normal.dot(light).abs();
    color.map(|value| (value as f32 * factor) as u8)
}

pub(super) fn draw(
    origin: Vec3,
    ray: Vec3,
    ticks: usize,
    light: Vec3,
    far: &mut f32,
    rgb: &mut [u8; 3],
) {
    if ray_box(
        origin,
        ray,
        Vec3::new(-0.18, -0.171, -0.05),
        Vec3::new(0.18, 0.171, 0.071),
        0.015,
        *far,
    )
    .is_none()
    {
        return;
    }
    let mut box_hit = |eye, direction, center, half, color| {
        if let Some(t) = ray_box(eye, direction, center - half, center + half, 0.015, *far) {
            *far = t;
            *rgb = color;
        }
    };
    box_hit(
        origin,
        ray,
        Vec3::new(-0.005, 0., -0.025),
        Vec3::new(0.082, 0.035, 0.013),
        CARBON,
    );
    box_hit(
        origin,
        ray,
        Vec3::new(0.04, 0., -0.033),
        Vec3::new(0.040, 0.021, 0.014),
        [70, 66, 59],
    );
    box_hit(
        origin,
        ray,
        Vec3::new(-0.124, 0., 0.003),
        Vec3::new(0.007, 0.017, 0.007),
        [242, 63, 24],
    );
    box_hit(
        origin,
        ray,
        Vec3::new(0.117, 0., 0.025),
        Vec3::new(0.013, 0.013, 0.012),
        [19, 23, 31],
    );
    for x in [-1., 1.] {
        for y in [-1., 1.] {
            let motor = Vec3::new(x * MOTOR_X, y * MOTOR_Y, -0.005);
            let arm = motor.with_z(0.).normalize();
            let cross = Vec3::Z.cross(arm);
            let local = |p: Vec3| Vec3::new(p.dot(arm), p.dot(cross), p.z);
            let eye = local(origin);
            let direction = local(ray);
            box_hit(
                eye,
                direction,
                Vec3::new(0.094, 0., -0.017),
                Vec3::new(0.063, 0.010, 0.004),
                CARBON,
            );
            box_hit(
                eye,
                direction,
                Vec3::new(0.098, 0., -0.011),
                Vec3::new(0.040, 0.0035, 0.0018),
                LED,
            );
            box_hit(
                origin,
                ray,
                Vec3::new(0., motor.y, -0.016),
                Vec3::new(MOTOR_X, 0.0035, 0.003),
                CARBON,
            );
            box_hit(
                origin,
                ray,
                motor,
                Vec3::new(0.014, 0.014, 0.010),
                [55, 52, 50],
            );
            box_hit(
                origin,
                ray,
                motor.with_z(0.009),
                Vec3::new(0.008, 0.008, 0.007),
                [125, 94, 50],
            );
        }
    }
    if ray.z.abs() > 1e-6 {
        let t = (0.017 - origin.z) / ray.z;
        if t >= 0.015 && t < *far {
            let contact = (origin + ray * t).truncate();
            for x in [-1., 1.] {
                for y in [-1., 1.] {
                    let delta = contact - Vec2::new(x * MOTOR_X, y * MOTOR_Y);
                    let r = delta.length();
                    if r <= PROP_RADIUS && r > 0.008 {
                        let phase = x * y * (ticks as f32 * 1.83) + x * 0.4;
                        let angle = delta.y.atan2(delta.x) + phase + r * 6.;
                        let blade = (angle * 3.).cos() > 0.65 + 0.24 * r / PROP_RADIUS;
                        if blade {
                            *rgb = [59, 61, 67];
                            *far = t;
                        }
                        if ticks >= 240 && !blade {
                            for channel in rgb.iter_mut() {
                                *channel = (*channel as f32 * 0.91) as u8;
                            }
                        }
                    }
                }
            }
        }
    }
    for pair in CANOPY.windows(2) {
        let a = canopy_ring(pair[0]);
        let b = canopy_ring(pair[1]);
        for edge in 0..4 {
            let next = (edge + 1) % 4;
            for vertices in [[a[edge], b[edge], b[next]], [a[edge], b[next], a[next]]] {
                if let Some((t, normal)) = triangle(origin, ray, vertices, *far) {
                    let p = origin + ray * t;
                    let strip = p.z + 0.14 * p.x;
                    let color = if edge == 2 || strip < 0.013 || p.x < -0.12 {
                        ORANGE
                    } else if strip < 0.017 {
                        LED
                    } else {
                        PURPLE
                    };
                    *far = t;
                    *rgb = shade(color, normal, light);
                }
            }
        }
    }
    for station in [CANOPY[0], CANOPY[CANOPY.len() - 1]] {
        let [a, b, c, d] = canopy_ring(station);
        for face in [[a, b, c], [a, c, d]] {
            if let Some((t, normal)) = triangle(origin, ray, face, *far) {
                *far = t;
                *rgb = shade(PURPLE, normal, light);
            }
        }
    }
}
