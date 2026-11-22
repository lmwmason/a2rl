//! The Season 2 teacher follows a continuous, curvature-limited racing line.
use std::sync::OnceLock;

use glam::{Quat, Vec3};

use crate::{Course, Gate, Race};

const SAMPLES_PER_LEG: usize = 64;
const MAX_SPEED: f32 = 11.5;
const TURN_ACCEL: f32 = 7.2;
const SPEED_ACCEL: f32 = 6.;
const BRAKE_ACCEL: f32 = 6.5;

#[derive(Clone, Copy)]
struct Point {
    position: Vec3,
    tangent: Vec3,
    curvature: Vec3,
    speed: f32,
}

struct Line {
    points: Vec<Point>,
}

static SEASON2_LINE: OnceLock<Line> = OnceLock::new();

impl Line {
    fn season2() -> Self {
        let course = Course::Season2Practice2026;
        let gates: Vec<_> = Gate::on_course(course).collect();
        let (start, yaw) = course.start();
        let start_direction = Quat::from_rotation_z(yaw.to_radians()) * Vec3::X;
        let mut points = Vec::with_capacity(3 * gates.len() * SAMPLES_PER_LEG + 1);
        let mut begin = start;
        let mut direction = start_direction;
        for gate in gates.iter().cycle().take(3 * gates.len()) {
            let distance = gate.center.distance(begin);
            let reversal = gate.normal.dot(direction) < -0.9
                && gate.center.truncate().distance(begin.truncate()) < 0.5;
            let handle = if reversal { 8.5 } else { 0.88 * distance };
            for tick in 0..SAMPLES_PER_LEG {
                let t = tick as f32 / SAMPLES_PER_LEG as f32;
                points.push(hermite(
                    begin,
                    direction * handle,
                    gate.center,
                    gate.normal * handle,
                    t,
                ));
            }
            begin = gate.center;
            direction = gate.normal;
        }
        points.push(Point {
            position: begin,
            tangent: direction,
            curvature: Vec3::ZERO,
            speed: MAX_SPEED,
        });
        for index in 1..points.len() - 1 {
            let before = points[index - 1];
            let after = points[index + 1];
            let arc = before.position.distance(points[index].position)
                + after.position.distance(points[index].position);
            let curvature = (after.tangent - before.tangent) / arc;
            let point = &mut points[index];
            point.curvature = curvature;
            point.speed = (TURN_ACCEL / curvature.length().max(0.001))
                .sqrt()
                .min(MAX_SPEED);
        }
        points[0].speed = 2.2;
        for index in 1..points.len() {
            let previous = points[index - 1];
            let point = &mut points[index];
            let distance = point.position.distance(previous.position);
            point.speed = point.speed.min(
                previous
                    .speed
                    .mul_add(previous.speed, 2. * SPEED_ACCEL * distance)
                    .sqrt(),
            );
        }
        for index in (0..points.len() - 1).rev() {
            let next = points[index + 1];
            let point = &mut points[index];
            let distance = point.position.distance(next.position);
            point.speed = point.speed.min(
                next.speed
                    .mul_add(next.speed, 2. * BRAKE_ACCEL * distance)
                    .sqrt(),
            );
        }
        Self { points }
    }

    fn closest(&self, gate_index: usize, position: Vec3) -> (usize, f32) {
        let first = (gate_index * SAMPLES_PER_LEG).saturating_sub(5);
        let end = (gate_index + 1) * SAMPLES_PER_LEG + 5;
        let mut closest = (first, 0.);
        let mut best = f32::INFINITY;
        for index in first..end {
            let a = self.points[index].position;
            let delta = self.points[index + 1].position - a;
            let along = ((position - a).dot(delta) / delta.length_squared()).clamp(0., 1.);
            let distance = (a + delta * along).distance_squared(position);
            if distance < best {
                best = distance;
                closest = (index, along);
            }
        }
        closest
    }

    fn look_ahead(&self, index: usize, distance: f32) -> Point {
        let mut remaining = distance;
        for pair in self.points[index..].windows(2) {
            let distance = pair[0].position.distance(pair[1].position);
            if remaining <= distance {
                return blend(pair[0], pair[1], remaining / distance);
            }
            remaining -= distance;
        }
        *self.points.last().unwrap()
    }
}

fn hermite(a: Vec3, da: Vec3, b: Vec3, db: Vec3, t: f32) -> Point {
    let quadratic = 3. * (b - a) - 2. * da - db;
    let cubic = 2. * (a - b) + da + db;
    Point {
        position: a + t * (da + t * (quadratic + t * cubic)),
        tangent: (da + t * (2. * quadratic + 3. * t * cubic)).normalize(),
        curvature: Vec3::ZERO,
        speed: MAX_SPEED,
    }
}

fn blend(a: Point, b: Point, t: f32) -> Point {
    Point {
        position: a.position.lerp(b.position, t),
        tangent: a.tangent.lerp(b.tangent, t).normalize_or_zero(),
        curvature: a.curvature.lerp(b.curvature, t),
        speed: a.speed + t * (b.speed - a.speed),
    }
}

pub(super) fn track(race: &Race) -> Option<(Vec3, Vec3)> {
    let target = race.gates[race.next_gate()];
    let position = race.drone.position;
    if (position - target.center).dot(target.normal) > 0.60 {
        return None;
    }
    let line = SEASON2_LINE.get_or_init(Line::season2);
    let (index, fraction) = line.closest(race.passed, position);
    let here = blend(line.points[index], line.points[index + 1], fraction);
    if here.position.distance_squared(position) > 3.0_f32.powi(2) {
        return None;
    }
    let preview = line.look_ahead(index, 0.32 + 0.18 * here.speed);
    let velocity_error = here.tangent * preview.speed - race.drone.velocity;
    let correction = here.position - position;
    let accel = 3.8 * velocity_error
        + 10. * correction
        + 0.75 * here.speed.powi(2) * here.curvature
        - race.drone.rotation * crate::drag(race.drone.rotation.conjugate() * race.drone.velocity)
            / crate::MASS_KG;
    let toward_gate = (target.center - position).truncate().normalize_or_zero();
    let heading = preview.tangent.truncate();
    let heading = (heading + 0.65 * toward_gate).normalize_or_zero();
    let heading = if heading.length_squared() > 0.5 {
        heading.extend(0.)
    } else {
        target.normal
    };
    Some((accel, heading))
}
