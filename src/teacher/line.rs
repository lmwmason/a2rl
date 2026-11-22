//! The Season 2 teacher follows a continuous, curvature-limited racing line.
use std::sync::OnceLock;

use glam::{Quat, Vec3};

use crate::{Course, Gate, Race, GRAVITY};

const SAMPLES_PER_LEG: usize = 64;
const SPEED_ACCEL: f32 = 12.;

struct Pace {
    max_speed: f32,
    open: LegLimits,
    tight: LegLimits,
    fast: LegLimits,
    split_s_turn_accel: f32,
    split_s_exit: LegLimits,
    coast_at_open_gates: bool,
}

#[derive(Clone, Copy)]
struct LegLimits {
    turn_accel: f32,
    brake_accel: f32,
}

const NATIVE_LIMITS: LegLimits = LegLimits {
    turn_accel: 9.,
    brake_accel: 3.1,
};
const NATIVE_PACE: Pace = Pace {
    max_speed: 11.5,
    open: NATIVE_LIMITS,
    tight: NATIVE_LIMITS,
    fast: NATIVE_LIMITS,
    split_s_turn_accel: 6.5,
    split_s_exit: LegLimits {
        turn_accel: 6.5,
        brake_accel: 3.1,
    },
    coast_at_open_gates: true,
};
const FIRMWARE_PACE: Pace = Pace {
    max_speed: 18.,
    open: LegLimits {
        turn_accel: 15.,
        brake_accel: 5.,
    },
    tight: LegLimits {
        turn_accel: 11.5,
        brake_accel: 3.8,
    },
    fast: LegLimits {
        turn_accel: 23.,
        brake_accel: 8.,
    },
    split_s_turn_accel: 8.5,
    split_s_exit: LegLimits {
        turn_accel: 15.,
        brake_accel: 8.,
    },
    coast_at_open_gates: false,
};

impl Pace {
    fn limits_for_leg(&self, target: usize) -> LegLimits {
        match target {
            2 | 3 | 9 => self.fast,
            // Keep the approach limits through the chicane and the Ladder.
            4 | 5 | 10 | 11 => self.tight,
            6 | 7 => LegLimits {
                turn_accel: self.split_s_turn_accel,
                brake_accel: self.open.brake_accel,
            },
            8 => self.split_s_exit,
            _ => self.open,
        }
    }
}

pub(super) fn is_ladder_leg(target: usize) -> bool {
    matches!(target, 10 | 11)
}

pub(super) fn is_fast_leg(target: usize) -> bool {
    matches!(target, 2 | 3 | 8 | 9)
}

// Horizontal guide positions from FlyBy Fig. 5. Heights interpolate between
// the two estimated opening centers and keep the loop clear of the frame.
const LADDER_ARC: [(Vec3, Vec3); 3] = [
    (Vec3::new(16.75, 5.58, 3.55), Vec3::new(-0.9, 0.25, -0.33)),
    (Vec3::new(16.45, 7.1, 2.75), Vec3::new(0.42, 0.81, -0.42)),
    (Vec3::new(18.1, 7.8, 1.75), Vec3::new(0.54, -0.79, -0.28)),
];

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
static SEASON2_FIRMWARE_LINE: OnceLock<Line> = OnceLock::new();

impl Line {
    fn season2(pace: Pace) -> Self {
        let course = Course::Season2Practice2026;
        let gates: Vec<_> = Gate::on_course(course).collect();
        let (start, yaw) = course.start();
        let start_direction = Quat::from_rotation_z(yaw.to_radians()) * Vec3::X;
        let mut points = Vec::with_capacity(3 * gates.len() * SAMPLES_PER_LEG + 1);
        let mut begin = start;
        let mut direction = start_direction;
        for (leg, gate) in gates.iter().cycle().take(3 * gates.len()).enumerate() {
            let mut start = (begin, direction);
            let target = (gate.center, gate.normal);
            if leg % gates.len() == gates.len() - 1 {
                let knots = LADDER_ARC.into_iter().chain([target]);
                for end in knots {
                    append_curve(&mut points, start, end, SAMPLES_PER_LEG / 4);
                    start = end;
                }
            } else {
                append_curve(&mut points, start, target, SAMPLES_PER_LEG);
            }
            begin = gate.center;
            direction = gate.normal;
        }
        points.push(Point {
            position: begin,
            tangent: direction,
            curvature: Vec3::ZERO,
            speed: pace.max_speed,
        });
        for index in 1..points.len() - 1 {
            let before = points[index - 1];
            let after = points[index + 1];
            let arc = before.position.distance(points[index].position)
                + after.position.distance(points[index].position);
            let curvature = (after.tangent - before.tangent) / arc;
            let point = &mut points[index];
            point.curvature = curvature;
            let leg = index / SAMPLES_PER_LEG % gates.len();
            let turn_accel = pace.limits_for_leg(leg).turn_accel;
            point.speed = (turn_accel / curvature.length().max(0.001))
                .sqrt()
                .min(pace.max_speed);
        }
        for gate in (SAMPLES_PER_LEG..points.len() - SAMPLES_PER_LEG).step_by(SAMPLES_PER_LEG) {
            let target = (gate / SAMPLES_PER_LEG - 1) % gates.len();
            // The curvature and braking passes already cap the speed on
            // open approaches. Hold entry speed only at constrained gates.
            if pace.coast_at_open_gates || !matches!(target, 1 | 2 | 3 | 8 | 9) {
                coast_through_gate(&mut points, gate);
            }
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
            let leg = index / SAMPLES_PER_LEG % gates.len();
            let braking = pace.limits_for_leg(leg).brake_accel;
            let next = points[index + 1];
            let point = &mut points[index];
            let distance = point.position.distance(next.position);
            point.speed = point.speed.min(
                next.speed
                    .mul_add(next.speed, 2. * braking * distance)
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

fn coast_through_gate(points: &mut [Point], gate: usize) {
    // Finish braking before the aperture and preserve forward speed until clear.
    let mut first = gate;
    let mut last = gate;
    let mut length = 0.;
    while length < 1.45 && first > 0 {
        length += points[first - 1].position.distance(points[first].position);
        first -= 1;
    }
    length = 0.;
    while length < 0.55 && last + 1 < points.len() {
        length += points[last].position.distance(points[last + 1].position);
        last += 1;
    }
    let limit = points[first..=last]
        .iter()
        .map(|point| point.speed)
        .fold(f32::INFINITY, f32::min);
    for point in &mut points[first..=last] {
        point.speed = limit;
    }
}

fn append_curve(
    points: &mut Vec<Point>,
    (begin, direction): (Vec3, Vec3),
    (end, heading): (Vec3, Vec3),
    samples: usize,
) {
    let distance = begin.distance(end);
    let reversal = direction.dot(heading) < -0.9 && begin.truncate().distance(end.truncate()) < 0.5;
    let handle = if reversal { 8.5 } else { 0.88 * distance };
    for tick in 0..samples {
        let t = tick as f32 / samples as f32;
        points.push(hermite(
            begin,
            direction.normalize() * handle,
            end,
            heading.normalize() * handle,
            t,
        ));
    }
}

fn hermite(a: Vec3, da: Vec3, b: Vec3, db: Vec3, t: f32) -> Point {
    let quadratic = 3. * (b - a) - 2. * da - db;
    let cubic = 2. * (a - b) + da + db;
    Point {
        position: a + t * (da + t * (quadratic + t * cubic)),
        tangent: (da + t * (2. * quadratic + 3. * t * cubic)).normalize(),
        curvature: Vec3::ZERO,
        speed: f32::INFINITY,
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

pub(super) fn track(race: &Race, firmware: bool) -> Option<(Vec3, Vec3)> {
    let target = race.gates[race.next_gate()];
    let position = race.drone.position;
    let line = if firmware {
        SEASON2_FIRMWARE_LINE.get_or_init(|| Line::season2(FIRMWARE_PACE))
    } else {
        SEASON2_LINE.get_or_init(|| Line::season2(NATIVE_PACE))
    };
    let (index, fraction) = line.closest(race.passed, position);
    if index + 5 >= (race.passed + 1) * SAMPLES_PER_LEG
        && (position - target.center).dot(target.normal) > 0.60
    {
        return None;
    }
    let here = blend(line.points[index], line.points[index + 1], fraction);
    if here.position.distance_squared(position) > 3.0_f32.powi(2) {
        return None;
    }
    let preview = line.look_ahead(index, 0.32 + 0.18 * here.speed);
    let velocity_error = here.tangent * preview.speed - race.drone.velocity;
    let correction = here.position - position;
    let mut accel = 3.8 * velocity_error
        + 10. * correction
        + 0.75 * here.speed.powi(2) * here.curvature
        - race.drone.rotation * crate::drag(race.drone.rotation.conjugate() * race.drone.velocity)
            / crate::MASS_KG;
    if firmware && matches!(race.next_gate(), 2 | 3 | 8) {
        // Add forward thrust on longer legs where the aircraft is aligned,
        // with enough room before the gate and speed left in the plan.
        let depth = (target.center - position).dot(target.normal);
        let room = ((depth - 2.) / 2.).clamp(0., 1.);
        let alignment = (1. - correction.length() / 0.9).clamp(0., 1.);
        let straightness = (1. - here.curvature.length() / 0.22).clamp(0., 1.);
        let speed_deficit = (preview.speed - race.drone.velocity.dot(here.tangent)).clamp(0., 5.);
        accel += here.tangent * (7. * speed_deficit * alignment * room * straightness);
    }
    let toward_gate = (target.center - position).truncate().normalize_or_zero();
    let heading = preview.tangent.truncate();
    let heading = (heading + 0.65 * toward_gate).normalize_or_zero();
    let heading = if heading.length_squared() > 0.5 {
        heading.extend(0.)
    } else {
        target.normal
    };
    // Avoid braking through the opening. The backwards speed pass already
    // limits entry speed before the gate according to the upcoming curvature.
    let minimum_forward = (GRAVITY + accel.z.clamp(-5.5, 8.)) * 0.065;
    accel += heading * (level_approach(race) * (minimum_forward - accel.dot(heading)).max(0.));
    Some((accel, heading))
}

pub(super) fn level_approach(race: &Race) -> f32 {
    let index = race.next_gate();
    let gate = race.gates[index];
    let previous = race.gates[(index + race.gates.len() - 1) % race.gates.len()];
    // Allow unrestricted pitch on legs that climb to a higher gate.
    if gate.center.z > previous.center.z + 0.6 {
        return 0.;
    }
    let offset = gate.center - race.drone.position;
    let depth = offset.dot(gate.normal);
    let lateral = (offset - gate.normal * depth).length();
    ((3.2 - depth.abs()) / 1.4).clamp(0., 1.) * ((2.4 - lateral) / 0.7).clamp(0., 1.)
}
