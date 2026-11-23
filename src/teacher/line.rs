//! The Season 2 teacher follows a continuous, curvature-limited racing line.
use std::sync::OnceLock;

use glam::{Quat, Vec3};

use crate::{Course, Gate, Race, GRAVITY};

const SAMPLES_PER_LEG: usize = 64;
const SPEED_ACCEL: f32 = 12.;

// Calibrated per leg against the reconstructed course with Betaflight SITL.
const FIRMWARE_SPEED_SCALE: [f32; crate::LAYOUT_2026.len()] = [
    1.070800, 1.042064, 1.080000, 1.094382, 1.124069, 1.073815, 1.071267, 1.171599, 1.096856,
    1.069650, 1.169062, 1.,
];
const ATTITUDE_PREVIEW_SECONDS: f32 = 1.102928 * 0.15;

#[derive(Clone, Copy)]
struct TrackingGains {
    preview: f32,
    velocity: f32,
    position: f32,
    curvature: f32,
    level_approach: f32,
    lateral: f32,
}

const REFERENCE_TRACKING: TrackingGains = TrackingGains {
    preview: 1.,
    velocity: 3.8,
    position: 10.,
    curvature: 1.,
    level_approach: 1.,
    lateral: 1.,
};
const PREDICTIVE_TRACKING: TrackingGains = TrackingGains {
    preview: 0.876500,
    velocity: 3.8 * 1.016821,
    position: 10. * 1.010967,
    curvature: 1.076000,
    level_approach: 0.957144,
    lateral: 1.086832,
};

#[derive(Clone, Copy)]
struct Pace {
    max_speed: f32,
    open: LegLimits,
    tight: LegLimits,
    fast: LegLimits,
    split_s_turn_accel: f32,
    split_s_exit: LegLimits,
    ladder_exit: LegLimits,
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
    ladder_exit: NATIVE_LIMITS,
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
    ladder_exit: LegLimits {
        turn_accel: 14.2,
        brake_accel: 4.7,
    },
    coast_at_open_gates: false,
};

impl Pace {
    fn limits_for_leg(&self, target: usize) -> LegLimits {
        match target {
            2 | 3 | 9 => self.fast,
            // Keep the approach limits through the chicane and the Ladder.
            4 | 5 | 10 => self.tight,
            6 | 7 => LegLimits {
                turn_accel: self.split_s_turn_accel,
                brake_accel: self.open.brake_accel,
            },
            8 => self.split_s_exit,
            11 => self.ladder_exit,
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

type Ladder = [(Vec3, Vec3); 3];

// Horizontal guide positions from FlyBy Fig. 5. Heights interpolate between
// the two estimated opening centers and keep the loop clear of the frame.
const LADDER_ARC: Ladder = [
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

pub(super) struct Line {
    points: Vec<Point>,
}

#[derive(Clone, Copy, Default)]
struct Crossing {
    side: f32,
    yaw: f32,
}

static SEASON2_LINE: OnceLock<Line> = OnceLock::new();
static SEASON2_FIRMWARE_LINE: OnceLock<Line> = OnceLock::new();

impl Line {
    fn season2(pace: Pace, crossings: &[Crossing; crate::LAYOUT_2026.len()]) -> Self {
        Self::with_ladder(pace, crossings, LADDER_ARC)
    }

    fn with_ladder(
        pace: Pace,
        crossings: &[Crossing; crate::LAYOUT_2026.len()],
        ladder: Ladder,
    ) -> Self {
        let course = Course::Season2Practice2026;
        let gates: Vec<_> = Gate::on_course(course).collect();
        let (start, yaw) = course.start();
        let start_direction = Quat::from_rotation_z(yaw.to_radians()) * Vec3::X;
        let mut points = Vec::with_capacity(3 * gates.len() * SAMPLES_PER_LEG + 1);
        let mut begin = start;
        let mut direction = start_direction;
        for (leg, gate) in gates.iter().cycle().take(3 * gates.len()).enumerate() {
            let mut start = (begin, direction);
            let crossing = crossings[leg % gates.len()];
            let target = (
                gate.center + crossing.side * gate.side,
                Quat::from_rotation_z(crossing.yaw) * gate.normal,
            );
            if leg % gates.len() == gates.len() - 1 {
                let knots = ladder.into_iter().chain([target]);
                for end in knots {
                    append_curve(&mut points, start, end, SAMPLES_PER_LEG / 4);
                    start = end;
                }
            } else {
                append_curve(&mut points, start, target, SAMPLES_PER_LEG);
            }
            (begin, direction) = target;
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

    fn firmware() -> Self {
        Self::firmware_with_ladder_scale(1.).expect("nominal firmware line")
    }

    pub(super) fn firmware_with_ladder_scale(ladder_scale: f32) -> Option<Self> {
        // The measured reference flight passes close to the sides of the
        // opening at gates 2, 6 and 10. Keep their targets fixed.
        const MOVABLE_GATES: [(usize, f32); 4] = [(2, 1.), (3, -1.), (4, -1.), (8, 1.)];
        let mut crossings = [Crossing::default(); crate::LAYOUT_2026.len()];
        let mut best = Self::season2(FIRMWARE_PACE, &crossings);
        let mut cost = best.estimated_time(&crossings);

        // Optimize within each opening, allowing oblique crossings. The
        // launch, Split-S and Ladder guide positions serve as fixed anchors.
        for (side_step, yaw_step) in [(0.09, 0.10), (0.06, 0.075), (0.03, 0.04)] {
            for (gate, inward) in MOVABLE_GATES {
                for (side, yaw) in [
                    (-side_step, 0.),
                    (side_step, 0.),
                    (0., -yaw_step),
                    (0., yaw_step),
                ] {
                    let mut trial = crossings;
                    // Use the opposite side of the measured tracking error
                    // to preserve clearance on both laps with the firmware.
                    trial[gate].side =
                        inward * ((trial[gate].side + side) * inward).clamp(0., 0.18);
                    trial[gate].yaw = (trial[gate].yaw + yaw).clamp(-0.22, 0.22);
                    let line = Self::season2(FIRMWARE_PACE, &trial);
                    let time = line.estimated_time(&trial);
                    if time < cost && line.clears_frames() {
                        crossings = trial;
                        best = line;
                        cost = time;
                    }
                }
            }
        }
        // Pull the loop inward after clearing the upper frame, and align the
        // descent with the lower opening earlier. Use the frame test before
        // handing the re-planned line to the flight controller.
        let mut pace = FIRMWARE_PACE;
        pace.ladder_exit.turn_accel *= 1.468000;
        pace.ladder_exit.brake_accel *= 0.568000;
        let offsets = [
            Vec3::new(0.316816, 0.330859, 0.252000),
            Vec3::new(0.307844, -0.222176, 0.194575),
            Vec3::new(-0.280147, -0.359855, 0.180000),
        ];
        let gates: Vec<_> = Gate::on_course(Course::Season2Practice2026).collect();
        let midpoint = (gates[10].center + gates[11].center) * 0.5;
        let ladder = std::array::from_fn(|index| {
            let (position, tangent) = LADDER_ARC[index];
            let yaw = [-0.7, 0.4, 0.8][index] * -0.216000;
            let mut direction = Quat::from_rotation_z(yaw) * tangent;
            direction.z += 0.064829;
            let point = position + offsets[index];
            (
                point + (point - midpoint).with_z(0.) * (ladder_scale - 1.),
                direction,
            )
        });
        let mut adjusted = Self::with_ladder(pace, &crossings, ladder);
        // Keep the independently planned approach speed up to the upper gate.
        // The slower braking schedule is local to the descending loop.
        for (index, (point, reference)) in adjusted.points.iter_mut().zip(&best.points).enumerate()
        {
            if index / SAMPLES_PER_LEG % crate::LAYOUT_2026.len() != crate::LAYOUT_2026.len() - 1 {
                *point = *reference;
            }
        }
        if adjusted.clears_frames() {
            Some(adjusted)
        } else if ladder_scale == 1. {
            Some(best)
        } else {
            None
        }
    }

    fn estimated_time(&self, crossings: &[Crossing]) -> f32 {
        let finish = 2 * crossings.len() * SAMPLES_PER_LEG;
        let time: f32 = self.points[SAMPLES_PER_LEG..=finish]
            .windows(2)
            .map(|pair| {
                2. * pair[0].position.distance(pair[1].position) / (pair[0].speed + pair[1].speed)
            })
            .sum();
        // Prefer extra clearance where off-center paths gain little time.
        time + 0.25 * crossings.iter().map(|gate| gate.side.powi(2)).sum::<f32>()
    }

    fn clears_frames(&self) -> bool {
        let course = Course::Season2Practice2026;
        let gates: Vec<_> = Gate::on_course(course)
            .chain(course.extra_frames().iter().copied())
            .collect();
        self.points.windows(2).all(|pair| {
            !gates
                .iter()
                .any(|gate| gate.progress(pair[0].position, pair[1].position, course).0)
        })
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

fn predictive_leg(race: &Race) -> bool {
    // The takeoff and the Ladder loop use their dedicated pace and geometry.
    race.passed > 0 && race.next_gate() != crate::LAYOUT_2026.len() - 1
}

pub(super) fn gate_height_guard(race: &Race, accel: Vec3, command: &mut [f32; 4]) {
    if !predictive_leg(race) {
        return;
    }
    let gate = race.gates[race.next_gate()];
    let depth = (gate.center - race.drone.position).dot(gate.normal);
    let closing = race.drone.velocity.dot(gate.normal);
    let up_z = (race.drone.rotation * Vec3::Z).z;
    if depth <= 0. || closing < 2. || depth > 6. || up_z < 0.15 {
        return;
    }
    let horizon = (depth / closing).min(0.55);
    let predicted_height = race.drone.position.z + race.drone.velocity.z * horizon;
    let error = predicted_height - gate.center.z;
    if error <= 0.12 {
        return;
    }
    // Limit excess lift as collective rises before the aircraft has banked.
    let trim = (8. * (error - 0.12)).min(7.);
    let collective = ((accel.z - trim + 1.5) * crate::HOVER / (GRAVITY * up_z)).clamp(0., 1.);
    command[1] = command[1].min(2. * collective - 1.);
}

pub(super) fn track(
    race: &Race,
    firmware: bool,
    pace: f32,
    motion_scale: f32,
    experimental_line: Option<&Line>,
) -> Option<(Vec3, Vec3)> {
    let target = race.gates[race.next_gate()];
    let position = race.drone.position;
    let line = if let Some(line) = experimental_line {
        line
    } else if firmware {
        SEASON2_FIRMWARE_LINE.get_or_init(Line::firmware)
    } else {
        SEASON2_LINE.get_or_init(|| {
            Line::season2(
                NATIVE_PACE,
                &[Crossing::default(); crate::LAYOUT_2026.len()],
            )
        })
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
    let predictive = firmware && predictive_leg(race);
    let (speed_scale, gains) = if predictive {
        (FIRMWARE_SPEED_SCALE[race.next_gate()], PREDICTIVE_TRACKING)
    } else {
        (1., REFERENCE_TRACKING)
    };
    let speed_scale = speed_scale * pace * motion_scale;
    let preview = line.look_ahead(
        index,
        (0.32 + 0.18 * here.speed * pace * motion_scale) * gains.preview,
    );
    let velocity_error = here.tangent * preview.speed * speed_scale - race.drone.velocity;
    let correction = here.position - position;
    let curvature_accel =
        0.75 * (here.speed * speed_scale).powi(2) * here.curvature * gains.curvature;
    let mut accel = gains.velocity * velocity_error + gains.position * correction + curvature_accel
        - race.drone.rotation * crate::drag(race.drone.rotation.conjugate() * race.drone.velocity)
            / crate::MASS_KG;
    if race.passed == 0 {
        // On takeoff, arrest the climb before the opening instead of letting
        // the launch carry the aircraft above the planned line.
        let depth = (target.center - position).dot(target.normal);
        let closing = race.drone.velocity.dot(target.normal);
        if (0.0..3.5).contains(&depth) && closing > 1. {
            let time = (depth / closing).min(0.65);
            let predicted_height = position.z + race.drone.velocity.z * time;
            let error = predicted_height - (target.center.z - 0.08);
            accel.z -= (3. * error).clamp(0., 3.);
        }
    }
    if predictive {
        // Command the change in target velocity before the aircraft reaches
        // the turn so that rate and rotor lag do not delay the maneuver.
        let delay = ATTITUDE_PREVIEW_SECONDS;
        let step = (race.drone.velocity.dot(here.tangent).max(1.) * delay).max(0.2);
        let near = line.look_ahead(index, step * 0.65);
        let far = line.look_ahead(index, step * 1.35);
        let preview_accel =
            (far.speed * far.tangent - near.speed * near.tangent) * speed_scale / (0.7 * delay);
        accel += 0.131834 * (preview_accel - curvature_accel);
    }
    if firmware && matches!(race.next_gate(), 2 | 3 | 8) {
        // Add forward thrust on longer legs where the aircraft is aligned,
        // with enough room before the gate and speed left in the plan.
        let depth = (target.center - position).dot(target.normal);
        let room = ((depth - 2.) / 2.).clamp(0., 1.);
        let alignment = (1. - correction.length() / 0.9).clamp(0., 1.);
        let straightness = (1. - here.curvature.length() / 0.22).clamp(0., 1.);
        let speed_deficit = (preview.speed * pace * motion_scale
            - race.drone.velocity.dot(here.tangent))
        .clamp(0., 5.);
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
    accel += heading
        * (gains.level_approach
            * level_approach(race)
            * (minimum_forward - accel.dot(heading)).max(0.));
    if firmware {
        let offset = position - target.center;
        let depth = -offset.dot(target.normal);
        let closing = race.drone.velocity.dot(target.normal);
        let minimum_closing = if race.next_gate() == 11 { 1.8 } else { 3. };
        // Gate three has little lateral margin on the second lap. Start
        // correcting before the faster line carries the aircraft to the edge.
        let (lateral_preview, lateral_margin) = if predictive && race.next_gate() == 2 {
            (4.5, 0.15)
        } else {
            (3.2, 0.28)
        };
        if (0.85..lateral_preview).contains(&depth) && closing > minimum_closing {
            // Reserve room inside the opening based on the predicted crossing
            // position, allowing time for the attitude and motor response.
            let until_plane = depth / closing;
            let response_time = (until_plane - 0.1).max(0.08);
            let lateral = offset.dot(target.side)
                + race.drone.velocity.dot(target.side) * until_plane
                + 0.5 * accel.dot(target.side) * response_time.powi(2);
            let outside = lateral - lateral.clamp(-lateral_margin, lateral_margin);
            let correction = (2. * outside / response_time.powi(2)).clamp(-6., 6.);
            accel -= target.side * correction * gains.lateral;
        }
        if predictive && (0.75..6.).contains(&depth) && closing > 2. {
            // Correct the predicted crossing height before committing to the
            // aperture. Use the time remaining after the attitude response.
            let time = (depth / closing).min(0.65);
            let response = (time - 0.16).max(0.12);
            let estimate =
                offset.z + race.drone.velocity.z * time + 0.5 * accel.z * response.powi(2);
            let excess = estimate - estimate.clamp(-0.18, 0.18);
            let correction = (2. * excess / response.powi(2)).clamp(-9., 9.);
            accel.z -= correction;
        }
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn firmware_height_guard_limits_collective_when_projected_above_the_gate() {
        let mut race = Race::grand_challenge_on(Course::Season2Practice2026);
        race.passed = 2;
        let gate = race.gates[race.next_gate()];
        race.drone.position = gate.center - 2. * gate.normal;
        race.drone.velocity = 5. * gate.normal + Vec3::Z;
        race.drone.rotation = Quat::IDENTITY;
        let desired = Vec3::Z * GRAVITY;
        let mut command = [0., 1., 0., 0.];

        gate_height_guard(&race, desired, &mut command);
        assert!((command[1] + 1.) / 2. < crate::HOVER);

        race.drone.velocity = 5. * gate.normal - Vec3::Z;
        command[1] = 1.;
        gate_height_guard(&race, desired, &mut command);
        assert_eq!(command[1], 1.);
    }

    #[test]
    fn firmware_line_reduces_estimated_time_and_clears_frames() {
        let centers = [Crossing::default(); crate::LAYOUT_2026.len()];
        let centered_line = Line::season2(FIRMWARE_PACE, &centers);
        let optimized = Line::firmware();
        assert!(optimized.estimated_time(&centers) < centered_line.estimated_time(&centers));
        assert!(optimized.clears_frames());
        for (index, gate) in Gate::on_course(Course::Season2Practice2026).enumerate() {
            let point = optimized.points[(index + 1) * SAMPLES_PER_LEG];
            let crossing = gate.basis(point.position - gate.center);
            assert!(crossing.x.abs() < 1.0e-4);
            assert!(crossing.y.abs() <= 0.1801);
            assert!(point.tangent.dot(gate.normal) >= 0.22_f32.cos() - 1.0e-5);
        }
    }
}
