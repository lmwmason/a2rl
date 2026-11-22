//! Training-only expert. Its state and gate path are never input to the RGB policy.
use glam::{Mat3, Quat, Vec3};

use crate::{Course, Race, HOVER};

const GRAVITY: f32 = 9.80665;
// 55% arc-length anchors between consecutive gates in 4TU MonoRace Log 5,
// after converting NED to the simulator coordinate convention.
const LOG5_LINE: [Vec3; 11] = [
    Vec3::new(22.69, 22.36, 1.26),
    Vec3::new(38.76, 19.59, 1.06),
    Vec3::new(55.19, 21.71, 2.58),
    Vec3::new(74.89, 18.43, 4.01),
    Vec3::new(88.68, 16.79, 3.03),
    Vec3::new(75.70, 15.63, 0.98),
    Vec3::new(61.23, 9.28, 1.27),
    Vec3::new(45.31, 9.53, 0.96),
    Vec3::new(27.13, 10.10, 1.34),
    Vec3::new(12.30, 8.92, 1.68),
    Vec3::new(10.20, 20.59, 0.98),
];

#[derive(Clone, Copy)]
enum Phase {
    Launch,
    Exit,
    Transit,
    Approach,
    Cross,
    RecoverOutside,
    RecoverBehind,
}

pub struct Teacher {
    passed: usize,
    phase: Phase,
    recovery_side: f32,
    flow: bool,
}

impl Teacher {
    pub fn new(race: &Race, after_gate: bool) -> Self {
        let phase = if after_gate {
            Phase::Exit
        } else if race.drone.position.z < 0.5 {
            Phase::Launch
        } else {
            Phase::Approach
        };
        Self {
            passed: race.passed,
            phase,
            recovery_side: 1.,
            flow: false,
        }
    }

    pub fn with_flow(mut self, flow: bool) -> Self {
        self.flow = flow;
        self
    }

    pub fn action(&mut self, race: &Race) -> [f32; 4] {
        let next = race.next_gate();
        let gate = race.gates[next];
        let previous_index = (next + race.gates.len() - 1) % race.gates.len();
        let previous = race.gates[previous_index];
        let incoming = (gate.center - previous.center).normalize_or_zero();
        let straight = incoming.dot(previous.normal).min(incoming.dot(gate.normal)) > 0.65;
        let following = race.gates[(next + 1) % race.gates.len()];
        let outgoing = (following.center - gate.center).normalize_or_zero();
        let carry = straight
            && outgoing
                .dot(gate.normal)
                .min(outgoing.dot(following.normal))
                > 0.65;
        let entry = gate.center - gate.normal * 2.6;
        if race.passed != self.passed {
            self.passed = race.passed;
            self.phase = Phase::Exit;
        }

        let position = race.drone.position;
        let depth = (position - gate.center).dot(gate.normal);
        let (arena_min, arena_max) = race.course.bounds();
        let season2 = race.course == Course::Season2Practice2026;
        if matches!(self.phase, Phase::Approach | Phase::Cross) && depth > 0.65 {
            let side = (position - gate.center).dot(gate.side).signum();
            let trial = gate.center + gate.side * side * 2.5;
            self.recovery_side =
                if trial.cmpgt(arena_max - 0.4).any() || trial.cmplt(arena_min + 0.4).any() {
                    -side
                } else {
                    side
                };
            self.phase = Phase::RecoverOutside;
        }
        let (target, heading, speed_limit) = loop {
            let (point, direction, tolerance, speed, following) = match self.phase {
                Phase::Launch => {
                    let (start, heading) = race.course.start();
                    (
                        start,
                        Quat::from_rotation_z(heading.to_radians()) * Vec3::X,
                        0.20,
                        3.,
                        Phase::Approach,
                    )
                }
                Phase::Exit => (
                    previous.center + previous.normal * 2.0,
                    previous.normal,
                    0.43,
                    4.,
                    Phase::Transit,
                ),
                Phase::Transit => {
                    if self.flow && straight {
                        self.phase = Phase::Approach;
                        continue;
                    }
                    let anchor = if season2 {
                        // Travel clear of the lower gate before climbing into the stacked aperture.
                        if (gate.center - previous.center).truncate().length() < 0.5 {
                            Vec3::new(entry.x, entry.y, gate.center.z)
                        } else {
                            (previous.center + previous.normal * 2.0 + entry) * 0.5
                        }
                    } else {
                        LOG5_LINE[previous_index]
                    };
                    // Keep the split-S turn outside the frame while descending.
                    let delta = anchor - position;
                    let heading = Vec3::new(delta.x, delta.y, 0.)
                        .try_normalize()
                        .unwrap_or(gate.normal);
                    (
                        anchor,
                        heading,
                        if self.flow { 1.45 } else { 0.8 },
                        5.3,
                        Phase::Approach,
                    )
                }
                Phase::Approach => (
                    entry,
                    gate.normal,
                    0.29,
                    if self.flow && carry { 8. } else { 6. },
                    Phase::Cross,
                ),
                Phase::Cross => {
                    break (
                        gate.center + 2.0 * gate.normal,
                        gate.normal,
                        if self.flow && carry { 7.5 } else { 6. },
                    )
                }
                Phase::RecoverOutside => (
                    gate.center + gate.side * self.recovery_side * 2.5 + gate.normal * 1.8,
                    gate.normal,
                    0.53,
                    3.,
                    Phase::RecoverBehind,
                ),
                Phase::RecoverBehind => (
                    gate.center + gate.side * self.recovery_side * 2.5 - gate.normal * 2.2,
                    -gate.normal,
                    0.45,
                    3.,
                    Phase::Approach,
                ),
            };
            let passed_exit = matches!(self.phase, Phase::Exit)
                && (position - previous.center).dot(previous.normal)
                    > if self.flow {
                        if straight {
                            0.95
                        } else {
                            1.60
                        }
                    } else {
                        2.05
                    };
            let lateral = gate.center - position + gate.normal * depth;
            let lateral_velocity =
                race.drone.velocity - gate.normal * race.drone.velocity.dot(gate.normal);
            let aligned_approach = self.flow
                && matches!(self.phase, Phase::Approach)
                && depth > if carry { -3.4 } else { -2.95 }
                && lateral.length() < if carry { 0.29 } else { 0.24 }
                && lateral_velocity.length() < 1.0;
            if position.distance(point) > tolerance && !passed_exit && !aligned_approach {
                break (point, direction, speed);
            }
            self.phase = following;
        };

        let desired_velocity =
            if self.flow && (carry || season2) && matches!(self.phase, Phase::Cross) {
                let lateral = gate.center - position + gate.normal * depth;
                gate.normal * speed_limit + lateral * 2.5
            } else {
                ((target - position) * 1.5).clamp_length_max(speed_limit)
            };
        let error = (desired_velocity - race.drone.velocity) * 2.6;
        let accel = Vec3::new(error.x, error.y, 0.).clamp_length_max(if self.flow && carry {
            7.2
        } else {
            5.2
        }) + Vec3::Z * (GRAVITY + error.z.clamp(-5.5, 6.0));
        let up = accel.normalize();
        let left = up.cross(heading).normalize();
        let desired = Quat::from_mat3(&Mat3::from_cols(left.cross(up), left, up));
        let rotation_error = race.drone.rotation.conjugate() * desired;
        let rotation_error = if rotation_error.w < 0. {
            -rotation_error
        } else {
            rotation_error
        };
        let rates = rotation_error.to_scaled_axis() * 6.0 / Vec3::new(5.5, 5.5, 3.4);
        let throttle = (accel.dot(race.drone.rotation * Vec3::Z) / GRAVITY * HOVER).clamp(0., 1.);
        [-rates.z, 2. * throttle - 1., rates.x, rates.y].map(|axis| axis.clamp(-1., 1.))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{mode2_acro, Status};

    #[test]
    fn expert_follows_logged_route_for_both_laps() {
        let mut race = Race::grand_challenge();
        let mut teacher = Teacher::new(&race, false);
        while race.status == Status::Racing {
            let action = teacher.action(&race);
            race.step(mode2_acro(action)).unwrap();
        }
        assert_eq!(
            race.status,
            Status::Finished,
            "passed {}, pos {:?}",
            race.passed,
            race.drone.position
        );
        assert_eq!(race.passed, 22);
    }

    #[test]
    fn flow_teacher_finishes_the_course_while_carrying_speed() {
        let mut race = Race::grand_challenge();
        let mut teacher = Teacher::new(&race, false).with_flow(true);
        while race.status == Status::Racing {
            let action = teacher.action(&race);
            race.step(mode2_acro(action)).unwrap();
        }
        assert_eq!((race.status, race.passed), (Status::Finished, 22));
        assert!(race.official_seconds().unwrap() < 95.);
    }

    #[test]
    fn season2_flow_teacher_finishes_both_laps() {
        let mut race = Race::grand_challenge_on(Course::Season2Practice2026);
        let mut teacher = Teacher::new(&race, false).with_flow(true);
        while race.status == Status::Racing {
            race.step(mode2_acro(teacher.action(&race))).unwrap();
        }
        assert_eq!((race.status, race.passed), (Status::Finished, 22));
        assert!(race.official_seconds().unwrap() < 80.);
    }
}
