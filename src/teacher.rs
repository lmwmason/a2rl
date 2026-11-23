//! Training-only expert. Its state and gate path are never input to the RGB policy.
mod line;

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
    pace: f32,
    sprint_ticks: u8,
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
            pace: 1.,
            sprint_ticks: 0,
        }
    }

    pub fn with_flow(mut self, flow: bool) -> Self {
        self.flow = flow;
        self
    }

    pub(crate) fn set_pace(&mut self, pace: f32) -> bool {
        if !(0.60..=1.).contains(&pace) {
            return false;
        }
        self.pace = pace;
        true
    }

    pub fn external_action(&mut self, race: &Race) -> [f32; 4] {
        self.control(race, true)
    }

    pub fn action(&mut self, race: &Race) -> [f32; 4] {
        self.control(race, false)
    }

    fn control(&mut self, race: &Race, firmware: bool) -> [f32; 4] {
        if self.flow && race.grand_challenge && race.course == Course::Season2Practice2026 {
            if let Some((accel, heading)) = line::track(race, firmware, self.pace) {
                let horizontal_limit = if !firmware {
                    17.
                } else if line::is_ladder_leg(race.next_gate()) {
                    23.
                } else if line::is_fast_leg(race.next_gate()) {
                    38.
                } else {
                    27.
                };
                let accel = accel.with_z(0.).clamp_length_max(horizontal_limit)
                    + Vec3::Z * (GRAVITY + accel.z.clamp(-5.5, 8.));
                let mut command = rate_action(race, accel, heading);
                protect_level_pitch(race, &mut command);
                if firmware {
                    reinforce_braking(race, accel, &mut command);
                    if self.pace >= 0.97 {
                        self.sprint(race, &mut command);
                    }
                    line::gate_height_guard(race, accel, &mut command);
                }
                // Lift off with a brief, repeatable pulse for the low first gate.
                // Return to path tracking once the climb carries us clear of the pad.
                if race.passed == 0 {
                    let height = race.drone.position.z;
                    let climb = race.drone.velocity.z.max(0.);
                    if height < 0.55 {
                        command[1] = if height + climb * climb / 16. < 0.95 {
                            0.1
                        } else {
                            command[1].min(0.1)
                        };
                    } else if climb > 0. {
                        let excess = height + 0.2 * climb - race.gates[0].center.z;
                        command[1] = (command[1] - (0.28 * excess).clamp(0., 0.24)).max(-1.);
                    }
                }
                return command;
            }
        }
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
        let mut command = rate_action(race, accel, heading);
        if self.flow && season2 && race.grand_challenge {
            protect_level_pitch(race, &mut command);
        }
        command
    }

    fn sprint(&mut self, race: &Race, command: &mut [f32; 4]) {
        // The exit from the Split-S opens onto the long straight to gate 9.
        let target = race.next_gate();
        if target != 8 {
            self.sprint_ticks = 0;
            return;
        }
        let gate = race.gates[target];
        let drone = &race.drone;
        let offset = gate.center - drone.position;
        let depth = offset.dot(gate.normal);
        let thrust_up = (drone.rotation * Vec3::Z).z;
        // Leave the entry controller time to regain the aperture after the
        // pulse, and wait until most thrust is directed along the straight.
        if self.sprint_ticks < 2
            && (4.8..6.4).contains(&depth)
            && offset.dot(gate.side).abs() < 0.7
            && offset.z.abs() < 0.25
            && (-1.5..0.6).contains(&drone.velocity.z)
            && (0.28..0.55).contains(&thrust_up)
            && (4.0..9.0).contains(&drone.velocity.dot(gate.normal))
        {
            command[1] = 1.;
            self.sprint_ticks += 1;
        }
    }
}

fn reinforce_braking(race: &Race, accel: Vec3, command: &mut [f32; 4]) {
    let travel = race.drone.velocity.truncate();
    let direction = travel.normalize_or_zero();
    let up = race.drone.rotation * Vec3::Z;
    let braking_demand = -accel.truncate().dot(direction);
    let braking_alignment = -up.truncate().dot(direction);
    if travel.length_squared() < 9. || braking_demand < 2. || braking_alignment < 0.25 || up.z < 0.3
    {
        return;
    }

    // Only use extra collective once its horizontal component opposes travel.
    // Bound vertical lift while the craft is still rotating into the turn.
    let braking_thrust = braking_demand / braking_alignment;
    let vertical_limit = (accel.z + 1.5) / up.z;
    let throttle = (braking_thrust.min(vertical_limit) * HOVER / GRAVITY).clamp(0., 1.);
    let boost_limit = 0.16 * (1. - line::level_approach(race));
    command[1] += (2. * throttle - 1. - command[1]).clamp(0., boost_limit);
}

fn protect_level_pitch(race: &Race, command: &mut [f32; 4]) {
    let rotation = race.drone.rotation;
    let forward_z = (rotation * Vec3::X).z;
    let forward_z_rate = (rotation * race.drone.omega.cross(Vec3::X)).z;
    // Brake the upward angular motion before reaching level, accounting for the
    // effect of yaw while the aircraft is banked.
    let horizon = forward_z + 0.24 * forward_z_rate;
    let correction = line::level_approach(race) * (horizon + 0.065).max(0.) * 14.;
    command[3] = (command[3] + correction / 5.5).clamp(-1., 1.);
}

fn rate_action(race: &Race, accel: Vec3, heading: Vec3) -> [f32; 4] {
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

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn firmware_braking_only_adds_thrust_when_it_opposes_travel() {
        let mut race = Race::grand_challenge_on(Course::Season2Practice2026);
        race.drone.velocity = Vec3::X * 5.;
        let desired = Vec3::new(-4., 0., GRAVITY);
        let mut command = [0., -0.9, 0., 0.];
        race.drone.rotation = Quat::from_rotation_y(30_f32.to_radians());
        reinforce_braking(&race, desired, &mut command);
        assert_eq!(command[1], -0.9);

        race.drone.rotation = Quat::from_rotation_y(-30_f32.to_radians());
        reinforce_braking(&race, desired, &mut command);
        assert!(command[1] > -0.9);
        assert!(command[1] <= -0.9 + 0.16);
    }

    #[test]
    fn firmware_sprint_ends_before_the_gate_and_rearms_on_the_next_lap() {
        let mut race = Race::grand_challenge_on(Course::Season2Practice2026);
        let mut teacher = Teacher::new(&race, false);
        race.passed = 8;
        let gate = race.gates[race.next_gate()];
        race.drone.position = gate.center - gate.normal * 5.6;
        race.drone.velocity = gate.normal * 7.;
        race.drone.rotation = Quat::from_rotation_y(60_f32.to_radians());

        for expected in [1., 1., 0.] {
            let mut command = [0.; 4];
            teacher.sprint(&race, &mut command);
            assert_eq!(command[1], expected);
        }
        race.passed = 9;
        teacher.sprint(&race, &mut [0.; 4]);

        race.passed = 20;
        race.drone.position = gate.center - gate.normal * 3.;
        let mut command = [0.; 4];
        teacher.sprint(&race, &mut command);
        assert_eq!(command[1], 0.);
        race.drone.position = gate.center - gate.normal * 5.6;
        teacher.sprint(&race, &mut command);
        assert_eq!(command[1], 1.);
    }

    #[test]
    fn season2_launch_and_final_gate_orientation() {
        let race = Race::grand_challenge_on(Course::Season2Practice2026);
        let mut teacher = Teacher::new(&race, false).with_flow(true);
        assert!(teacher.external_action(&race)[1] <= 0.1);
        let ladder = &race.gates[race.gates.len() - 2..];
        assert!(ladder[0].center.z > ladder[1].center.z);
        assert!(ladder
            .iter()
            .all(|gate| gate.normal.x < 0. && gate.normal.y < 0.));
    }
}
