pub mod camera;
mod ffi;

use glam::{Quat, Vec3};

pub const PHYSICS_HZ: usize = 240;
pub const CONTROL_HZ: usize = 30;
const DT: f32 = 1.0 / PHYSICS_HZ as f32;
pub const ARENA_MIN: Vec3 = Vec3::new(-10., -18., 0.);
pub const ARENA_MAX: Vec3 = Vec3::new(43., 20., 12.);
pub const START: Vec3 = Vec3::new(3., -10., 2.6);
const GRAVITY: f32 = 9.80665;
pub const MASS_KG: f32 = 0.966;
pub const HOVER: f32 = MASS_KG * GRAVITY / (4. * KT * RPM_RAD_MAX * RPM_RAD_MAX);
const KT: f32 = 4.8e-7;
const RPM_RAD_MAX: f32 = 5100.;
const RADIUS: f32 = 0.205;
const ARM_AXIS: f32 = 0.155 * std::f32::consts::FRAC_1_SQRT_2;
const YAW_MOMENT: f32 = 0.016;
const MOTOR_TAU: f32 = 0.032;
const INERTIA: Vec3 = Vec3::new(0.0036, 0.0038, 0.0062);
const X: [f32; 4] = [ARM_AXIS, ARM_AXIS, -ARM_AXIS, -ARM_AXIS];
const Y: [f32; 4] = [ARM_AXIS, -ARM_AXIS, -ARM_AXIS, ARM_AXIS];
const SPIN: [f32; 4] = [1., -1., 1., -1.];

pub const LAYOUT: [[f32; 4]; 12] = [
    [9., -10., 2.6, 0.],
    [18., -10., 3.1, 0.],
    [27., -9., 4., 0.34],
    [34., -4., 4.8, 0.89],
    [36., 4., 4.1, 1.61],
    [31., 11., 3.1, 2.5],
    [22., 13., 3.5, 3.08],
    [13., 12., 4.6, 3.37],
    [4., 8., 4.1, 3.8],
    [-2., 1., 3., 4.3],
    [-2.5, -6., 2.8, 5.],
    [3., -10., 2.6, 6.],
];

const W: f32 = 1.55;
const H: f32 = 1.36;
const BEAM: f32 = 0.22;
const DEPTH: f32 = 0.20;
pub const BARS: [(Vec3, Vec3); 4] = [
    (
        Vec3::new(-DEPTH, -W - BEAM, H),
        Vec3::new(DEPTH, W + BEAM, H + BEAM),
    ),
    (
        Vec3::new(-DEPTH, -W - BEAM, -H - BEAM),
        Vec3::new(DEPTH, W + BEAM, -H),
    ),
    (Vec3::new(-DEPTH, -W - BEAM, -H), Vec3::new(DEPTH, -W, H)),
    (Vec3::new(-DEPTH, W, -H), Vec3::new(DEPTH, W + BEAM, H)),
];

#[derive(Clone, Copy)]
pub struct Gate {
    pub center: Vec3,
    pub normal: Vec3,
    pub side: Vec3,
}

impl Gate {
    pub fn all() -> [Self; LAYOUT.len()] {
        LAYOUT.map(|[x, y, z, yaw]| {
            let (sin, cos) = yaw.sin_cos();
            Self {
                center: Vec3::new(x, y, z),
                normal: Vec3::new(cos, sin, 0.),
                side: Vec3::new(-sin, cos, 0.),
            }
        })
    }

    pub fn basis(self, vector: Vec3) -> Vec3 {
        Vec3::new(self.normal.dot(vector), self.side.dot(vector), vector.z)
    }

    fn progress(self, start: Vec3, end: Vec3) -> (bool, bool) {
        let a = self.basis(start - self.center);
        let b = self.basis(end - self.center);
        if a.x.min(b.x) > DEPTH + RADIUS || a.x.max(b.x) < -DEPTH - RADIUS {
            return (false, false);
        }
        let delta = b - a;
        if BARS
            .iter()
            .any(|&(min, max)| ray_box(a, delta, min - RADIUS, max + RADIUS, 0., 1.).is_some())
        {
            return (true, false);
        }
        if a.x >= 0. || b.x < 0. {
            return (false, false);
        }
        let hit = a + delta * (-a.x / delta.x);
        (false, hit.y.abs() < W - RADIUS && hit.z.abs() < H - RADIUS)
    }
}

pub fn ray_box(
    origin: Vec3,
    direction: Vec3,
    min: Vec3,
    max: Vec3,
    low: f32,
    high: f32,
) -> Option<f32> {
    let safe_direction = direction.map(|value| if value.abs() < 1e-10 { 1e-10 } else { value });
    let a = (min - origin) / safe_direction;
    let b = (max - origin) / safe_direction;
    let entry = a.min(b).max_element().max(low);
    let exit = a.max(b).min_element().min(high);
    (entry <= exit).then_some(entry)
}

pub struct Drone {
    pub position: Vec3,
    pub velocity: Vec3,
    pub rotation: Quat,
    pub omega: Vec3,
    rotors: [f32; 4],
}

impl Drone {
    fn new() -> Self {
        Self {
            position: START,
            velocity: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            omega: Vec3::ZERO,
            rotors: [RPM_RAD_MAX * HOVER.sqrt(); 4],
        }
    }

    fn integrate(&mut self, action: [f32; 4]) {
        let wanted = Vec3::new(action[1], action[2], action[3]) * Vec3::new(5.5, 5.5, 3.4);
        let torque = (wanted - self.omega) * Vec3::new(0.040, 0.041, 0.027);
        let max_thrust = KT * RPM_RAD_MAX.powi(2);
        let correction: [f32; 4] = std::array::from_fn(|i| {
            (Y[i] * torque.x - X[i] * torque.y) / (4. * ARM_AXIS.powi(2))
                + SPIN[i] * torque.z / (4. * YAW_MOMENT)
        });
        let lo = correction.iter().copied().fold(0., f32::min);
        let hi = correction.iter().copied().fold(0., f32::max);
        let authority = (max_thrust / (hi - lo)).min(1.);
        let collective = (max_thrust * action[0])
            .max(-lo * authority)
            .min(max_thrust - hi * authority);
        let response = 1. - (-DT / MOTOR_TAU).exp();

        let airflow = self.rotation.conjugate() * self.velocity;
        let mut force = -(Vec3::new(0.09, 0.13, 0.16)
            + Vec3::new(0.045, 0.065, 0.075) * airflow.abs())
            * airflow;
        let mut moment = Vec3::ZERO;
        for i in 0..4 {
            let target = ((collective + correction[i] * authority).max(0.) / KT).sqrt();
            self.rotors[i] += response * (target - self.rotors[i]);
            let thrust = KT * self.rotors[i].powi(2);
            force.z += thrust;
            moment += Vec3::new(Y[i], -X[i], SPIN[i] * YAW_MOMENT) * thrust;
        }

        let accel = self.rotation * force / MASS_KG - Vec3::Z * GRAVITY;
        let net = moment
            - self.omega.cross(INERTIA * self.omega)
            - Vec3::new(0.002, 0.002, 0.003) * self.omega;
        self.omega += DT * net / INERTIA;
        self.rotation = (self.rotation * Quat::from_scaled_axis(self.omega * DT)).normalize();
        self.velocity += accel * DT;
        self.position += self.velocity * DT;
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum Status {
    Racing = 0,
    GateCollision = 1,
    Boundary = 2,
    Timeout = 3,
    Stalled = 4,
}

pub struct Race {
    pub drone: Drone,
    pub gates: [Gate; LAYOUT.len()],
    pub ticks: usize,
    pub passed: usize,
    pub laps: usize,
    pub lap_start: usize,
    pub last_lap: Option<f32>,
    pub best_lap: Option<f32>,
    pub status: Status,
    pub exposure: f32,
    start_gate: usize,
    timed_lap: bool,
    last_gate_tick: usize,
    initial_distance: f32,
    progress: f32,
}

impl Default for Race {
    fn default() -> Self {
        let gates = Gate::all();
        Self {
            drone: Drone::new(),
            gates,
            ticks: 0,
            passed: 0,
            laps: 0,
            lap_start: 0,
            last_lap: None,
            best_lap: None,
            status: Status::Racing,
            exposure: 1.,
            start_gate: 0,
            timed_lap: true,
            last_gate_tick: 0,
            initial_distance: gates[0].center.distance(START),
            progress: 0.,
        }
    }
}

fn lap_reward(elapsed: f32, best: Option<f32>) -> f32 {
    let improvement = best.map_or(0., |best| (best - elapsed).max(0.));
    20. + 8. * ((24. - elapsed) / 24.).clamp(-1., 1.) + (improvement * 0.18).min(3.)
}

impl Race {
    pub fn next_gate(&self) -> usize {
        (self.passed + self.start_gate) % self.gates.len()
    }

    pub fn from_start(side: f32, height: f32, exposure: f32) -> Self {
        let mut race = Self {
            exposure,
            ..Self::default()
        };
        race.drone.position.y += side;
        race.drone.position.z += height;
        race.initial_distance = race.gates[0].center.distance(race.drone.position);
        race
    }

    pub fn practice(
        start_gate: usize,
        side_offset: f32,
        height_offset: f32,
        exposure: f32,
    ) -> Self {
        let mut race = Self::default();
        race.start_gate = start_gate % race.gates.len();
        race.timed_lap = false;
        let gate = race.gates[race.start_gate];
        race.drone.position =
            gate.center - gate.normal * 4.0 + gate.side * side_offset + Vec3::Z * height_offset;
        race.drone.rotation = Quat::from_rotation_z(gate.normal.y.atan2(gate.normal.x));
        race.initial_distance = gate.center.distance(race.drone.position);
        race.exposure = exposure;
        race
    }

    pub fn step(&mut self, action: [f32; 4]) -> Option<f32> {
        if self.status != Status::Racing || action.iter().any(|x| !x.is_finite()) {
            return None;
        }
        let action = [
            action[0].clamp(0., 1.),
            action[1].clamp(-1., 1.),
            action[2].clamp(-1., 1.),
            action[3].clamp(-1., 1.),
        ];
        let mut reward = 0.;
        for _ in 0..(PHYSICS_HZ / CONTROL_HZ) {
            let previous = self.drone.position;
            self.drone.integrate(action);
            self.ticks += 1;
            reward -= 0.20 * DT;
            let current = self.drone.position;
            let target = self.next_gate();
            let mut passed = false;

            if current.cmplt(ARENA_MIN + RADIUS).any() || current.cmpgt(ARENA_MAX - RADIUS).any() {
                self.status = Status::Boundary;
            } else {
                for (i, gate) in self.gates.iter().enumerate() {
                    let (collision, crossed) = gate.progress(previous, current);
                    if collision {
                        self.status = Status::GateCollision;
                        break;
                    }
                    passed |= crossed && i == target;
                }
            }
            if self.status != Status::Racing {
                reward -= 18.;
                break;
            }

            let distance = self.gates[target].center.distance(current);
            let progress = if passed {
                1.
            } else {
                (1. - distance / self.initial_distance).clamp(-1., 1.)
            };
            reward += 0.8 * (progress - self.progress);
            self.progress = progress;
            if passed {
                self.passed += 1;
                self.last_gate_tick = self.ticks;
                reward += 2.;
                self.initial_distance = self.gates[self.next_gate()]
                    .center
                    .distance(current)
                    .max(0.1);
                self.progress = 0.;
                if target + 1 == self.gates.len() {
                    if self.timed_lap {
                        let elapsed = (self.ticks - self.lap_start) as f32 * DT;
                        reward += lap_reward(elapsed, self.best_lap);
                        self.best_lap =
                            Some(self.best_lap.map_or(elapsed, |best| best.min(elapsed)));
                        self.last_lap = Some(elapsed);
                        self.laps += 1;
                    }
                    self.lap_start = self.ticks;
                    self.timed_lap = true;
                }
            }
            if self.ticks - self.last_gate_tick >= 14 * PHYSICS_HZ {
                self.status = Status::Stalled;
                reward -= 10.;
                break;
            }
            if self.ticks >= 180 * PHYSICS_HZ {
                self.status = Status::Timeout;
                break;
            }
        }
        Some(reward)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hover_and_motor_lag() {
        let mut drone = Drone::new();
        for _ in 0..PHYSICS_HZ * 3 {
            drone.integrate([HOVER, 0., 0., 0.]);
        }
        assert!(drone.position.distance(START) < 0.002);
        let original = drone.rotors[0];
        drone.integrate([1., 0., 0., 0.]);
        assert!(drone.rotors[0] > original && drone.rotors[0] < RPM_RAD_MAX);
        assert!(drone.velocity.z > 0.);
    }

    #[test]
    fn individual_axes_and_quaternion_norm() {
        for axis in 1..=3 {
            let mut drone = Drone::new();
            let mut action = [HOVER, 0., 0., 0.];
            action[axis] = 0.5;
            for _ in 0..100 {
                drone.integrate(action);
            }
            assert!(drone.omega[axis - 1] > 0.5);
            assert!((drone.rotation.length() - 1.).abs() < 1e-5);
        }
    }

    #[test]
    fn gate_intersection_requires_direction_and_clearance() {
        let gate = Gate::all()[0];
        let a = gate.center - Vec3::X;
        let b = gate.center + Vec3::X;
        assert_eq!(gate.progress(a, b), (false, true));
        assert_eq!(gate.progress(b, a), (false, false));
        assert_eq!(
            gate.progress(a + W * Vec3::Y, b + W * Vec3::Y),
            (true, false)
        );
        assert_eq!(
            gate.progress(a + 3. * Vec3::Y, b + 3. * Vec3::Y),
            (false, false)
        );
        assert_eq!(gate.progress(a, gate.center), (false, true));
        assert_eq!(gate.progress(gate.center, b), (false, false));
    }

    #[test]
    fn ordered_laps_repeat_and_reward_improves() {
        let mut race = Race::default();
        let mut times = Vec::new();
        for pace in [220, 180] {
            for i in 0..race.gates.len() {
                let gate = race.gates[i];
                race.ticks += pace;
                race.drone.position = gate.center - gate.normal * 0.04;
                race.drone.velocity = gate.normal * 5.;
                race.step([HOVER, 0., 0., 0.]).unwrap();
            }
            times.push(race.last_lap.unwrap());
        }
        assert_eq!(race.laps, 2);
        assert_eq!(race.next_gate(), 0);
        assert_eq!(race.best_lap, Some(times[1]));
        assert!(lap_reward(times[1], Some(times[0])) > lap_reward(times[0], None));
        assert!(lap_reward(26., Some(28.)) > lap_reward(26., Some(26.)));
        let no_delay = lap_reward(40., None) + lap_reward(36., Some(40.));
        let delay = lap_reward(60., None) + lap_reward(36., Some(60.)) - 0.20 * 20.;
        assert!(no_delay > delay);
    }

    #[test]
    fn practice_partial_lap_does_not_set_a_record() {
        let mut race = Race::practice(LAYOUT.len() - 1, 0., 0., 1.);
        let gate = race.gates[race.next_gate()];
        race.drone.position = gate.center - gate.normal * 0.04;
        race.drone.velocity = gate.normal * 5.;
        race.step([HOVER, 0., 0., 0.]).unwrap();
        assert_eq!(race.next_gate(), 0);
        assert_eq!(race.laps, 0);
        assert_eq!(race.best_lap, None);
        assert!(race.lap_start > 0);
    }

    #[test]
    fn stationary_hover_eventually_stalls() {
        let mut race = Race {
            ticks: 14 * PHYSICS_HZ - 1,
            ..Race::default()
        };
        assert!(race.step([HOVER, 0., 0., 0.]).unwrap() <= -10.);
        assert_eq!(race.status, Status::Stalled);
    }

    #[test]
    fn collision_and_timeout_are_distinct() {
        let mut race = Race::default();
        race.drone.position.z = RADIUS - 0.01;
        assert!(race.step([HOVER, 0., 0., 0.]).unwrap() < -17.);
        assert_eq!(race.status, Status::Boundary);
        assert!(race.step([HOVER, 0., 0., 0.]).is_none());
        let mut race = Race {
            ticks: 180 * PHYSICS_HZ - 1,
            last_gate_tick: 179 * PHYSICS_HZ,
            ..Race::default()
        };
        race.step([HOVER, 0., 0., 0.]).unwrap();
        assert_eq!(race.status, Status::Timeout);
    }
}
