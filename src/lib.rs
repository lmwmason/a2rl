pub mod camera;
mod ffi;
pub mod teacher;

use glam::{Quat, Vec3};
use parry3d::{
    na::{Isometry3, Vector3},
    query::{cast_shapes, ShapeCastOptions},
    shape::{Ball, Cuboid},
};

pub const PHYSICS_HZ: usize = 240;
pub const PHYSICS_MODEL_VERSION: u32 = 2;
pub const REWARD_MODEL_VERSION: u32 = 2;
pub const PACE_REWARD_VERSION: u32 = 3;
pub const CONTROL_HZ: usize = 30;
pub const GRAND_CHALLENGE_LAPS: usize = 2;
const DT: f32 = 1.0 / PHYSICS_HZ as f32;
pub const ARENA_MIN: Vec3 = Vec3::new(1., -1., 0.);
pub const ARENA_MAX: Vec3 = Vec3::new(95., 27., 6.2);
pub const START: Vec3 = Vec3::new(8., 22., 1.47);
const START_HEADING: f32 = 13.;
const GRAVITY: f32 = 9.80665;
pub const MASS_KG: f32 = 0.966;
pub const HOVER: f32 = MASS_KG * GRAVITY / (4. * KT * RPM_RAD_MAX * RPM_RAD_MAX);
pub const HOVER_STICK: f32 = 2. * HOVER - 1.;
const KT: f32 = 1.436723e-6;
const RPM_RAD_MAX: f32 = 3100.;
const RADIUS: f32 = 0.205;
const ARM_AXIS: f32 = 0.155 * std::f32::consts::FRAC_1_SQRT_2;
const YAW_MOMENT: f32 = 0.016;
const MOTOR_TAU: f32 = 0.030;
const INERTIA: Vec3 = Vec3::new(0.0036, 0.0038, 0.0062);
// Fit on the first airborne segment of MonoRace Log 8. Logs 5 and 7 are held out.
const LINEAR_DRAG: Vec3 = Vec3::new(0.3935197, 0.7070415, 0.);
const QUADRATIC_DRAG: Vec3 = Vec3::new(0.002257, 0., 0.);
const X: [f32; 4] = [ARM_AXIS, ARM_AXIS, -ARM_AXIS, -ARM_AXIS];
const Y: [f32; 4] = [ARM_AXIS, -ARM_AXIS, -ARM_AXIS, ARM_AXIS];
const SPIN: [f32; 4] = [1., -1., 1., -1.];

// Mode 2 transmitter axes, with right and up positive on each stick.
pub fn mode2_acro([left_x, left_y, right_x, right_y]: [f32; 4]) -> [f32; 4] {
    [(left_y + 1.) * 0.5, right_x, right_y, -left_x]
}

fn drag(airspeed: Vec3) -> Vec3 {
    -(LINEAR_DRAG + QUADRATIC_DRAG * airspeed.abs()) * airspeed
}

fn thrust_coefficient(mean_rotor_speed: f32, axial_airspeed: f32, horizontal_airspeed: f32) -> f32 {
    let tip = (0.0485775 * mean_rotor_speed).max(1.);
    let axial_angle = (-axial_airspeed).atan2(tip);
    let horizontal_angle = horizontal_airspeed.powi(2).atan2(tip.powi(2));
    (KT + 4.492359e-6 * axial_angle + 3.180793e-6 * horizontal_angle).max(KT * 0.05)
}

fn axial_specific_force(omega_squared: f32, mean_speed: f32, axial: f32, horizontal: f32) -> f32 {
    thrust_coefficient(mean_speed, axial, horizontal) * omega_squared / MASS_KG
}

// TU Delft MAVLab's Grand Challenge flight plan, transformed to +Z up, +Y left.
// github.com/tudelft/optimal_quad_control_RL/blob/a2rl2025/flight_plans/grand_challenge.json
pub const LAYOUT: [[f32; 4]; 11] = [
    [14., 25., 1.47, 35.],
    [30., 19., 1.47, -15.],
    [46., 22., 1.47, 10.],
    [63., 20., 4.17, 0.],
    [85., 18., 4.17, 0.],
    [85., 18., 1.47, 180.],
    [68., 13., 1.47, -165.],
    [55., 7., 1.47, 180.],
    [37., 12., 1.47, 180.],
    [19., 7., 1.47, 180.],
    [9., 14., 1.47, 90.],
];

// Top-down gate poses digitized from FlyBy's 2026 competition overlay (IMAV 2026-1,
// Fig. 5). Fig. 2 supplies the double-aperture geometry. Heights are estimates.
pub const LAYOUT_2026: [[f32; 4]; 12] = [
    [12.5, 2., 1.47, 180.],
    [6.5, 6., 1.47, 135.],
    [5.5, 14., 1.47, 120.],
    [2.5, 24., 1.47, 90.],
    [7.5, 30., 1.47, -10.],
    [12.2, 22., 1.47, 0.],
    [17.5, 30., 3.97, 80.],
    [17.5, 30., 1.47, -100.],
    [18.5, 22., 1.47, -80.],
    [20.5, 14., 1.47, -100.],
    [18.5, 6., 3.97, -135.],
    [18.5, 6., 1.47, -135.],
];

#[derive(Clone, Copy, Default, PartialEq, Eq)]
pub enum Course {
    #[default]
    GrandChallenge2025,
    Season2Practice2026,
}

impl Course {
    pub fn layout(self) -> &'static [[f32; 4]] {
        match self {
            Self::GrandChallenge2025 => &LAYOUT,
            Self::Season2Practice2026 => &LAYOUT_2026,
        }
    }

    pub fn start(self) -> (Vec3, f32) {
        match self {
            Self::GrandChallenge2025 => (START, START_HEADING),
            Self::Season2Practice2026 => (Vec3::new(18.7, 2., 1.47), 180.),
        }
    }

    pub fn bounds(self) -> (Vec3, Vec3) {
        match self {
            Self::GrandChallenge2025 => (ARENA_MIN, ARENA_MAX),
            Self::Season2Practice2026 => (Vec3::new(-1., -2., 0.), Vec3::new(26., 36., 6.2)),
        }
    }

    fn extra_frames(self) -> &'static [Gate] {
        match self {
            Self::GrandChallenge2025 => &[LOWER_FRAME],
            Self::Season2Practice2026 => &[],
        }
    }

    const fn gate_half_depth(self) -> f32 {
        match self {
            Self::GrandChallenge2025 => DEPTH,
            // From the 0.1322825 m front-to-back extent of Berkeley's gate.usd.
            Self::Season2Practice2026 => SEASON2_GATE_DEPTH,
        }
    }

    fn gate_bars(self) -> &'static [(Vec3, Vec3); 4] {
        match self {
            Self::GrandChallenge2025 => &BARS,
            Self::Season2Practice2026 => &SEASON2_BARS,
        }
    }
}

const W: f32 = 0.75;
const H: f32 = 0.75;
const BEAM: f32 = 0.60;
const DEPTH: f32 = 0.40;
const SEASON2_GATE_DEPTH: f32 = 0.066_141_27;
const LOWER_FRAME: Gate = Gate {
    center: Vec3::new(63., 20., 1.47),
    normal: Vec3::X,
    side: Vec3::Y,
};
const fn bars_with_depth(half_depth: f32) -> [(Vec3, Vec3); 4] {
    [
        (
            Vec3::new(-half_depth, -W - BEAM, H),
            Vec3::new(half_depth, W + BEAM, H + BEAM),
        ),
        (
            Vec3::new(-half_depth, -W - BEAM, -H - BEAM),
            Vec3::new(half_depth, W + BEAM, -H),
        ),
        (
            Vec3::new(-half_depth, -W - BEAM, -H),
            Vec3::new(half_depth, -W, H),
        ),
        (
            Vec3::new(-half_depth, W, -H),
            Vec3::new(half_depth, W + BEAM, H),
        ),
    ]
}

pub const BARS: [(Vec3, Vec3); 4] = bars_with_depth(DEPTH);
const SEASON2_BARS: [(Vec3, Vec3); 4] = bars_with_depth(SEASON2_GATE_DEPTH);

const FLOW_APPROACH_METERS: f32 = 4.;
const FLOW_SLOW_METERS_PER_SECOND: f32 = 1.5;
const FLOW_APPROACH_COST_PER_SECOND: f32 = 0.45;
const FLOW_CROSSING_SPEED_CAP: f32 = 10.;
const FLOW_CROSSING_BONUS: f32 = 0.60;

#[derive(Clone, Copy)]
pub struct Gate {
    pub center: Vec3,
    pub normal: Vec3,
    pub side: Vec3,
}

impl Gate {
    pub fn all() -> [Self; LAYOUT.len()] {
        LAYOUT.map(Self::from_pose)
    }

    fn from_pose([x, y, z, yaw]: [f32; 4]) -> Self {
        let (sin, cos) = yaw.to_radians().sin_cos();
        Self {
            center: Vec3::new(x, y, z),
            normal: Vec3::new(cos, sin, 0.),
            side: Vec3::new(-sin, cos, 0.),
        }
    }

    fn on_course(course: Course) -> impl Iterator<Item = Self> {
        course.layout().iter().copied().map(Self::from_pose)
    }

    pub fn basis(self, vector: Vec3) -> Vec3 {
        Vec3::new(self.normal.dot(vector), self.side.dot(vector), vector.z)
    }

    fn progress(self, start: Vec3, end: Vec3, course: Course) -> (bool, bool) {
        let a = self.basis(start - self.center);
        let b = self.basis(end - self.center);
        let reach = course.gate_half_depth() + RADIUS;
        if a.x.min(b.x) > reach || a.x.max(b.x) < -reach {
            return (false, false);
        }
        let delta = b - a;
        if course
            .gate_bars()
            .iter()
            .any(|&(min, max)| frame_contact(a, delta, min, max))
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

fn flow_alignment(direction: Vec3, normal: Vec3) -> f32 {
    ((direction.normalize_or_zero().dot(normal) - 0.25) / 0.75).clamp(0., 1.)
}

fn approach_delay(gate: Gate, incoming: Vec3, outgoing: Vec3, offset: Vec3, velocity: Vec3) -> f32 {
    let distance = offset.length();
    if distance >= FLOW_APPROACH_METERS || offset.dot(gate.normal) <= 0. {
        return 0.;
    }
    let closing_speed = velocity.dot(offset) / distance;
    let slow_fraction = (1. - closing_speed / FLOW_SLOW_METERS_PER_SECOND).clamp(0., 1.);
    let proximity = 1. - distance / FLOW_APPROACH_METERS;
    let turn_weight =
        flow_alignment(incoming, gate.normal).min(flow_alignment(outgoing, gate.normal));
    DT * slow_fraction * proximity * turn_weight
}

fn frame_contact(origin: Vec3, travel: Vec3, min: Vec3, max: Vec3) -> bool {
    if ray_box(origin, travel, min - RADIUS, max + RADIUS, 0., 1.).is_none() {
        return false;
    }
    let position = Isometry3::translation(origin.x, origin.y, origin.z);
    let motion = Vector3::new(travel.x, travel.y, travel.z);
    let center = (min + max) * 0.5;
    let extent = (max - min) * 0.5;
    let frame = Isometry3::translation(center.x, center.y, center.z);
    cast_shapes(
        &position,
        &motion,
        &Ball::new(RADIUS),
        &frame,
        &Vector3::zeros(),
        &Cuboid::new(Vector3::new(extent.x, extent.y, extent.z)),
        ShapeCastOptions::with_max_time_of_impact(1.),
    )
    .map_or(true, |hit| hit.is_some())
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
            rotation: Quat::from_rotation_z(START_HEADING.to_radians()),
            omega: Vec3::ZERO,
            rotors: [RPM_RAD_MAX * HOVER.sqrt(); 4],
        }
    }

    fn integrate(&mut self, action: [f32; 4]) {
        let (force, torque) = self.body_wrench(action);
        let accel = self.rotation * force / MASS_KG - Vec3::Z * GRAVITY;
        let net = torque - self.omega.cross(INERTIA * self.omega) - self.angular_damping();
        self.omega += DT * net / INERTIA;
        self.rotation = (self.rotation * Quat::from_scaled_axis(self.omega * DT)).normalize();
        self.velocity += accel * DT;
        self.position += self.velocity * DT;
    }

    fn body_wrench(&mut self, action: [f32; 4]) -> (Vec3, Vec3) {
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
        let mut force = drag(airflow);
        for (rotor, correction) in self.rotors.iter_mut().zip(correction) {
            let target = ((collective + correction * authority).max(0.) / KT).sqrt();
            *rotor += response * (target - *rotor);
        }
        let mean_speed = self.rotors.iter().sum::<f32>() / 4.;
        let coefficient = thrust_coefficient(mean_speed, airflow.z, airflow.x.hypot(airflow.y));
        let mut moment = Vec3::ZERO;
        for i in 0..4 {
            let thrust = coefficient * self.rotors[i].powi(2);
            force.z += thrust;
            moment += Vec3::new(Y[i], -X[i], SPIN[i] * YAW_MOMENT) * thrust;
        }

        (force, moment)
    }

    fn angular_damping(&self) -> Vec3 {
        Vec3::new(0.002, 0.002, 0.003) * self.omega
    }

    fn rigid_state(&self) -> [f32; 13] {
        let mut state = [0.; 13];
        state[..3].copy_from_slice(&self.position.to_array());
        state[3..6].copy_from_slice(&self.velocity.to_array());
        state[6..10].copy_from_slice(&self.rotation.to_array());
        state[10..].copy_from_slice(&self.omega.to_array());
        state
    }

    fn set_rigid_state(&mut self, state: [f32; 13]) -> bool {
        let attitude = Quat::from_slice(&state[6..10]);
        if !state.iter().all(|value| value.is_finite()) || attitude.length_squared() < 0.5 {
            return false;
        }
        self.position = Vec3::from_slice(&state[..3]);
        self.velocity = Vec3::from_slice(&state[3..6]);
        self.rotation = attitude.normalize();
        self.omega = Vec3::from_slice(&state[10..]);
        true
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
    Finished = 5,
}

#[derive(Clone, Copy, Default, PartialEq, Eq)]
enum RewardProfile {
    #[default]
    Standard,
    Pace,
}

pub struct Race {
    pub drone: Drone,
    pub gates: Vec<Gate>,
    pub course: Course,
    pub ticks: usize,
    pub passed: usize,
    pub laps: usize,
    pub lap_start: usize,
    pub last_lap: Option<f32>,
    pub best_lap: Option<f32>,
    pub status: Status,
    pub exposure: f32,
    pub video_breakup: f32,
    pub video_seed: u32,
    pub approach_delay_seconds: f32,
    pub gate_speed_sum: f32,
    reward_profile: RewardProfile,
    start_gate: usize,
    timed_lap: bool,
    last_gate_tick: usize,
    initial_distance: f32,
    progress: f32,
    on_launch_pad: bool,
    grand_challenge: bool,
    first_gate_tick: Option<usize>,
}

impl Default for Race {
    fn default() -> Self {
        Self::on_course(Course::default())
    }
}

impl Race {
    fn on_course(course: Course) -> Self {
        let gates: Vec<_> = Gate::on_course(course).collect();
        let (position, yaw) = course.start();
        let mut drone = Drone::new();
        drone.position = position;
        drone.rotation = Quat::from_rotation_z(yaw.to_radians());
        let initial_distance = gates[0].center.distance(position);
        Self {
            drone,
            gates,
            course,
            ticks: 0,
            passed: 0,
            laps: 0,
            lap_start: 0,
            last_lap: None,
            best_lap: None,
            status: Status::Racing,
            exposure: 1.,
            video_breakup: 0.,
            video_seed: 1,
            approach_delay_seconds: 0.,
            gate_speed_sum: 0.,
            reward_profile: RewardProfile::default(),
            start_gate: 0,
            timed_lap: true,
            last_gate_tick: 0,
            initial_distance,
            progress: 0.,
            on_launch_pad: false,
            grand_challenge: false,
            first_gate_tick: None,
        }
    }
}

fn lap_reward(elapsed: f32, best: Option<f32>) -> f32 {
    let improvement = best.map_or(0., |best| (best - elapsed).max(0.));
    20. + 8. * ((24. - elapsed) / 24.).clamp(-1., 1.) + (improvement * 0.18).min(3.)
}

fn segment_pace_reward(previous: Gate, next: Gate, elapsed: f32) -> f32 {
    let distance = previous.center.distance(next.center);
    let heading_change = (1. - previous.normal.dot(next.normal)).clamp(0., 2.);
    let reference = (distance / 9.).max(0.35) + 0.8 * heading_change;
    0.9 * (reference / elapsed.max(DT)).ln().clamp(-1.6, 1.6)
}

impl Race {
    pub fn grand_challenge() -> Self {
        Self::grand_challenge_on(Course::default())
    }

    pub fn grand_challenge_on(course: Course) -> Self {
        let mut race = Self::ground_start_on(course, 0., 1.);
        race.grand_challenge = true;
        race
    }

    pub fn official_seconds(&self) -> Option<f32> {
        self.first_gate_tick
            .map(|start| (self.ticks - start) as f32 * DT)
    }

    pub fn lap_seconds(&self) -> f32 {
        if self.status == Status::Finished {
            return self.last_lap.unwrap_or_default();
        }
        if self.grand_challenge && self.first_gate_tick.is_none() {
            return 0.;
        }
        (self.ticks - self.lap_start) as f32 * DT
    }

    pub fn gate_frames(&self) -> impl Iterator<Item = &Gate> {
        self.gates.iter().chain(self.course.extra_frames())
    }

    pub fn next_gate(&self) -> usize {
        (self.passed + self.start_gate) % self.gates.len()
    }

    fn sprint_window(&self) -> bool {
        if self.status != Status::Racing || self.on_launch_pad {
            return false;
        }
        let target = self.next_gate();
        let gate = self.gates[target];
        let previous = self.gates[(target + self.gates.len() - 1) % self.gates.len()];
        let leg = gate.center - previous.center;
        let direction = leg.normalize_or_zero();
        let position = self.drone.position;
        let velocity = self.drone.velocity;
        let up = self.drone.rotation * Vec3::Z;
        let forward_speed = velocity.dot(direction);
        let distance = (gate.center - position).dot(direction);
        let offset = (gate.center - position) - direction * distance;
        // Full thrust is tried only during training. Reserve room to correct before the gate.
        direction
            .dot(previous.normal)
            .min(direction.dot(gate.normal))
            > 0.75
            && (position - previous.center).dot(previous.normal) > 1.5
            && distance > 6. + forward_speed * 0.45
            && distance < leg.length()
            && offset.length() < 2.0
            && forward_speed > 1.8
            && (velocity - direction * forward_speed).length() < 2.0
            && up.dot(direction) > 0.14
            && position.z > 0.75
            && position.z < self.course.bounds().1.z - 1.8
            && velocity.z < 1.2
    }

    pub fn from_start(side: f32, height: f32, exposure: f32) -> Self {
        Self::from_start_on(Course::default(), side, height, exposure)
    }

    fn from_start_on(course: Course, side: f32, height: f32, exposure: f32) -> Self {
        let mut race = Self {
            exposure,
            ..Self::on_course(course)
        };
        race.drone.position.y += side;
        race.drone.position.z += height;
        race.initial_distance = race.gates[0].center.distance(race.drone.position);
        race
    }

    pub fn ground_start(side: f32, exposure: f32) -> Self {
        Self::ground_start_on(Course::default(), side, exposure)
    }

    fn ground_start_on(course: Course, side: f32, exposure: f32) -> Self {
        let mut race = Self::from_start_on(course, side, 0., exposure);
        race.drone.position.z = RADIUS;
        race.drone.rotors = [0.; 4];
        race.initial_distance = race.gates[0].center.distance(race.drone.position);
        race.on_launch_pad = true;
        race
    }

    pub fn practice(
        start_gate: usize,
        side_offset: f32,
        height_offset: f32,
        exposure: f32,
    ) -> Self {
        Self::practice_on(
            Course::default(),
            start_gate,
            side_offset,
            height_offset,
            exposure,
        )
    }

    fn practice_on(
        course: Course,
        start_gate: usize,
        side_offset: f32,
        height_offset: f32,
        exposure: f32,
    ) -> Self {
        let mut race = Self::on_course(course);
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

    pub fn after_gate(
        gate_index: usize,
        side_offset: f32,
        height_offset: f32,
        speed: f32,
        exposure: f32,
    ) -> Self {
        Self::after_gate_on(
            Course::default(),
            gate_index,
            side_offset,
            height_offset,
            speed,
            exposure,
        )
    }

    fn after_gate_on(
        course: Course,
        gate_index: usize,
        side_offset: f32,
        height_offset: f32,
        speed: f32,
        exposure: f32,
    ) -> Self {
        let index = gate_index % course.layout().len();
        let mut race = Self::practice_on(course, index + 1, 0., 0., exposure);
        let gate = race.gates[index];
        race.drone.position =
            gate.center + 1.35 * gate.normal + side_offset * gate.side + height_offset * Vec3::Z;
        race.drone.rotation = Quat::from_rotation_z(gate.normal.y.atan2(gate.normal.x));
        race.drone.velocity = gate.normal * speed;
        race.initial_distance = race.gates[race.next_gate()]
            .center
            .distance(race.drone.position);
        race
    }

    pub fn step(&mut self, action: [f32; 4]) -> Option<f32> {
        self.step_with(action, |drone, input| {
            drone.integrate(input);
            true
        })
    }

    fn step_with(
        &mut self,
        action: [f32; 4],
        mut integrate: impl FnMut(&mut Drone, [f32; 4]) -> bool,
    ) -> Option<f32> {
        if self.status != Status::Racing || action.iter().any(|x| !x.is_finite()) {
            return None;
        }
        let action = [
            action[0].clamp(0., 1.),
            action[1].clamp(-1., 1.),
            action[2].clamp(-1., 1.),
            action[3].clamp(-1., 1.),
        ];
        let pace = self.reward_profile == RewardProfile::Pace;
        let mut reward = 0.;
        for _ in 0..(PHYSICS_HZ / CONTROL_HZ) {
            let previous = self.drone.position;
            let rotation = self.drone.rotation;
            if !integrate(&mut self.drone, action) {
                return None;
            }
            if self.on_launch_pad {
                if self.drone.position.z > RADIUS {
                    self.on_launch_pad = false;
                } else {
                    self.drone.position = previous;
                    self.drone.rotation = rotation;
                    self.drone.velocity = Vec3::ZERO;
                    self.drone.omega = Vec3::ZERO;
                }
            }
            self.ticks += 1;
            reward -= if pace { 0.32 } else { 0.20 } * DT;
            let current = self.drone.position;
            let target = self.next_gate();
            let mut passed = false;

            let (arena_min, arena_max) = self.course.bounds();
            if current.cmplt(arena_min + RADIUS).any() || current.cmpgt(arena_max - RADIUS).any() {
                self.status = Status::Boundary;
            } else {
                let mut hit_frame = false;
                for (i, gate) in self.gate_frames().enumerate() {
                    let (collision, crossed) = gate.progress(previous, current, self.course);
                    if collision {
                        hit_frame = true;
                        break;
                    }
                    passed |= crossed && i == target;
                }
                if hit_frame {
                    self.status = Status::GateCollision;
                }
            }
            if self.status != Status::Racing {
                reward -= if pace { 24. } else { 18. };
                break;
            }

            let gate = self.gates[target];
            let offset = gate.center - current;
            let distance = offset.length();
            let next = (target + 1) % self.gates.len();
            let progress = if passed {
                1.
            } else {
                (1. - distance / self.initial_distance).clamp(-1., 1.)
            };
            reward += 0.8 * (progress - self.progress);
            self.progress = progress;
            if passed {
                let crossing_speed = ((current - previous).dot(gate.normal) / DT).max(0.);
                self.gate_speed_sum += crossing_speed;
                let outgoing = self.gates[next].center - gate.center;
                let speed_fraction = (crossing_speed / FLOW_CROSSING_SPEED_CAP).min(1.);
                reward +=
                    FLOW_CROSSING_BONUS * speed_fraction * flow_alignment(outgoing, gate.normal);
                if pace && self.passed > 0 {
                    let previous_gate = (target + self.gates.len() - 1) % self.gates.len();
                    let elapsed = (self.ticks - self.last_gate_tick) as f32 * DT;
                    reward += segment_pace_reward(self.gates[previous_gate], gate, elapsed);
                }
                if self.passed == 0 && self.grand_challenge {
                    self.first_gate_tick = Some(self.ticks);
                    self.lap_start = self.ticks;
                }
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
                        reward += lap_reward(elapsed, if pace { None } else { self.best_lap });
                        self.best_lap =
                            Some(self.best_lap.map_or(elapsed, |best| best.min(elapsed)));
                        self.last_lap = Some(elapsed);
                        self.laps += 1;
                        if pace && self.laps == GRAND_CHALLENGE_LAPS {
                            reward += 12.;
                        }
                    }
                    self.lap_start = self.ticks;
                    self.timed_lap = true;
                    if self.grand_challenge && self.laps == GRAND_CHALLENGE_LAPS {
                        self.status = Status::Finished;
                        break;
                    }
                }
            } else if !self.on_launch_pad && distance < FLOW_APPROACH_METERS {
                let previous_gate = (target + self.gates.len() - 1) % self.gates.len();
                let incoming = gate.center - self.gates[previous_gate].center;
                let outgoing = self.gates[next].center - gate.center;
                let delay = approach_delay(gate, incoming, outgoing, offset, self.drone.velocity);
                self.approach_delay_seconds += delay;
                reward -= FLOW_APPROACH_COST_PER_SECOND * delay;
            }
            if self.ticks - self.last_gate_tick >= 14 * PHYSICS_HZ {
                self.status = Status::Stalled;
                reward -= if pace { 12. } else { 10. };
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
    fn ground_launch_uses_throttle_without_a_separate_control_mode() {
        let mut race = Race::ground_start(0., 1.);
        for _ in 0..12 {
            race.step(mode2_acro([0., -1., 0., 0.])).unwrap();
        }
        assert_eq!(race.drone.position.z, RADIUS);
        for _ in 0..7 {
            race.step(mode2_acro([0., 0.4, 0., 0.])).unwrap();
        }
        assert!(race.drone.position.z > RADIUS + 0.2 && race.drone.velocity.z > 1.);
        assert!(!race.on_launch_pad);
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
        let a = gate.center - gate.normal;
        let b = gate.center + gate.normal;
        let course = Course::GrandChallenge2025;
        assert_eq!(gate.progress(a, b, course), (false, true));
        assert_eq!(gate.progress(b, a, course), (false, false));
        assert_eq!(
            gate.progress(a + W * gate.side, b + W * gate.side, course),
            (true, false)
        );
        assert_eq!(
            gate.progress(a + 3. * gate.side, b + 3. * gate.side, course),
            (false, false)
        );
        assert_eq!(gate.progress(a, gate.center, course), (false, true));
        assert_eq!(gate.progress(gate.center, b, course), (false, false));
    }

    #[test]
    fn season2_gate_contact_uses_the_thinner_frame() {
        let gate = Gate::all()[0];
        let beside_opening = gate.center + gate.side;
        let near = beside_opening - 0.46 * gate.normal;
        let far = beside_opening - 0.68 * gate.normal;
        assert_eq!(
            gate.progress(far, near, Course::GrandChallenge2025),
            (true, false)
        );
        assert_eq!(
            gate.progress(far, near, Course::Season2Practice2026),
            (false, false)
        );
        assert_eq!(
            gate.progress(near, beside_opening, Course::Season2Practice2026),
            (true, false)
        );
    }

    #[test]
    fn round_vehicle_distinguishes_corner_clearance_from_contact() {
        let (min, max) = BARS[3];
        let travel = Vec3::new(0., 0., 0.09);
        assert!(!frame_contact(Vec3::new(-0.55, 0.60, 0.), travel, min, max));
        assert!(frame_contact(Vec3::new(-0.53, 0.61, 0.), travel, min, max));
    }

    #[test]
    fn gate_dwell_cost_applies_to_aligned_stops_and_lateral_drift() {
        let gates = Gate::all();
        let gate = gates[1];
        let incoming = gate.center - gates[0].center;
        let outgoing = gates[2].center - gate.center;
        let offset = gate.normal * 2.;
        let delay = |speed| approach_delay(gate, incoming, outgoing, offset, speed);
        assert!(delay(Vec3::ZERO) > 0.);
        assert_eq!(delay(gate.side), delay(Vec3::ZERO));
        assert_eq!(delay(gate.normal * 2.), 0.);
        assert_eq!(
            approach_delay(gate, incoming, outgoing, -offset, Vec3::ZERO),
            0.
        );
        let split = gates[4];
        assert_eq!(
            approach_delay(
                split,
                split.center - gates[3].center,
                gates[5].center - split.center,
                split.normal * 2.,
                Vec3::ZERO
            ),
            0.
        );
    }

    #[test]
    fn pace_scores_shorter_segments_and_allows_time_for_reversal() {
        let gates = Gate::all();
        assert!(
            segment_pace_reward(gates[1], gates[2], 2.)
                > segment_pace_reward(gates[1], gates[2], 4.)
        );
        let reversed = gates[5];
        let straight = Gate {
            normal: gates[4].normal,
            ..reversed
        };
        assert!(
            segment_pace_reward(gates[4], reversed, 2.)
                > segment_pace_reward(gates[4], straight, 2.)
        );
        assert!(segment_pace_reward(gates[1], gates[2], 0.) < 1.5);
    }

    #[test]
    fn reward_profile_changes_time_cost_without_changing_dynamics() {
        let mut normal = Race::default();
        let mut pace = Race {
            reward_profile: RewardProfile::Pace,
            ..Race::default()
        };
        let action = [HOVER, 0., 0., 0.];
        let standard_reward = normal.step(action).unwrap();
        let pace_reward = pace.step(action).unwrap();
        assert_eq!(normal.drone.rigid_state(), pace.drone.rigid_state());
        assert!((standard_reward - pace_reward - 0.12 / CONTROL_HZ as f32).abs() < 1e-6);
    }

    #[test]
    fn forward_speed_bonus_requires_a_valid_crossing() {
        let run = |speed: f32, side: f32| {
            let mut race = Race::practice(2, 0., 0., 1.);
            let gate = race.gates[race.next_gate()];
            let travel = gate.normal * (speed * DT);
            race.drone.position = gate.center - travel * 0.5 + gate.side * side;
            race.progress = 1.;
            let mut first = true;
            let reward = race
                .step_with([HOVER, 0., 0., 0.], |drone, _| {
                    if first {
                        drone.position += travel;
                        first = false;
                    }
                    true
                })
                .unwrap();
            (reward, race.passed, race.status)
        };
        let (slow, _, _) = run(1., 0.);
        let (fast, passed, status) = run(10., 0.);
        assert_eq!((passed, status), (1, Status::Racing));
        assert!(fast > slow + 0.3);
        let (crash, passed, status) = run(10., W);
        assert_eq!((passed, status), (0, Status::GateCollision));
        assert!(crash < 0.);
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
    fn grand_challenge_times_gate_one_to_gate_twenty_two() {
        let mut race = Race::grand_challenge();
        let gates_per_lap = race.gates.len();
        race.ticks = 4 * PHYSICS_HZ;
        assert_eq!(race.official_seconds(), None);
        assert_eq!(race.lap_seconds(), 0.);
        race.on_launch_pad = false;
        for i in 0..gates_per_lap * GRAND_CHALLENGE_LAPS {
            let gate = race.gates[i % gates_per_lap];
            race.ticks += PHYSICS_HZ;
            race.drone.position = gate.center - gate.normal * 0.04;
            race.drone.velocity = gate.normal * 5.;
            race.step([HOVER, 0., 0., 0.]).unwrap();
            if i == 0 {
                assert!(race.first_gate_tick.unwrap() >= 5 * PHYSICS_HZ);
                assert!(race.official_seconds().unwrap() < 1. / CONTROL_HZ as f32);
            }
            if i == gates_per_lap - 1 {
                assert_eq!(race.status, Status::Racing);
            }
        }
        assert_eq!(race.status, Status::Finished);
        assert_eq!(race.passed, 22);
        assert_eq!(race.laps, 2);
        let official = race.official_seconds().unwrap();
        assert!((21. ..22.).contains(&official));
        assert_eq!(race.lap_seconds(), race.last_lap.unwrap());
        assert!(race.step([HOVER, 0., 0., 0.]).is_none());
        assert_eq!(race.official_seconds(), Some(official));
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
