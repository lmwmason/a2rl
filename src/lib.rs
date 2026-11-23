pub mod camera;
mod ffi;
pub mod teacher;

use glam::{Quat, Vec3};
use parry3d::{
    na::{Isometry3, Vector3},
    query::{cast_shapes, ShapeCastOptions},
    shape::{Ball, Cuboid},
};

const VIDEO_CLOCK_HZ: usize = 240;
const VIDEO_FRAME_HZ: usize = 30;

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
// Fit on the first airborne segment of MonoRace Log 8. Logs 5 and 7 are held out.
const LINEAR_DRAG: Vec3 = Vec3::new(0.3935197, 0.7070415, 0.);
const QUADRATIC_DRAG: Vec3 = Vec3::new(0.002257, 0., 0.);
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
}

impl Drone {
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

pub struct Race {
    pub drone: Drone,
    pub gates: Vec<Gate>,
    pub course: Course,
    pub ticks: usize,
    pub passed: usize,
    pub camera_tilt: f32,
    pub exposure: f32,
    pub video_breakup: f32,
    pub video_seed: u32,
    grand_challenge: bool,
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
        Self {
            drone: Drone {
                position,
                velocity: Vec3::ZERO,
                rotation: Quat::from_rotation_z(yaw.to_radians()),
                omega: Vec3::ZERO,
            },
            gates,
            course,
            ticks: 0,
            passed: 0,
            camera_tilt: camera::mount_angle(course),
            exposure: 1.,
            video_breakup: 0.,
            video_seed: 1,
            grand_challenge: false,
        }
    }
}

impl Race {
    pub fn grand_challenge_on(course: Course) -> Self {
        let mut race = Self::on_course(course);
        race.drone.position.z = RADIUS;
        race.grand_challenge = true;
        race
    }

    pub fn gate_frames(&self) -> impl Iterator<Item = &Gate> {
        self.gates.iter().chain(self.course.extra_frames())
    }

    pub fn next_gate(&self) -> usize {
        self.passed % self.gates.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
