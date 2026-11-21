//! Waypoint demonstration for the 2025 course. Independent of the vision policy.
use std::{
    env, fs,
    io::{self, Write},
    path::Path,
    process::{Command, Stdio},
};

use a2rl_sim::{
    camera::{camera_pose, hall_bounds, Camera, LENS},
    mode2_acro, Race, Status, BARS, CONTROL_HZ, PHYSICS_HZ, START,
};
use glam::{Mat3, Quat, Vec3};

const WIDTH: usize = 820;
const HEIGHT: usize = 616;
const PLAYBACK_SPEED: usize = 2;
const GRAVITY: f32 = 9.80665;

#[derive(Clone, Copy)]
struct Waypoint {
    position: Vec3,
    heading: Vec3,
    clearance: f32,
}

fn course(race: &Race) -> Vec<Waypoint> {
    let mut route = vec![Waypoint {
        position: START,
        heading: race.drone.rotation * Vec3::X,
        clearance: 0.2,
    }];
    for gate in race.gates {
        route.push(Waypoint {
            position: gate.center - gate.normal * 2.6,
            heading: gate.normal,
            clearance: 0.27,
        });
        route.push(Waypoint {
            position: gate.center + gate.normal * 2.0,
            heading: gate.normal,
            clearance: 0.36,
        });
    }
    route
}

fn sticks(race: &Race, waypoint: Waypoint) -> [f32; 4] {
    let drone = &race.drone;
    let error = waypoint.position - drone.position;
    let desired_velocity = (error * 1.5).clamp_length_max(6.0);
    let mut accel = (desired_velocity - drone.velocity) * 2.6;
    accel = Vec3::new(accel.x, accel.y, 0.).clamp_length_max(5.2)
        + Vec3::Z * (GRAVITY + accel.z.clamp(-5.5, 6.0));
    let up = accel.normalize();
    let left = up.cross(waypoint.heading).normalize();
    let desired = Quat::from_mat3(&Mat3::from_cols(left.cross(up), left, up));
    let rotation_error = drone.rotation.conjugate() * desired;
    let rotation_error = if rotation_error.w < 0. {
        -rotation_error
    } else {
        rotation_error
    };
    let rates = rotation_error.to_scaled_axis() * 6.0 / Vec3::new(5.5, 5.5, 3.4);
    let throttle = (accel.dot(drone.rotation * Vec3::Z) / GRAVITY * a2rl_sim::HOVER).clamp(0., 1.);
    [-rates.z, 2. * throttle - 1., rates.x, rates.y].map(|axis| axis.clamp(-1., 1.))
}

fn write_pose(out: &mut impl Write, race: &Race) -> io::Result<()> {
    for vector in camera_pose(race) {
        write!(out, "{} {} {} ", vector.x, vector.y, vector.z)?;
    }
    writeln!(out)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args_os().skip(1);
    let path = args.next();
    let trace = path.as_deref() == Some(std::ffi::OsStr::new("--poses"));
    let mut race = Race::ground_start(0., 1.);
    let mut poses = if trace {
        let mut output = fs::File::create(args.next().ok_or("Expected a pose output path")?)?;
        writeln!(
            output,
            "{WIDTH} {HEIGHT} {CONTROL_HZ} {}",
            race.gate_frames().count()
        )?;
        for value in LENS {
            write!(output, "{value} ")?;
        }
        writeln!(output)?;
        let mut vectors = hall_bounds().to_vec();
        for (min, max) in BARS {
            vectors.extend([min, max]);
        }
        for gate in race.gate_frames() {
            vectors.extend([gate.center, gate.normal, gate.side]);
        }
        for vector in vectors {
            writeln!(output, "{} {} {}", vector.x, vector.y, vector.z)?;
        }
        let mut output = io::BufWriter::new(output);
        for _ in 0..(3 * CONTROL_HZ) {
            write_pose(&mut output, &race)?;
        }
        Some(output)
    } else {
        None
    };
    let camera = Camera::new(WIDTH, HEIGHT);
    let mut rgb = vec![0; camera.byte_len()];
    let mut encoder = if let Some(ref path) = path.filter(|_| !trace) {
        let path = Path::new(path);
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            fs::create_dir_all(parent)?;
        }
        Some(
            Command::new("ffmpeg")
                .args([
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-n",
                    "-f",
                    "rawvideo",
                    "-pixel_format",
                    "rgb24",
                    "-video_size",
                ])
                .arg(format!("{WIDTH}x{HEIGHT}"))
                .arg("-framerate")
                .arg(CONTROL_HZ.to_string())
                .args([
                    "-i",
                    "pipe:0",
                    "-an",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "fast",
                    "-crf",
                    "19",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                ])
                .arg(path)
                .stdin(Stdio::piped())
                .spawn()?,
        )
    } else {
        None
    };
    let route = course(&race);
    let mut point = 0;
    let mut peak_speed = 0_f32;
    let mut result: io::Result<()> = Ok(());
    let mut ticks = 0_usize;
    while race.status == Status::Racing && point < route.len() {
        let waypoint = route[point];
        let previous = race.passed;
        race.step(mode2_acro(sticks(&race, waypoint)));
        let position = race.drone.position;
        peak_speed = peak_speed.max(race.drone.velocity.length());
        if race.passed != previous {
            eprintln!(
                "Gate {:02}/{} at {:.2}s",
                race.passed,
                race.gates.len(),
                race.ticks as f32 / PHYSICS_HZ as f32
            );
        }
        ticks += 1;
        if ticks.is_multiple_of(PLAYBACK_SPEED) || race.status != Status::Racing {
            if let Some(output) = &mut poses {
                write_pose(output, &race)?;
            }
            if let Some(child) = &mut encoder {
                camera.render(&race, &mut rgb);
                result = child.stdin.as_mut().unwrap().write_all(&rgb);
                if result.is_err() {
                    break;
                }
            }
        }
        if position.distance(waypoint.position) < waypoint.clearance {
            point += 1;
        }
    }
    if let Some(mut output) = poses {
        output.flush()?;
    }
    if let Some(mut child) = encoder {
        drop(child.stdin.take());
        if !child.wait()?.success() {
            return Err(io::Error::other("FFmpeg encoding failed").into());
        }
    }
    result?;
    println!(
        "{} / {} gates, {} laps, {:.2}s elapsed, peak {:.1} km/h, {:?}",
        race.passed,
        race.gates.len(),
        race.laps,
        race.ticks as f32 / PHYSICS_HZ as f32,
        peak_speed * 3.6,
        race.status
    );
    if race.laps != 1 || race.status != Status::Racing {
        eprintln!(
            "Waypoint {point}, position {:?}, velocity {:?}",
            race.drone.position, race.drone.velocity
        );
        return Err(io::Error::other(
            "The Grand Challenge demonstration did not complete one lap.",
        )
        .into());
    }
    Ok(())
}
