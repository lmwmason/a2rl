use std::time::{Duration, Instant};

use a2rl_sim::{camera::Camera, mode2_acro, Race, Status, CONTROL_HZ, PHYSICS_HZ};
use eframe::egui::{self, Align2, Color32, FontId, Key, Pos2, Rect, Stroke, Vec2, ViewportCommand};

const WIDTH: usize = 640;
const HEIGHT: usize = 480;
const TICK: Duration = Duration::from_nanos(1_000_000_000 / CONTROL_HZ as u64);
const THROTTLE_PER_SECOND: f32 = 0.65;
const COUNTDOWN_TICKS: usize = 3 * CONTROL_HZ;
const HUD: Color32 = Color32::from_rgb(222, 252, 239);

#[derive(Default)]
struct Sticks {
    throttle: f32,
    roll: f32,
    pitch: f32,
    yaw: f32,
}

impl Sticks {
    fn read(input: &egui::InputState) -> Self {
        let axis = |positive, negative| {
            u8::from(input.key_down(positive)) as f32 - u8::from(input.key_down(negative)) as f32
        };
        Self {
            throttle: axis(Key::W, Key::S),
            roll: axis(Key::ArrowRight, Key::ArrowLeft),
            pitch: axis(Key::ArrowUp, Key::ArrowDown),
            yaw: axis(Key::A, Key::D),
        }
    }

    fn action(&self, throttle: &mut f32) -> [f32; 4] {
        *throttle =
            (*throttle + self.throttle * THROTTLE_PER_SECOND / CONTROL_HZ as f32).clamp(0., 1.);
        mode2_acro([-self.yaw, 2. * *throttle - 1., self.roll, self.pitch])
    }
}

struct Fpv {
    race: Race,
    camera: Camera,
    rgb: Vec<u8>,
    texture: egui::TextureHandle,
    throttle: f32,
    running: bool,
    countdown: usize,
    last_frame: Instant,
    pending: Duration,
}

impl Fpv {
    fn new(ctx: &egui::Context) -> Self {
        let race = Race::ground_start(0., 1.);
        let camera = Camera::new(WIDTH, HEIGHT);
        let mut rgb = vec![0; camera.byte_len()];
        camera.render(&race, &mut rgb);
        let texture = ctx.load_texture(
            "fpv",
            egui::ColorImage::from_rgb([WIDTH, HEIGHT], &rgb),
            egui::TextureOptions::LINEAR,
        );
        Self {
            race,
            camera,
            rgb,
            texture,
            throttle: 0.,
            running: false,
            countdown: 0,
            last_frame: Instant::now(),
            pending: Duration::ZERO,
        }
    }

    fn update_camera(&mut self) {
        self.camera.render(&self.race, &mut self.rgb);
        self.texture.set(
            egui::ColorImage::from_rgb([WIDTH, HEIGHT], &self.rgb),
            egui::TextureOptions::LINEAR,
        );
    }

    fn set_running(&mut self, running: bool) {
        self.running = running && self.race.status == Status::Racing;
        self.pending = Duration::ZERO;
        self.last_frame = Instant::now();
    }

    fn reset(&mut self) {
        self.set_running(false);
        let breakup = self.race.video_breakup;
        self.race = Race::ground_start(0., 1.);
        self.race.video_breakup = breakup;
        self.throttle = 0.;
        self.countdown = 0;
        self.update_camera();
    }

    fn advance(&mut self, sticks: &Sticks) {
        self.pending += self.last_frame.elapsed().min(Duration::from_millis(100));
        self.last_frame = Instant::now();
        let mut advanced = false;
        while self.pending >= TICK && self.race.status == Status::Racing {
            self.pending -= TICK;
            advanced |= self.advance_tick(sticks);
        }
        if advanced {
            self.update_camera();
        }
        if self.race.status != Status::Racing {
            self.set_running(false);
        }
    }

    fn advance_tick(&mut self, sticks: &Sticks) -> bool {
        let action = sticks.action(&mut self.throttle);
        self.countdown = self.countdown.saturating_sub(1);
        if self.countdown > 0 {
            return false;
        }
        self.race.step(action).is_some()
    }

    fn hud(&self, painter: &egui::Painter, frame: Rect) {
        let scale = (frame.width() / 790.).clamp(0.70, 1.45);
        let inset = frame.shrink(25. * scale);
        let font = FontId::monospace(17. * scale);
        let show = |position: Pos2, align: Align2, text: &str| {
            painter.text(
                position + Vec2::splat(1.4),
                align,
                text,
                font.clone(),
                Color32::BLACK,
            );
            painter.text(position, align, text, font.clone(), HUD);
        };
        let mode = if self.race.video_breakup > 0. {
            "ACRO\nHDZERO"
        } else {
            "ACRO"
        };
        show(inset.left_top(), Align2::LEFT_TOP, mode);
        show(
            inset.center_top(),
            Align2::CENTER_TOP,
            &format!(
                "GATE {:02}/{:02}",
                self.race.next_gate() + 1,
                self.race.gates.len()
            ),
        );
        let lap = (self.race.ticks - self.race.lap_start) as f32 / PHYSICS_HZ as f32;
        let total = self.race.ticks as f32 / PHYSICS_HZ as f32;
        let time = format!(
            "LAP {:02}  {:06.2}s\nTOTAL   {:06.2}s",
            self.race.laps + 1,
            lap,
            total
        );
        show(
            inset.right_top(),
            Align2::RIGHT_TOP,
            &self
                .race
                .best_lap
                .map_or(time.clone(), |best| format!("{time}\nPB {best:.2}s")),
        );
        show(
            inset.left_bottom(),
            Align2::LEFT_BOTTOM,
            &format!("{:3.0}% THR", self.throttle * 100.),
        );
        show(
            inset.right_bottom(),
            Align2::RIGHT_BOTTOM,
            &format!(
                "{:4.1} km/h\n{:4.1} m",
                self.race.drone.velocity.length() * 3.6,
                self.race.drone.position.z,
            ),
        );

        let center = frame.center();
        let stroke = Stroke::new(1_f32, HUD.gamma_multiply(0.65));
        for dir in [-Vec2::X, Vec2::X, Vec2::Y] {
            painter.line_segment(
                [center + dir * 5. * scale, center + dir * 10. * scale],
                stroke,
            );
        }
        if self.running && (self.countdown > 0 || self.race.ticks < PHYSICS_HZ / 2) {
            let (text, color) = if self.countdown > 0 {
                (
                    self.countdown.div_ceil(CONTROL_HZ).to_string(),
                    Color32::from_rgb(255, 221, 132),
                )
            } else {
                ("GO".to_owned(), HUD)
            };
            painter.text(
                center + Vec2::new(2., -55. * scale + 2.),
                Align2::CENTER_CENTER,
                &text,
                FontId::monospace(62. * scale),
                Color32::BLACK,
            );
            painter.text(
                center - Vec2::Y * 55. * scale,
                Align2::CENTER_CENTER,
                &text,
                FontId::monospace(62. * scale),
                color,
            );
        }
    }

    fn flight_menu(&mut self, ctx: &egui::Context) {
        let (title, action) = match self.race.status {
            Status::Racing if self.race.ticks > 0 || self.countdown > 0 => {
                ("PAUSED", "Resume flight")
            }
            Status::Racing => ("MODE 2 ACRO", "Start flight"),
            Status::GateCollision => ("GATE COLLISION", "Restart flight"),
            Status::Boundary => ("COURSE BOUNDARY", "Restart flight"),
            Status::Timeout => ("SESSION COMPLETE", "Restart flight"),
            Status::Stalled => ("GATE TIME EXPIRED", "Restart flight"),
        };
        egui::Area::new(egui::Id::new("flight_menu"))
            .anchor(Align2::CENTER_CENTER, Vec2::ZERO)
            .movable(false)
            .show(ctx, |ui| {
                egui::Frame::new()
                    .fill(Color32::from_rgba_unmultiplied(8, 18, 28, 235))
                    .stroke(Stroke::new(1_f32, Color32::from_rgb(71, 96, 107)))
                    .corner_radius(9)
                    .inner_margin(22)
                    .show(ui, |ui| {
                        ui.vertical_centered(|ui| {
                            ui.label(egui::RichText::new(title).monospace().size(24.).color(HUD));
                            ui.add_space(12.);
                            if ui
                                .add_sized([170., 36.], egui::Button::new(action))
                                .clicked()
                            {
                                self.toggle_flight();
                                ctx.request_repaint();
                            }
                            ui.add_space(11.);
                            ui.monospace("W/S Throttle    A/D Yaw");
                            ui.monospace("↑/↓ Pitch       ←/→ Roll");
                            ui.weak("Set launch throttle during the countdown.");
                            ui.add_space(6.);
                            ui.weak("Space  Pause    R  Reset    F11  Fullscreen");
                            ui.weak("N  HDZero breakup");
                        });
                    });
            });
    }

    fn toggle_flight(&mut self) {
        if self.race.status != Status::Racing {
            self.reset();
        }
        if !self.running && self.race.ticks == 0 && self.countdown == 0 {
            self.countdown = COUNTDOWN_TICKS;
        }
        self.set_running(!self.running);
    }
}

impl eframe::App for Fpv {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let pressed = |wanted| {
            ctx.input(|i| i.events.iter().any(|event|
            matches!(event, egui::Event::Key { key, pressed: true, repeat: false, .. } if *key == wanted)))
        };
        if pressed(Key::F11) || pressed(Key::Escape) {
            let fullscreen = ctx.input(|i| i.viewport().fullscreen.unwrap_or(false));
            let requested = pressed(Key::F11) && !fullscreen;
            ctx.send_viewport_cmd(ViewportCommand::Fullscreen(requested));
        }
        if !ctx.input(|i| i.focused) && self.running {
            self.set_running(false);
        }
        if pressed(Key::R) {
            self.reset();
        } else if pressed(Key::Space) {
            self.toggle_flight();
        }
        if pressed(Key::N) {
            self.race.video_breakup = if self.race.video_breakup > 0. {
                0.
            } else {
                0.36
            };
            self.update_camera();
        }
        if self.running {
            self.advance(&ctx.input(Sticks::read));
            ctx.request_repaint_after(
                TICK.saturating_sub(self.pending + self.last_frame.elapsed()),
            );
        }

        egui::CentralPanel::default()
            .frame(egui::Frame::new().fill(Color32::BLACK))
            .show(ctx, |ui| {
                let available = ui.max_rect();
                let scale =
                    (available.width() / WIDTH as f32).min(available.height() / HEIGHT as f32);
                let size = scale * Vec2::new(WIDTH as f32, HEIGHT as f32);
                let frame = Rect::from_center_size(available.center(), size);
                let painter = ui.painter_at(frame);
                painter.image(
                    self.texture.id(),
                    frame,
                    Rect::from_min_max(Pos2::ZERO, Pos2::new(1., 1.)),
                    Color32::WHITE,
                );
                self.hud(&painter, frame);
            });
        if !self.running {
            self.flight_menu(ctx);
        }
    }
}

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1024., 768.])
            .with_min_inner_size([490., 360.]),
        ..Default::default()
    };
    eframe::run_native(
        "A2RL FPV Simulator",
        options,
        Box::new(|cc| Ok(Box::new(Fpv::new(&cc.egui_ctx)))),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use a2rl_sim::HOVER;

    #[test]
    fn countdown_preloads_sticks_and_releases_on_go() {
        let mut app = Fpv::new(&egui::Context::default());
        app.toggle_flight();
        let sticks = Sticks {
            throttle: 1.,
            ..Sticks::default()
        };
        for _ in 1..COUNTDOWN_TICKS {
            assert!(!app.advance_tick(&sticks));
        }
        assert_eq!(app.race.ticks, 0);
        assert_eq!(app.throttle, 1.);
        assert!(app.advance_tick(&Sticks::default()));
        assert_eq!(app.race.ticks, PHYSICS_HZ / CONTROL_HZ);
        assert_eq!(app.countdown, 0);
    }

    #[test]
    fn acro_keeps_throttle_after_release_and_holds_bank() {
        let mut throttle = HOVER;
        let sticks = Sticks {
            throttle: 1.,
            roll: -1.,
            pitch: 1.,
            yaw: -1.,
        };
        assert_eq!(sticks.action(&mut throttle)[1..], [-1., 1., -1.]);
        let held = throttle;
        assert!((Sticks::default().action(&mut throttle)[0] - held).abs() < 1e-6);

        let mut race = Race::default();
        let bank = race.drone.rotation * glam::Quat::from_rotation_x(0.45);
        race.drone.rotation = bank;
        for _ in 0..10 {
            race.step([HOVER, 0., 0., 0.]);
        }
        assert!(race.drone.rotation.abs_diff_eq(bank, 0.001));
        assert!(race.drone.velocity.z < -0.25);
    }
}
