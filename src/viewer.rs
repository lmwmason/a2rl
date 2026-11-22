use std::{
    io::{self, Read},
    process::{Child, Command, Stdio},
    sync::{Arc, Mutex},
    thread,
};

use eframe::egui::{self, Align2, Color32, FontId, Pos2, Rect, Stroke, Vec2};
use serde_json::{json, Value};

struct Frame {
    info: Value,
    pixels: Vec<u8>,
}

fn read_frame(input: &mut impl Read) -> io::Result<Frame> {
    let mut header = [0; 8];
    input.read_exact(&mut header)?;
    let info_size = u32::from_le_bytes(header[..4].try_into().unwrap()) as usize;
    let image_size = u32::from_le_bytes(header[4..].try_into().unwrap()) as usize;
    if info_size > 128_000 || image_size > 3_000_000 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Invalid frame size",
        ));
    }
    let mut info = vec![0; info_size];
    let mut pixels = vec![0; image_size];
    input.read_exact(&mut info)?;
    input.read_exact(&mut pixels)?;
    Ok(Frame {
        info: serde_json::from_slice(&info).map_err(io::Error::other)?,
        pixels,
    })
}

struct Viewer {
    child: Option<Child>,
    pending: Arc<Mutex<Option<Frame>>>,
    info: Value,
    texture: Option<egui::TextureHandle>,
}

impl Viewer {
    fn new(ctx: &egui::Context, host: std::ffi::OsString) -> Self {
        let pending = Arc::new(Mutex::new(None));
        let mut viewer = Self {
            child: None,
            pending: pending.clone(),
            info: json!({"connection": "Opening SSH connection", "view": "Waiting for frames"}),
            texture: None,
        };
        match Command::new("python3")
            .args(["-u", "-m", "training.watch"])
            .arg(host)
            .stdout(Stdio::piped())
            .spawn()
        {
            Ok(mut child) => {
                let mut output = io::BufReader::new(child.stdout.take().unwrap());
                let ctx = ctx.clone();
                thread::spawn(move || loop {
                    match read_frame(&mut output) {
                        Ok(frame) => *pending.lock().unwrap() = Some(frame),
                        Err(error) => {
                            *pending.lock().unwrap() = Some(Frame {
                                info: json!({"connection": error.to_string(), "view": "Stream ended"}),
                                pixels: Vec::new(),
                            });
                            ctx.request_repaint();
                            break;
                        }
                    }
                    ctx.request_repaint();
                });
                viewer.child = Some(child);
            }
            Err(error) => {
                viewer.info = json!({"connection": error.to_string(), "view": "Viewer unavailable"})
            }
        }
        viewer
    }

    fn receive(&mut self, ctx: &egui::Context) {
        let frame = self.pending.lock().unwrap().take();
        if let Some(frame) = frame {
            let width = number(&frame.info, "width") as usize;
            let height = number(&frame.info, "height") as usize;
            if (1..=1024).contains(&width)
                && (1..=768).contains(&height)
                && frame.pixels.len() == width * height * 3
            {
                let image = egui::ColorImage::from_rgb([width, height], &frame.pixels);
                if let Some(texture) = &mut self.texture {
                    texture.set(image, egui::TextureOptions::LINEAR);
                } else {
                    self.texture =
                        Some(ctx.load_texture("training-fpv", image, egui::TextureOptions::LINEAR));
                }
            }
            self.info = frame.info;
        }
    }
}

impl Drop for Viewer {
    fn drop(&mut self) {
        if let Some(child) = &mut self.child {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

fn number(data: &Value, key: &str) -> u64 {
    data[key].as_u64().unwrap_or_default()
}
fn rate(data: &Value, key: &str) -> f64 {
    data[key].as_f64().unwrap_or_default()
}

impl eframe::App for Viewer {
    fn update(&mut self, ctx: &egui::Context, _: &mut eframe::Frame) {
        self.receive(ctx);
        let info = &self.info;
        let view = info["view"].as_str().unwrap_or("Connecting");
        let accent = Color32::from_rgb(116, 228, 189);
        egui::TopBottomPanel::top("training-header").show(ctx, |ui| {
            ui.add_space(5.);
            ui.horizontal(|ui| {
                ui.heading("A2RL FPV  /  TRAINING");
                ui.separator();
                ui.colored_label(accent, info["phase"].as_str().unwrap_or("CONNECTING"));
                ui.label(if info["active"].as_bool().unwrap_or(false) {
                    "Running"
                } else {
                    "Stopped"
                });
            });
            ui.add_space(5.);
        });
        egui::SidePanel::right("training-metrics")
            .exact_width(222.)
            .resizable(false)
            .show(ctx, |ui| {
                ui.add_space(10.);
                ui.heading("Evaluation");
                ui.add_space(6.);
                ui.monospace(format!("Transitions    {:>9}", number(info, "steps")));
                if number(info, "round") != 0 {
                    ui.monospace(format!("Round          {:>9}", number(info, "round")));
                }
                if number(info, "target_gate") != 0 {
                    ui.monospace(format!(
                        "Curriculum gate {:>8}",
                        number(info, "target_gate")
                    ));
                }
                ui.monospace(format!("Best mean gates {:>6.1}", rate(info, "best_gates")));
                ui.monospace(format!(
                    "2-lap finish {:>10.0}%",
                    100. * rate(info, "finish_rate")
                ));
                ui.monospace(format!(
                    "Collision    {:>10.0}%",
                    100. * rate(info, "collision_rate")
                ));
                ui.add_space(18.);
                ui.weak("Mean gates / recent evaluations");
                let (rect, _) = ui.allocate_exact_size(Vec2::new(200., 106.), egui::Sense::hover());
                let painter = ui.painter_at(rect);
                painter.rect_filled(rect, 5., Color32::from_rgb(18, 26, 34));
                if let Some(series) = info["recent_gates"].as_array() {
                    let peak = (rate(info, "best_gates") as f32).max(2.);
                    let width = rect.width() / series.len().saturating_sub(1).max(1) as f32;
                    let points: Vec<_> = series
                        .iter()
                        .enumerate()
                        .map(|(i, value)| {
                            Pos2::new(
                                rect.left() + i as f32 * width,
                                rect.bottom()
                                    - value.as_f64().unwrap_or_default() as f32 / peak
                                        * (rect.height() - 8.),
                            )
                        })
                        .collect();
                    if points.len() > 1 {
                        painter.add(egui::Shape::line(points, Stroke::new(2_f32, accent)));
                    }
                }
                ui.add_space(10.);
                ui.weak(info["connection"].as_str().unwrap_or_default());
            });
        egui::CentralPanel::default()
            .frame(egui::Frame::new().fill(Color32::BLACK))
            .show(ctx, |ui| {
                let area = ui.max_rect();
                let painter = ui.painter_at(area);
                if let Some(texture) = &self.texture {
                    let scale = (area.width() / texture.size_vec2().x)
                        .min(area.height() / texture.size_vec2().y);
                    let image = Rect::from_center_size(area.center(), scale * texture.size_vec2());
                    painter.image(
                        texture.id(),
                        image,
                        Rect::from_min_max(Pos2::ZERO, Pos2::new(1., 1.)),
                        Color32::WHITE,
                    );
                    let top = image.left_top() + Vec2::new(12., 12.);
                    let show = |pos, align, label: &str| {
                        painter.text(
                            pos + Vec2::splat(2.),
                            align,
                            label,
                            FontId::monospace(14.),
                            Color32::BLACK,
                        );
                        painter.text(pos, align, label, FontId::monospace(14.), accent);
                    };
                    show(top, Align2::LEFT_TOP, view);
                    if view.starts_with("LIVE") {
                        show(
                            top + Vec2::new(0., 20.),
                            Align2::LEFT_TOP,
                            info["driver"].as_str().unwrap_or("Policy"),
                        );
                    }
                    let target = match number(info, "status") {
                        1 => format!("COLLISION  /  {} GATES", number(info, "gates")),
                        3 => "GATE TIME EXPIRED".to_owned(),
                        4 => "TWO LAPS COMPLETE".to_owned(),
                        _ => format!("{} GATES CLEARED", number(info, "gates")),
                    };
                    show(
                        image.right_top() + Vec2::new(-12., 12.),
                        Align2::RIGHT_TOP,
                        &target,
                    );
                } else {
                    painter.text(
                        area.center(),
                        Align2::CENTER_CENTER,
                        view,
                        FontId::monospace(17.),
                        accent,
                    );
                }
            });
    }
}

pub fn launch(host: std::ffi::OsString) -> eframe::Result {
    eframe::run_native(
        "A2RL FPV Training Viewer",
        eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default()
                .with_inner_size([1090., 740.])
                .with_min_inner_size([670., 440.]),
            ..Default::default()
        },
        Box::new(move |creation| Ok(Box::new(Viewer::new(&creation.egui_ctx, host)))),
    )
}
