use std::{
    net::SocketAddr,
    sync::{Arc, Mutex},
};

use a2rl_sim::{camera::Camera, Race, Status, HOVER, LAYOUT, MASS_KG, PHYSICS_HZ};
use axum::{
    extract::State,
    http::{header, HeaderValue, StatusCode},
    response::{Html, IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use glam::EulerRot;
use serde_json::json;

const WIDTH: usize = 640;
const HEIGHT: usize = 480;

#[derive(Clone)]
struct App {
    race: Arc<Mutex<Race>>,
    camera: Arc<Camera>,
}

fn frame(race: &Race, camera: &Camera) -> Response {
    let mut rgb = vec![0; camera.byte_len()];
    camera.render(race, &mut rgb);
    let elapsed = (race.ticks - race.lap_start) as f32 / PHYSICS_HZ as f32;
    let (yaw, _, _) = race.drone.rotation.to_euler(EulerRot::ZYX);
    let stats = json!({
        "status": race.status as u8, "gate": race.next_gate(), "laps": race.laps,
        "time": elapsed, "last": race.last_lap, "best": race.best_lap,
        "speed": race.drone.velocity.length() * 3.6, "height": race.drone.position.z,
        "position": race.drone.position.to_array(), "yaw": yaw
    });
    let mut response = rgb.into_response();
    response
        .headers_mut()
        .insert("x-race", HeaderValue::from_str(&stats.to_string()).unwrap());
    response
        .headers_mut()
        .insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
    response
}

async fn view(State(app): State<App>) -> Response {
    frame(&app.race.lock().unwrap(), &app.camera)
}

async fn reset(State(app): State<App>, Json(()): Json<()>) -> Response {
    let mut race = app.race.lock().unwrap();
    *race = Race::default();
    frame(&race, &app.camera)
}

async fn fly(State(app): State<App>, Json(action): Json<[f32; 4]>) -> Response {
    if action.iter().any(|axis| !axis.is_finite()) {
        return StatusCode::BAD_REQUEST.into_response();
    }
    let mut race = app.race.lock().unwrap();
    if race.status == Status::Racing {
        // Angle assistance applies only to the human keyboard controls.
        let tilt = 0.46;
        let (_, pitch, roll) = race.drone.rotation.to_euler(EulerRot::ZYX);
        let thrust = HOVER / (roll.cos() * pitch.cos()).max(0.5);
        let throttle = (thrust + 0.17 * action[0] - 0.035 * race.drone.velocity.z).clamp(0., 1.);
        race.step([
            throttle,
            (tilt * action[1] - roll) * 1.2,
            (tilt * action[2] - pitch) * 1.2,
            action[3] * 0.48,
        ]);
    }
    frame(&race, &app.camera)
}

fn router() -> Router {
    let app = App {
        race: Arc::new(Mutex::new(Race::default())),
        camera: Arc::new(Camera::new(WIDTH, HEIGHT)),
    };
    Router::new()
        .route(
            "/",
            get(|| async { Html(include_str!("../web/index.html")) }),
        )
        .route(
            "/app.js",
            get(|| async {
                (
                    [(header::CONTENT_TYPE, "text/javascript")],
                    include_str!("../web/app.js"),
                )
            }),
        )
        .route(
            "/config",
            get(|| async {
                Json(json!({
                    "width": WIDTH, "height": HEIGHT, "gates": LAYOUT, "mass": MASS_KG
                }))
            }),
        )
        .route("/frame", get(view))
        .route("/fly", post(fly))
        .route("/reset", post(reset))
        .with_state(app)
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let port = std::env::args()
        .nth(1)
        .map_or(Ok(8765), |value| value.parse::<u16>())?;
    let listener = tokio::net::TcpListener::bind(SocketAddr::from(([127, 0, 0, 1], port))).await?;
    println!("A2RL FPV  http://{}", listener.local_addr()?);
    axum::serve(listener, router()).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        body::{to_bytes, Body},
        http::Request,
    };
    use tower::ServiceExt;

    #[tokio::test]
    async fn gui_routes_deliver_live_rgb_and_reset() {
        let app = router();
        let mut stats = Vec::new();
        for (path, body) in [
            ("/frame", None),
            ("/fly", Some("[0,0,1,0]")),
            ("/reset", Some("null")),
        ] {
            let response = app
                .clone()
                .oneshot(
                    Request::builder()
                        .uri(path)
                        .method(if body.is_some() { "POST" } else { "GET" })
                        .header(header::CONTENT_TYPE, "application/json")
                        .body(Body::from(body.unwrap_or_default()))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            let race: serde_json::Value =
                serde_json::from_str(response.headers()["x-race"].to_str().unwrap()).unwrap();
            stats.push(race["time"].as_f64().unwrap());
            assert_eq!(
                to_bytes(response.into_body(), WIDTH * HEIGHT * 3)
                    .await
                    .unwrap()
                    .len(),
                WIDTH * HEIGHT * 3
            );
        }
        assert_eq!(stats[0], 0.);
        assert!(stats[1] > 0.);
        assert_eq!(stats[2], 0.);
    }
}
