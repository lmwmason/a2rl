// Smooth loft used for the removable upper and lower body shells.
// Each station is [longitudinal position, width, bottom Z, top Z].

function fair_slope(a, b) = a * b <= 0 ? 0 : 2 * a * b / (a + b);
function fair_value(stations, i, k, t) =
    let(
        a = stations[i], b = stations[i + 1],
        h = b.x - a.x, delta = (b[k] - a[k]) / h,
        prev = stations[max(0, i - 1)], next = stations[min(len(stations) - 1, i + 2)],
        m0 = (i == 0 ? delta : fair_slope((a[k] - prev[k]) / (a.x - prev.x), delta)) * h,
        m1 = (i == len(stations) - 2 ? delta :
              fair_slope(delta, (next[k] - b[k]) / (next.x - b.x))) * h
    )
    (2 * t^3 - 3 * t^2 + 1) * a[k] + (t^3 - 2 * t^2 + t) * m0
    + (-2 * t^3 + 3 * t^2) * b[k] + (t^3 - t^2) * m1;

function fair_profile(section, bottom_round, top_round) =
    let(w = section[1] / 2, b = section[2], t = section[3],
        top_r = min(top_round, w * 0.66, (t - b) * 0.45),
        bottom_r = min(bottom_round, w * 0.66, (t - b) * 0.45))
    concat(
        [for (angle = [-90 : 15 : 0])
            [w - bottom_r + bottom_r * cos(angle), b + bottom_r + bottom_r * sin(angle)]],
        [for (angle = [0 : 15 : 90])
            [w - top_r + top_r * cos(angle), t - top_r + top_r * sin(angle)]],
        [for (angle = [90 : 15 : 180])
            [-w + top_r + top_r * cos(angle), t - top_r + top_r * sin(angle)]],
        [for (angle = [180 : 15 : 270])
            [-w + bottom_r + bottom_r * cos(angle), b + bottom_r + bottom_r * sin(angle)]]
    );

module fair_loft(stations, bottom_round, top_round) {
    samples = [
        for (i = [0 : len(stations) - 2])
            let(steps = max(1, ceil((stations[i + 1].x - stations[i].x) / 4)))
            for (j = [0 : steps - 1])
                let(t = j / steps)
                [stations[i].x + t * (stations[i + 1].x - stations[i].x),
                 for (k = [1 : 3]) fair_value(stations, i, k, t)],
        stations[len(stations) - 1]
    ];
    n = len(fair_profile(samples[0], bottom_round, top_round));
    last = (len(samples) - 1) * n;
    vertices = [for (s = samples, p = fair_profile(s, bottom_round, top_round)) [s.x, p.x, p.y]];
    sides = [
        for (i = [0 : len(samples) - 2], j = [0 : n - 1])
            let(a = i * n + j, b = i * n + (j + 1) % n)
            each [[a, a + n, b + n], [a, b + n, b]]
    ];
    faces = concat([[for (j = [0 : n - 1]) j]], sides,
                   [[for (j = [n - 1 : -1 : 0]) last + j]]);
    polyhedron(vertices, faces, convexity = 12);
}

function fair_cavity(stations, wall, open_top, lip) = [
    for (i = [0 : len(stations) - 1])
        let(s = stations[i], end = i == 0 ? wall : i == len(stations) - 1 ? -wall : 0)
        [s.x + end, s[1] - 2 * wall, s[2] + (open_top ? wall : -lip),
         s[3] + (open_top ? lip : -wall)]
];
