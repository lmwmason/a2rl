// testFPV retrofit. Millimetres. +X is forward, +Z is up.
// The frame and unidentified parts are provisional reference geometry.
// Change the four motor centres to the measured existing frame before making parts.
use <battery.scad>
use <power-module.scad>
use <jetson-b01.scad>

/* [Assembly] */
view = "assembled"; // [assembled,exploded,clearance,stack-adapter]
stack_pitch = 30.5; // [20,30.5]
wheelbase = 220;
motor_x = wheelbase / sqrt(8);
motor_y = wheelbase / sqrt(8);
motor_centres = [[motor_x, motor_y], [motor_x, -motor_y],
                 [-motor_x, motor_y], [-motor_x, -motor_y]];
camera_tilt = 55;
show_top_plate = true;
show_propellers = true;

/* [Hidden] */
$fn = 32;
arm_thickness = 5;
prop_radius = 63.5;
prop_z = 32;
sweep_half_height = 4;
top_z = 25;
tray_z = 42;
nano_z = 49;
battery_z = -33;
regulator_origin = [-52, 0, 8];
camera_origin = [65, 0, 20];
board_mounts = [[-32, -17], [-32, 17], [32, -17], [32, 17]];
adapter_mounts = [[0, -17.5], [0, 17.5]];
spread = view == "exploded" ? 1 : 0;

module plate(length, width, thickness, radius) {
    hull() for (x = [-1, 1], y = [-1, 1])
        translate([x * (length / 2 - radius), y * (width / 2 - radius), 0])
            cylinder(r = radius, h = thickness);
}

module hole_at(point, diameter, z, height) {
    translate([point.x, point.y, z]) cylinder(d = diameter, h = height);
}

module screw(point, z, shaft, diameter = 3) {
    color("#52565d") translate([point.x, point.y, z]) {
        cylinder(d = diameter, h = shaft);
        translate([0, 0, shaft]) difference() {
            cylinder(d = diameter + 2.35, h = 1.7);
            translate([0, 0, 0.8]) cylinder(d = diameter * 0.59, h = 1, $fn = 6);
        }
    }
}

module standoff(point, z, height, outer, bore) {
    translate([point.x, point.y, z]) difference() {
        cylinder(d = outer, h = height, $fn = 6);
        translate([0, 0, -0.1]) cylinder(d = bore, h = height + 0.2);
    }
}

module arm_root(p) {
    translate([p.x < 0 ? -21 : 21, p.y < 0 ? -14 : 14, 0])
        cylinder(d = 13, h = arm_thickness);
}

module carbon_frame() {
    color("#22252d") difference() {
        union() {
            translate([-2, 0, 0]) plate(158, 46, 3, 8);
            for (p = motor_centres) hull() {
                arm_root(p);
                translate([p.x, p.y, 0]) cylinder(d = 31, h = arm_thickness);
            }
        }
        for (p = motor_centres) {
            hole_at(p, 8, -0.1, 6);
            for (dx = [-8, 8], dy = [-8, 8])
                hole_at([p.x + dx, p.y + dy], 3.2, -0.1, 6);
        }
        for (p = board_mounts) hole_at(p, 3.2, -0.1, 6);
        for (p = adapter_mounts) hole_at(p, 3.2, -0.1, 6);
        for (x = [-20, 17], y = [-20.6, 20.6])
            translate([x - 5.5, y - 0.95, -0.1]) cube([11, 1.9, 5.2]);
    }
    color("#8b9199") for (p = board_mounts)
        standoff(p, 3, top_z - 3, 6.4, 3.1);
    if (show_top_plate)
        translate([-2, 0, top_z + 30 * spread]) color("#272932") difference() {
            plate(156, 46, 2.4, 8);
            for (p = board_mounts) hole_at([p.x + 2, p.y], 3.3, -0.1, 3);
            for (x = [-57, -45, -13, 0, 13, 51, 64])
                translate([x - 2, -9, -0.1]) plate(4, 9, 3, 1.5);
            translate([52, -16, -0.1]) cube([30, 32, 2.6]);
            translate([-79, -19, -0.1]) cube([42, 38, 2.6]);
        }
}

module quad_motor(p) {
    translate([p.x, p.y, arm_thickness + 8 * spread]) {
        color("#4b4f58") cylinder(d = 27.5, h = 3);
        color("#b77838") for (i = [0 : 11]) rotate([0, 0, i * 30])
            translate([8.7, 0, 3]) cylinder(d = 4.8, h = 5);
        color("#282d36") difference() {
            union() {
                translate([0, 0, 5]) cylinder(d = 27.8, h = 12);
                translate([0, 0, 17]) cylinder(d1 = 27.8, d2 = 23, h = 2);
            }
            for (i = [0 : 5]) rotate([0, 0, i * 60])
                translate([8.9, 0, 16.7]) cylinder(d = 4.55, h = 3);
        }
        color("silver") translate([0, 0, 18.5]) cylinder(d = 5, h = prop_z - arm_thickness - 16);
    }
}

module propeller(index) {
    p = motor_centres[index];
    hand = index == 0 || index == 3 ? 1 : -1;
    translate([p.x, p.y, prop_z + 15 * spread]) rotate([0, 0, 20 + 17 * index]) {
        color(index < 2 ? "#da5d30" : "#ad4b28") scale([1, hand, 1]) {
            cylinder(d = 13, h = 4, center = true);
            for (angle = [0, 120, 240]) rotate([0, 0, angle])
                hull() for (section = [[5, 0, 2, 3], [28, 2.5, 5.8, 2],
                                       [52, 3.1, 4.7, 1.6], [61.3, 1.2, 2.1, 0.8]])
                    translate([section[0], section[1], 0])
                        scale([section[2], section[2], section[3] / 2]) sphere(r = 1, $fn = 12);
        }
        color("#bbbfc7") translate([0, 0, 2]) cylinder(d = 7.8, h = 3, $fn = 6);
    }
}

module stack_adapter() {
    // Both hole patterns are present. The carrier fits the measured frame separately.
    difference() {
        plate(45, 43, 2.5, 4);
        for (pitch = [20, 30.5], x = [-pitch / 2, pitch / 2], y = [-pitch / 2, pitch / 2])
            hole_at([x, y], pitch == 20 ? 2.2 : 3.2, -0.1, 2.7);
        for (p = adapter_mounts) hole_at(p, 3.2, -0.1, 2.7);
        hole_at([0, 0], 12, -0.1, 2.7);
    }
}

module electronics_board(width, z, fc) {
    translate([0, 0, z]) {
        color(fc ? "#28425a" : "#223a36") difference() {
            plate(width, width, 1.6, 2.7);
            for (x = [-stack_pitch / 2, stack_pitch / 2], y = [-stack_pitch / 2, stack_pitch / 2])
                hole_at([x, y], stack_pitch == 20 ? 2.2 : 3.2, -0.1, 2);
        }
        color("#daa962") for (x = [-width / 2 + 1.3, width / 2 - 1.3], y = [-7, -3.5, 0, 3.5, 7])
            translate([x - 0.85, y - 0.85, 1.56]) cube([1.7, 1.7, 0.1]);
        color("#23242a") for (x = fc ? [0] : [-8, 0, 8], y = fc ? [2] : [-6, 6])
            translate([x - 3, y - 3, 1.58]) cube([6, 6, 1.55]);
        color("#d3d1c4") translate([-4.2, -width / 2 + 2, 1.57]) cube([8.4, 3.6, 2.8]);
        if (fc) color("#abb1ba") translate([-width / 2, -3.4, 1.57])
            cube([6.2, 6.8, 2.7]);
    }
}

// F43030 mounting pitch is published. Outline, grommets and component sizes
// are estimates from the product photos and Betaflight board documentation.
module aikon_f43030(z) {
    translate([0, 0, z]) {
        color("#25252a") difference() {
            plate(36, 36, 1.6, 1.9);
            for (x = [-15.25, 15.25], y = [-15.25, 15.25])
                hole_at([x, y], 4.3, -0.1, 2);
        }
        for (x = [-15.25, 15.25], y = [-15.25, 15.25])
            color("#3d3b40") translate([x, y, -0.5]) difference() {
                cylinder(d = 5.1, h = 2.65);
                translate([0, 0, -0.1]) cylinder(d = 3.15, h = 2.85);
            }
        color("#25272d") {
            translate([-4.5, -4.5, 1.56]) cube([9, 9, 1.25]);
            translate([5, 5.5, 1.56]) cube([5.1, 4.5, 1.1]);
            translate([-12.7, 7, 1.56]) cube([4, 4, 1]);
            translate([-2, 9, 1.56]) cube([4, 3, 0.9]);
        }
        color("silver") translate([13.1, -3.9, 2.77])
            rotate([0, 0, 90]) nano_shield([7.8, 9.8, 2.4], [[0, 6.5, 1.55]]);
        color("#cec9b7") translate([-7.6, -15.5, 1.56]) cube([15.2, 3.9, 2.5]);
        color("#d9aa57") for (y = [-9 : 3 : 9])
            translate([-15.2, y, 1.55]) cylinder(d = 1.55, h = 0.12);
    }
}

module flight_stack() {
    z = 3 + 12 * spread;
    size = stack_pitch == 20 ? 28 : 40;
    color("#c5ad76") translate([0, 0, z]) stack_adapter();
    for (p = adapter_mounts) screw(p, z - 3, 5.5);
    for (x = [-stack_pitch / 2, stack_pitch / 2], y = [-stack_pitch / 2, stack_pitch / 2]) {
        color("#27282c") standoff([x, y], z + 2.5, 11 + 9 * spread, 4.8, stack_pitch == 20 ? 2 : 3);
        screw([x, y], z + 7 + 10 * spread, 10, stack_pitch == 20 ? 2 : 3);
    }
    electronics_board(size, z + 4.5, false);
    if (stack_pitch == 30.5)
        aikon_f43030(z + 13 + 10 * spread);
    else
        electronics_board(size, z + 13 + 10 * spread, true);
}

module nano_tray() {
    color("#242933") translate([0, 0, tray_z + 52 * spread]) difference() {
        union() {
            for (y = [-22.5, 36]) translate([-3, y, 0]) plate(101, 11, 3.2, 3);
            for (x = [-32, 32]) translate([x, 7, 0]) plate(11, 66, 3.2, 3);
        }
        for (p = nano_hole_centres()) hull()
            for (dx = [-1.2, 1.2]) hole_at([p.x + dx, p.y], 3.3, -0.1, 3.5);
        for (p = board_mounts) hole_at(p, 3.2, -0.1, 3.5);
    }
    color("#888f99") {
        for (p = board_mounts) standoff(p, top_z + 2.4, tray_z - top_z - 2.4 + 30 * spread, 6.4, 3.1);
        for (p = nano_hole_centres()) standoff(p, tray_z + 3.2 + 52 * spread,
                                              nano_z - tray_z - 3.2 + 16 * spread, 6, 3);
    }
    translate([0, 0, nano_z + 68 * spread]) jetson_b01();
    for (p = nano_hole_centres()) screw(p, nano_z + 68 * spread - 2, 3.8);
}

module camera_and_mount() {
    translate(camera_origin + [0, 0, 18 * spread]) {
        color("#333945") for (y = [-13, 13])
            translate([0, y, -12]) hull() {
                translate([-8, -2, -5.05]) cube([18, 4, 3]);
                translate([0, 0, 12]) rotate([90, 0, 0]) cylinder(d = 9, h = 4, center = true);
            }
        rotate([0, -camera_tilt, 0]) {
            // Waveshare IMX219-160, 25 x 24 mm. Holes and lens depth are estimates.
            color("#30614b") difference() {
                translate([-2, -12.5, -12]) cube([1.5, 25, 24]);
                for (y = [-10.5, 10.5], z = [-9, 9])
                    translate([-2.1, y, z]) rotate([0, 90, 0]) cylinder(d = 2.2, h = 1.7);
            }
            color("#d9d0b5") translate([-3.6, -8, -10.5]) cube([1.8, 16, 4]);
            color("#26272b") translate([-0.75, -8, -8]) cube([4, 16, 16]);
            color("#161920") rotate([0, 90, 0]) {
                translate([0, 0, 3]) cylinder(d = 12, h = 8);
                translate([0, 0, 10.8]) cylinder(d = 14, h = 2);
            }
            color("#314e69") translate([12.7, 0, 0]) rotate([0, 90, 0])
                cylinder(d = 9, h = 0.7);
        }
    }
}

module battery_installation() {
    translate([-5, 0, battery_z - 32 * spread]) rotate([0, 0, 180]) battery_model();
    color("#24272d") translate([-5, 0, -5]) plate(78, 34, 1.8, 3);
    color("#333844") for (x = [-20, 17])
        translate([x, 0, -15.2 - 20 * spread]) difference() {
            cube([10, 42.6, 38.2], center = true);
            cube([11, 39.8, 35.7], center = true);
        }
    color("#242933") for (x = [-37, 36], y = [-22, 22])
        hull() {
            translate([x, y, -1]) cube([10, 5, 3], center = true);
            translate([x, y * 1.12, -37]) cube([18, 5, 3], center = true);
        }
}

module power_installation() {
    translate(regulator_origin + [0, 0, 15 * spread]) {
        power_module_model();
        color("#a4a9af") for (x = [-26.5, 26.5], y = [-15, 15])
            standoff([x, y], -2.2, 2.2, 5.8, 3);
    }
    color("#414856") translate([regulator_origin.x, 0, 3]) difference() {
        plate(61.5, 39.5, 2.8, 4);
        for (x = [-26.5, 26.5], y = [-15, 15]) hole_at([x, y], 3.2, -0.1, 3.1);
        for (x = [-13, 0, 13]) translate([x, 0, -0.1]) plate(6, 20, 3, 2);
    }
}

module harness_wire(points, diameter) {
    for (i = [0 : len(points) - 2]) hull()
        for (p = [points[i], points[i + 1]]) translate(p) sphere(d = diameter, $fn = 12);
}

module csi_ribbon() {
    route = [[63, 0, 19], [56, 5, 26], [48, 7, 35], [39, 8, 41],
             [-45, 8, 41], [-49, 6, 51]];
    color("#e0d5ab") for (i = [0 : len(route) - 2]) hull()
        for (p = [route[i], route[i + 1]])
            translate(p) cube([0.8, 15.5, 0.32], center = true);
}

module power_and_signal_harness() {
    for (p = motor_centres, bend = [-1, 0, 1]) {
        direction_x = p.x < 0 ? -1 : 1;
        direction_y = p.y < 0 ? -1 : 1;
        color("#343439") harness_wire([
            [direction_x * 15, direction_y * (8 + 1.8 * bend), 9],
            [direction_x * 24, direction_y * (15 + 1.8 * bend), 6.1],
            [p.x - direction_x * 23, p.y - direction_y * (23 + 1.8 * bend), 6.1],
            [p.x - direction_x * 10, p.y - direction_y * (10 + 1.8 * bend), 8]
        ], 1.8);
    }
    for (i = [-1, 1]) color(i < 0 ? "#25262b" : "#b52b2b") {
        harness_wire([[-96, i * 3.55, -19], [-104, i * 3, -19],
                      [-106, i * 3, -1], [-98, i * 2.54, 8],
                      [-81, i * 2.54, 13]], 2.5);
        harness_wire([[-98, i * 3, 1], [-87, -15 + i * 2.5, 2],
                      [-35, -20 + i * 2, 4], [-19, -13 + i * 2, 10]], 2.5);
        harness_wire([[-22, i * 2.54, 13], [-17, i * 3, 15],
                      [-21, -20 + i * 2, 25], [-28, -31 + i * 2, 37],
                      [-43, -53 + i * 1.5, nano_z + 7.6]], 2.1);
    }
    translate(nano_j25_centre() + [0, -0.5, nano_z]) rotate([90, 0, 0]) {
        color("silver") translate([0, 0, -8]) difference() {
            cylinder(d = 5.5, h = 9.5);
            translate([0, 0, -0.1]) cylinder(d = 2.1, h = 9.7);
        }
        color("#292b30") translate([0, 0, 0.5]) cylinder(d = 9, h = 13);
    }
    csi_ribbon();
    color("#e0b64d")
        harness_wire([[11, -15, 18], [18, -11, 25], [20, 20, 40],
                      [37, 29, 47], [40, 27, 52]], 1.2);
}

module sweep_disks() {
    color([1, 0.3, 0.13, 0.16]) for (p = motor_centres)
        translate([p.x, p.y, prop_z - sweep_half_height])
            cylinder(r = prop_radius, h = sweep_half_height * 2, $fn = 100);
}

assert(stack_pitch == 20 || stack_pitch == 30.5);
assert(view == "assembled" || view == "exploded" || view == "clearance" || view == "stack-adapter");
assert(len(motor_centres) == 4);
for (i = [0 : 2], j = [i + 1 : 3])
    assert(norm(motor_centres[i] - motor_centres[j]) > 2 * prop_radius,
           "Motor centres produce overlapping propeller disks.");
assert(tray_z > prop_z + sweep_half_height);
assert(top_z + 2.4 < prop_z - sweep_half_height);
assert(regulator_origin.z + 18 < prop_z - sweep_half_height);

if (view == "stack-adapter") stack_adapter();
else {
    carbon_frame();
    for (i = [0 : 3]) {
        quad_motor(motor_centres[i]);
        if (show_propellers) propeller(i);
    }
    flight_stack();
    nano_tray();
    camera_and_mount();
    battery_installation();
    power_installation();
    if (view != "exploded") power_and_signal_harness();
    if (view == "clearance") sweep_disks();
}
