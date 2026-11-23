// testFPV retrofit. Millimetres. +X is forward, +Z is up.
// The frame and unidentified parts are provisional reference geometry.
// Change the four motor centres to the measured existing frame before making parts.
use <battery.scad>
use <power-module.scad>
use <jetson-b01.scad>
use <fairing.scad>

/* [Assembly] */
view = "assembled"; // [assembled,exploded,clearance,canopy,battery-hatch,nano-guard,skid-shoes,carbon-plate,stack-adapter]
stack_pitch = 30.5; // [20,30.5]
wheelbase = 220;
motor_x = wheelbase / sqrt(8);
motor_y = wheelbase / sqrt(8);
motor_centres = [[motor_x, motor_y], [motor_x, -motor_y],
                 [-motor_x, motor_y], [-motor_x, -motor_y]];
camera_tilt = 55;
show_canopy = true;
show_battery_hatch = true;
show_nano_pod = true;
show_propellers = true;

/* [Hidden] */
$fn = 32;
arm_thickness = 5;
prop_radius = 63.5;
prop_z = 32;
sweep_half_height = 4;
nano_z = -8;
pod_bottom = -51;
landing_z = -55;
pod_wall = 2.6;
battery_z = 28;
regulator_origin = [-53, 0, 8];
camera_origin = [70, 0, 26];
board_mounts = [[-32, -17], [-32, 17], [32, -17], [32, 17]];
adapter_mounts = [[0, -17.5], [0, 17.5]];
hatch_screw = [-63, 13];
hatch_head_z = 44.35;
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
            hull() {
                translate([-47, 0, 0]) plate(91, 46, arm_thickness, 8);
                translate([53, 0, 0]) plate(80, 34, arm_thickness, 8);
            }
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
        for (x = [62, 76], y = [-13, 13]) hole_at([x, y], 3.2, -0.1, 6);
        for (x = [-17, 32], y = [-20, 20]) hole_at([x, y], 3.2, -0.1, 6);
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
    z = arm_thickness + 12 * spread;
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

function underbody_hole(p) = [p.x, -p.y];
function underbody_port() = let(p = nano_j25_centre()) [p.x, -p.y, nano_z - p.z];

module nano_mount() {
    color("#343944") translate([0, 0, -3 - 12 * spread]) difference() {
        union() {
            for (x = [-32, 32]) translate([x, -6, 0]) plate(12, 71, 3, 3);
            for (y = [-36, 22.5]) translate([-3, y, 0]) plate(101, 12, 3, 3);
        }
        for (p = nano_hole_centres()) hull()
            for (dx = [-1.2, 1.2]) hole_at([p.x + dx, -p.y], 3.3, -0.1, 3.2);
        for (p = board_mounts) hole_at(p, 3.2, -0.1, 3.2);
    }
    color("#343944") for (p = nano_hole_centres())
        standoff(underbody_hole(p), nano_z - 19 * spread, 5 + 7 * spread, 6.5, 3.2);
    translate([0, 0, nano_z - 19 * spread]) rotate([180, 0, 0]) jetson_b01();
    for (p = nano_hole_centres()) translate([p.x, -p.y, nano_z - 19 * spread - 3])
        rotate([180, 0, 0]) screw([0, 0], -5, 5);
}

module pod_vents() {
    // Vent openings sit fore and aft of the fins, above the belly and skids.
    for (side = [-1, 1], x = [-64, -53, 53, 64], z = [-36, -26])
        hull() for (dx = [-2.6, 2.6])
            translate([x + dx, side * 48, z]) rotate([90, 0, 0])
                cylinder(d = 3.8, h = 36, center = true, $fn = 20);
}

module nano_guard() {
    stations = [[-84, 20, -27, -1], [-73, 69, -42, -1], [-59, 113, -48, -1],
                [-46, 122, pod_bottom, -1], [-28, 116, pod_bottom, -1],
                [24, 104, pod_bottom, -1], [49, 100, -48, -1],
                [65, 78, -41, -1], [79, 24, -24, -1]];
    cavity = fair_cavity(stations, pod_wall, true, 2);
    difference() {
        union() {
            difference() {
                fair_loft(stations, 11, 6);
                fair_loft(cavity, 8.4, 3.4);
            }
            // Four bridges share the carrier fasteners near the carbon spine.
            for (p = board_mounts) hull() {
                translate([p.x, p.y, -6.5]) cylinder(d = 10, h = 3.4);
                side_y = p.x < 0 ? 54 : 46.5;
                translate([p.x, p.y < 0 ? -side_y : side_y, -6.5]) cylinder(d = 8, h = 3.4);
            }
            // Inward lips shade the top edges while keeping the pod removable.
            intersection() {
                fair_loft(stations, 11, 6);
                for (side = [-1, 1]) translate([0, side * 56, -2.45])
                    cube([168, 28, 1.9], center = true);
            }
            for (x = [-37, 21], y = [-32, 32])
                translate([x, y, pod_bottom + 0.2]) cylinder(d = 9, h = 6.2);
        }
        pod_vents();
        translate([-43, 40, -4]) cube([17, 11, 4]);
        for (p = board_mounts) hole_at(p, 3.3, -6.6, 3.6);
        for (x = [-37, 21], y = [-32, 32]) hole_at([x, y], 2.6, landing_z, 9.3);
    }
}

module skid_shoes() {
    // Replaceable polymer shoes sit below the continuous pod floor.
    for (y = [-32, 32]) difference() {
        hull() for (x = [-42, 26]) {
            translate([x, y, landing_z]) cylinder(r = 4.2, h = 5);
            translate([x, y, pod_bottom + 1]) cylinder(r = 5, h = 1);
        }
        for (x = [-37, 21]) {
            hole_at([x, y], 3.2, landing_z - 0.1, 7);
            hole_at([x, y], 6, landing_z - 0.1, 2);
        }
    }
}

module blade_clearance_cuts() {
    for (p = motor_centres) translate([p.x, p.y, prop_z - sweep_half_height - 1])
        cylinder(r = prop_radius + 3, h = 2 * sweep_half_height + 2, $fn = 100);
}

function canopy_sections() = [[-90, 40, 5, 27], [-77, 46, 5, 36], [-49, 49, 5, 53],
                              [-37, 48, 5, 63], [42, 48, 5, 63], [53, 42, 5, 55],
                              [66, 36, 5, 46], [86, 19, 5, 24]];

module canopy_shell() {
    // Spine runs nose to tail. Side scallops keep the shell outside the prop volume.
    sections = canopy_sections();
    inner = fair_cavity(sections, 2.4, false, 2);
    difference() {
        union() {
            fair_loft(sections, 3.4, 7.5);
            for (p = board_mounts) translate([p.x, p.y, 5]) cylinder(d = 11, h = 5);
        }
        fair_loft(inner, 1, 5.1);
        blade_clearance_cuts();
        // Lens aperture along the tilted optical axis. Actual view requires calibration.
        translate(camera_origin) rotate([0, 90 - camera_tilt, 0])
            translate([0, 0, 3]) cylinder(d1 = 17, d2 = 38, h = 36, $fn = 48);
        translate(camera_origin) rotate([0, -camera_tilt, 0])
            translate([-4.5, -13.2, -12.8]) cube([5.4, 26.4, 25.6]);
        for (y = [-13, 13])
            translate([66.5, y - 2.9, 4.8]) cube([14.2, 5.8, 23.2]);
        for (x = [-25 : 8 : 23]) translate([x, 0, 60.4]) plate(3.5, 17, 3, 1.5);
        for (x = [-78, -69, -60], y = [-1, 1])
            translate([x, y * 18, 20]) cube([4, 17, 5], center = true);
        translate([-94, -9, 6]) cube([10, 18, 11]);
    }
    // Through tabs connect the canopy to the carbon plate near the centre.
    for (p = board_mounts) translate([p.x, p.y, 5]) difference() {
        hull() {
            cylinder(d = 10.5, h = 2.6);
            translate([0, p.y < 0 ? -6 : 6, 0]) cylinder(d = 5.2, h = 2.6);
        }
        translate([0, 0, -0.1]) cylinder(d = 3.3, h = 2.8);
    }
}

module battery_access(gap = 0) {
    // Main opening clears the 80 x 39 mm pack. The rear pocket exposes XT60.
    translate([0, 0, 30]) plate(89 + 2 * gap, 44 + 2 * gap, 43, 4);
    translate([-64, 0, 25]) plate(46 + 2 * gap, 36 + 2 * gap, 48, 4);
}

module canopy() {
    difference() {
        union() {
            difference() {
                canopy_shell();
                battery_access();
            }
            intersection() {
                fair_loft(canopy_sections(), 3.4, 7.5);
                hull() {
                    translate([hatch_screw.x, hatch_screw.y, 36.5]) cylinder(d = 8, h = 3.1);
                    translate([hatch_screw.x, 23, 36.5]) cylinder(d = 6, h = 3.1);
                }
            }
        }
        blade_clearance_cuts();
        hole_at(hatch_screw, 2.6, 36.3, 3.5);
        for (y = [-10, 10])
            translate([44.4, y - 3, 56.7]) cube([5.5, 6, 2.4]);
    }
}

module battery_hatch() {
    difference() {
        union() {
            intersection() {
                canopy_shell();
                battery_access(-0.55);
            }
            for (y = [-10, 10]) {
                translate([40.5, y - 2.4, 57]) cube([8.5, 4.8, 1.6]);
                translate([40.5, y - 2.4, 58.5]) cube([3, 4.8, 3.6]);
            }
        }
        hole_at(hatch_screw, 3.3, 39, 8);
    }
}

module hatch_fastener() {
    translate([hatch_screw.x, hatch_screw.y, hatch_head_z]) {
        cylinder(d = 7.5, h = 2.8, $fn = 24);
        translate([0, 0, -7.3]) cylinder(d = 3, h = 7.4);
    }
}

module camera_and_mount() {
    translate(camera_origin + [0, 0, 18 * spread]) {
        color("#333945") for (y = [-13, 13])
            translate([0, y, -12]) hull() {
                translate([-8, -2, -9.05]) cube([18, 4, 3]);
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
    translate([0, 0, battery_z + 25 * spread]) rotate([0, 0, 180]) battery_model(23);
    color("#383e49") {
        translate([0, 0, battery_z - 2 + 18 * spread]) difference() {
            plate(84, 43.4, 2, 5);
            for (x = [-15, 20], y = [-20, 20])
                translate([x, y, -0.1]) plate(11, 1.8, 2.25, 0.8);
        }
        for (x = [-17, 32], y = [-20, 20])
            standoff([x, y], 5, battery_z - 7 + 18 * spread, 5.5, 3.25);
    }
    color("#333844") for (x = [-15, 20]) {
        translate([x, 0, battery_z + 13.5 + 25 * spread]) difference() {
            cube([10, 42.2, 31], center = true);
            cube([11, 39.8, 28.7], center = true);
        }
        translate([x, 6, battery_z + 28.5 + 25 * spread]) plate(10, 12, 1, 2);
    }
}

module power_installation() {
    translate(regulator_origin + [0, 0, 15 * spread]) {
        power_module_model();
        color("#a4a9af") for (x = [-26.5, 26.5], y = [-15, 15])
            standoff([x, y], -1.4, 1.4, 5.8, 3);
    }
    color("#414856") translate([regulator_origin.x, 0, 5]) difference() {
        plate(61.5, 39.5, 1.6, 4);
        for (x = [-26.5, 26.5], y = [-15, 15]) hole_at([x, y], 3.2, -0.1, 1.8);
        for (x = [-13, 0, 13]) translate([x, 0, -0.1]) plate(6, 20, 1.8, 2);
    }
}

module harness_wire(points, diameter) {
    for (i = [0 : len(points) - 2]) hull()
        for (p = [points[i], points[i + 1]]) translate(p) sphere(d = diameter, $fn = 12);
}

module csi_ribbon() {
    route = [[66, 0, 22], [55, 3, 15], [51, 0, 6], [51, 0, -4],
             [43, -6, -5], [-45, -6, -5], [-49, -6, -12]];
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
        harness_wire([[-84, i * 3.55, 22], [-90, i * 3, 19],
                      [-92, i * 2.54, 13], [-88, i * 2.54, 11],
                      [-82, i * 2.54, 11]], 2.5);
        harness_wire([[-92, i * 2.54, 13], [-86, -10 + i * 2.5, 8],
                      [-35, -20 + i * 2, 8], [-19, -13 + i * 2, 12]], 2.5);
        harness_wire([[-23, i * 2.54, 11], [-19, 10 + i * 2, 10],
                      [-25, 26 + i * 2, 7], [-34, 36 + i * 2, 1],
                      [-43, 55 + i * 1.5, nano_z - 7.6]], 2.1);
    }
    translate(underbody_port() + [0, 0.5, 0]) rotate([-90, 0, 0]) {
        color("silver") translate([0, 0, -8]) difference() {
            cylinder(d = 5.5, h = 9.5);
            translate([0, 0, -0.1]) cylinder(d = 2.1, h = 9.7);
        }
        color("#292b30") translate([0, 0, 0.5]) cylinder(d = 9, h = 13);
    }
    csi_ribbon();
    color("#e0b64d")
        harness_wire([[11, -15, 20], [21, -18, 12], [36, -23, 5],
                      [47, -22, -5], [45, -27, -11], [40, -27, -13]], 1.2);
}

module sweep_disks() {
    color([1, 0.3, 0.13, 0.16]) for (p = motor_centres)
        translate([p.x, p.y, prop_z - sweep_half_height])
            cylinder(r = prop_radius, h = sweep_half_height * 2, $fn = 100);
}

assert(stack_pitch == 20 || stack_pitch == 30.5);
assert(view == "assembled" || view == "exploded" || view == "clearance"
       || view == "stack-adapter" || view == "canopy" || view == "nano-guard"
       || view == "skid-shoes" || view == "carbon-plate" || view == "battery-hatch");
assert(len(motor_centres) == 4);
for (i = [0 : 2], j = [i + 1 : 3])
    assert(norm(motor_centres[i] - motor_centres[j]) > 2 * prop_radius,
           "Motor centres produce overlapping propeller disks.");
assert(regulator_origin.z + 18 < prop_z - sweep_half_height);
assert(nano_z - 28.28 > pod_bottom + pod_wall + 8);
assert(battery_z - 2 > 5 + 13 + 5);

if (view == "stack-adapter") stack_adapter();
else if (view == "canopy") canopy();
else if (view == "battery-hatch") battery_hatch();
else if (view == "nano-guard") nano_guard();
else if (view == "skid-shoes") skid_shoes();
else if (view == "carbon-plate") projection(cut = true)
    translate([0, 0, -arm_thickness / 2]) carbon_frame();
else {
    carbon_frame();
    for (i = [0 : 3]) {
        quad_motor(motor_centres[i]);
        if (show_propellers) propeller(i);
    }
    flight_stack();
    nano_mount();
    if (show_nano_pod && view != "clearance") {
        color("#303844") translate([0, 0, -51 * spread]) render(convexity = 12) nano_guard();
        color("#22252b") translate([0, 0, -60 * spread]) skid_shoes();
    }
    camera_and_mount();
    battery_installation();
    power_installation();
    if (view != "exploded") power_and_signal_harness();
    if (show_canopy && view != "clearance")
        color("#e58128") translate([0, 0, 66 * spread]) render(convexity = 12) canopy();
    if (show_canopy && show_battery_hatch && view != "clearance") {
        color("#39404a") translate([0, 0, 92 * spread]) render(convexity = 12) battery_hatch();
        color("#555963") translate([0, 0, 92 * spread]) hatch_fastener();
    }
    if (view == "clearance") sweep_disks();
}
