// P3449 B01 carrier outline and connector arrangement from NVIDIA SP-09732,
// figures 4-1 and 4-2. Hole centres and connector details are drawing estimates.
// https://developer.nvidia.com/jetson-nano-developer-kit-carrier-board-p3449-b01-specification
// The port edge faces -Y. Z=0 is the bottom face of the carrier PCB.

function nano_hole_centres() = [[-46, -22.5], [-46, 36], [40, -22.5], [40, 36]];
function nano_j25_centre() = [-44, -40, 7.6];

module nano_shield(size, openings) {
    color("#aeb8bc") difference() {
        cube(size, center = true);
        for (port = openings)
            translate([0, -size[1] / 2, port[0]])
                cube([port[1], 4, port[2]], center = true);
    }
    color("#242930") for (port = openings)
        translate([0, -size[1] / 2 + 1.8, port[0]])
            cube([port[1], 0.4, port[2]], center = true);
}

module jetson_b01() {
    color("#164d32") difference() {
        translate([-50, -39.5, 0]) cube([100, 79, 1.58]);
        for (p = nano_hole_centres()) translate([p.x, p.y, -0.1])
            cylinder(d = 3.15, h = 2, $fn = 24);
    }
    color("#1b4630") translate([-37, -8, 5]) cube([70, 45, 1.6]);
    color("#262930") {
        translate([-34, -5, 6.4]) cube([61, 42, 5.8]);
        for (x = [-32.8 : 3.5 : 25])
            translate([x, -5, 12]) cube([1.2, 42, 16.28]);
        translate([-34, 11, 12]) cube([61, 4, 6.6]);
    }
    for (x = [-9, 9])
        translate([x, -31.8, 8.8]) nano_shield([14, 16.5, 14.4], [[-3.4, 12, 5], [3.4, 12, 5]]);
    translate([-26, -31.8, 11.25]) nano_shield([16, 16.5, 19.35], [[-4.4, 13, 5], [4.4, 13, 5]]);
    translate([27, -31.8, 8.15]) nano_shield([16, 16.5, 13.1], [[0, 12, 8]]);
    translate([43, -35.4, 3.7]) nano_shield([8.3, 8.8, 4.3], [[0, 6.4, 1.8]]);
    color("#263033") translate(nano_j25_centre()) difference() {
        translate([-4.5, 0, -6]) cube([9, 14, 11]);
        rotate([-90, 0, 0]) translate([0, 0, -0.1])
            cylinder(d = 5.5, h = 10, $fn = 32);
    }
    color("silver") translate(nano_j25_centre())
        rotate([-90, 0, 0]) cylinder(d = 2, h = 8, $fn = 24);
    color("#27282b") translate([37.5, -17, 1.57]) cube([5.8, 50.4, 2.1]);
    color("#cfbb71") for (x = [39, 41.54], i = [0 : 19])
        translate([x, -15 + i * 2.54, 3.55]) cube([0.64, 0.64, 4.2]);
    for (y = [2, 25]) {
        color("#ebe5cc") translate([-48.5, y - 8.5, 1.57]) cube([4.3, 17, 3.6]);
        color("#282c32") translate([-48.8, y - 7.5, 4.7]) cube([1.9, 15, 1]);
    }
}

jetson_b01();
