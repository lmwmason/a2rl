// SMG-A SZH-PWSDF-036 (XL4015). All dimensions are in millimetres.
// Overall size and mass (28 g) are from the DeviceMart listing.
// https://www.devicemart.co.kr/goods/view?no=1321262
// Component positions and hole locations are estimates from the product photos.
// Origin: centre of the underside of the PCB. Input is on the -X end.

module_size = [60, 37, 18];
underboard_clearance = 3;
side_clearance = 3;
top_clearance = 4;
input_wire_clearance = 15;
output_wire_clearance = 15;
view = "reference"; // [reference,keepout]
$fn = 32;

module xl4015_pcb() {
    color("#125878") difference() {
        hull() for (x = [-29.3, 29.3], y = [-17.8, 17.8])
            translate([x, y, 0]) cylinder(r = 0.7, h = 1.6);
        for (x = [-26.5, 26.5], y = [-15, 15])
            translate([x, y, -1]) cylinder(d = 3, h = 4);
    }
    color("#b7af91") for (x = [-26.5, 26.5], y = [-15, 15])
        translate([x, y, 1.55]) difference() {
            cylinder(d = 4.5, h = 0.13);
            translate([0, 0, -0.1]) cylinder(d = 3, h = 0.4);
        }
}

module xl4015_terminal(direction) {
    color("#157eae") difference() {
        hull() {
            translate([-4.2, -5.1, 0]) cube([8.4, 10.2, 6.5]);
            translate([-3.4, -5.1, 9]) cube([6.8, 10.2, 0.1]);
        }
        for (y = [-2.54, 2.54]) {
            translate([0, y, 4.5]) cylinder(d = 3.35, h = 5);
            translate([direction * 3.5, y, 3]) rotate([0, 90, 0])
                cylinder(d = 3, h = 2, center = true);
        }
    }
    color("silver") for (y = [-2.54, 2.54])
        translate([0, y, 4.6]) difference() {
            cylinder(d = 2.8, h = 0.7);
            translate([-2, -0.15, 0.5]) cube([4, 0.3, 0.3]);
        }
}

module xl4015_capacitor() {
    color("#303234") cylinder(d = 9.1, h = 0.7);
    color("#9b9da1") translate([0, 0, 0.55]) cylinder(d = 8.4, h = 10.8);
    color("#c7c9cd") translate([0, 0, 11.3]) difference() {
        cylinder(d = 8, h = 0.3);
        for (angle = [0, 90]) rotate([0, 0, angle])
            translate([-2, -0.15, 0.15]) cube([4, 0.3, 0.2]);
    }
}

module xl4015_torus(major, radial, vertical) {
    rotate_extrude($fn = 64) translate([major, 0])
        scale([radial, vertical]) circle(r = 1, $fn = 24);
}

// The continuous windings follow the radial cross-section of the core.
module xl4015_coil() {
    color("#2e302e") xl4015_torus(4.9, 2.5, 4.45);
    color("#be6c27") for (angle = [0 : 15 : 345])
        rotate([0, 0, angle]) translate([4.9, 0, 0])
            scale([2.82, 0.34, 4.76]) rotate([90, 0, 0])
                rotate_extrude($fn = 28)
                    translate([0.95, 0]) circle(r = 0.15, $fn = 10);
}

module xl4015_display() {
    color("#e1e1d8") translate([-11.4, -6.3, 0]) cube([22.8, 12.6, 6.8]);
    color("#1a1b20") translate([-11.4, -6.3, 6.75]) cube([22.8, 12.6, 0.85]);
    color("#737476") for (x = [-7.2, -0.2, 6.8]) translate([x, 0, 7.59]) {
        for (y = [-4.5, 0, 4.5])
            translate([0, y, 0.035]) cube([3.9, 0.6, 0.1], center = true);
        for (dx = [-2.2, 2.2], y = [-2.25, 2.25])
            translate([dx, y, 0.035]) cube([0.6, 3.45, 0.1], center = true);
        translate([3.1, -4.55]) cylinder(d = 0.8, h = 0.1);
    }
}

module xl4015_switch() {
    color("silver") translate([-3, -3, 0]) cube([6, 6, 2.2]);
    color("#24252a") translate([0, 0, 2.1]) cylinder(d = 3.55, h = 2.8);
    color("#28292d") for (x = [-2.1, 2.1], y = [-2.1, 2.1])
        translate([x, y, 2.15]) cylinder(d = 0.75, h = 0.25);
}

module xl4015_ic() {
    color("silver") translate([-5.1, -6, 0]) cube([10.2, 9.7, 0.9]);
    color("#24252a") translate([-4.9, -4.1, 0.75]) cube([9.8, 8.6, 3.75]);
    color("silver") for (x = [-3.4, -1.7, 0, 1.7, 3.4])
        hull() {
            translate([x - 0.3, 4.1, 1.55]) cube([0.6, 0.6, 0.45]);
            translate([x - 0.3, 6.5, 0]) cube([0.6, 1.1, 0.38]);
        }
}

module power_module_model() {
    xl4015_pcb();
    for (direction = [-1, 1])
        translate([direction * 25.6, 0, 1.57]) xl4015_terminal(direction);
    for (x = [-17.8, 18]) translate([x, 11.25, 1.57]) xl4015_capacitor();
    translate([7.8, 3.45, 6.08]) xl4015_coil();
    translate([-6, 4.25, 1.57]) xl4015_ic();
    translate([0, -10.05, 1.57]) xl4015_display();
    for (x = [-17.7, 17.7]) translate([x, -12.25, 1.57]) xl4015_switch();
    translate([7.7, 15.05, 1.57]) {
        color("#1b55a0") translate([-4.65, -2.35, 0]) cube([9.3, 4.7, 13.8]);
        color("#b18a38") translate([-2.7, 0.9, 13.75]) difference() {
            cylinder(d = 2.5, h = 2.68);
            translate([-2, -0.18, 2.25]) cube([4, 0.36, 0.5]);
        }
    }
    for (x = [-19, 19], y = [-5.25, 4.2])
        translate([x - 1.3, y - 1.9, 1.57]) {
            color("silver") cube([2.6, 3.8, 0.55]);
            color("#262728") translate([0, 0.55, 0.5]) cube([2.6, 2.7, 1.3]);
        }
}

module power_keepout() {
    width = module_size[1] + 2 * side_clearance;
    translate([-module_size[0] / 2 - input_wire_clearance,
               -width / 2, -underboard_clearance])
        cube([module_size[0] + input_wire_clearance + output_wire_clearance,
              width, module_size[2] + underboard_clearance + top_clearance]);
}

assert(view == "reference" || view == "keepout");
assert(min(underboard_clearance, side_clearance, top_clearance,
           input_wire_clearance, output_wire_clearance) >= 0);

if (view == "keepout")
    power_keepout();
else
    power_module_model();
