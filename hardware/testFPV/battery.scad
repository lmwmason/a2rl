// DOGCOM UCELL DC0058, 4S 1550 mAh, XT60. Dimensions are in millimetres.
// Manufacturer specifies an 80 x 39 x 28 mm body and approximately 181 g.
// https://dogcombattery.com/products/dogcom-1550mah-150c-4s-148v-ucell-series
// Cable curves and connector details are estimated from the product photos.
// Origin: centre of the pack at its bottom surface. Leads exit toward +X.

pack_size = [80, 39, 28];
pad_clearance = 2;
strap_clearance = 2;
side_clearance = 2;
lead_clearance = 58;
view = "reference"; // [reference,keepout]
$fn = 24;

module battery_wire(path, diameter) {
    for (i = [0 : len(path) - 2])
        hull() for (point = [path[i], path[i + 1]])
            translate(point) sphere(d = diameter, $fn = 12);
}

module battery_xt60() {
    housing = [[0, -7.9], [14, -7.9], [16.2, -5.7], [16.2, 5.7],
               [14, 7.9], [0, 7.9]];
    color("#f0a926") difference() {
        union() {
            translate([0, 0, -4.1]) linear_extrude(8.2) polygon(housing);
            translate([3, -8.1, -4.4]) cube([3.4, 16.2, 8.8]);
        }
        for (y = [-3.55, 3.55])
            translate([9, y, 0]) rotate([0, 90, 0]) cylinder(d = 4.85, h = 8);
        for (x = [4, 5.3, 6.6])
            translate([x, -8.2, 2.9]) cube([0.45, 16.4, 1.5]);
    }
    color("#c7a257") for (y = [-3.55, 3.55])
        translate([8.9, y, 0]) rotate([0, 90, 0]) difference() {
            cylinder(d = 3.8, h = 5.5);
            translate([0, 0, 0.55]) cylinder(d = 2.8, h = 5.2);
        }
}

module battery_balance_plug() {
    color("#e4e3df") difference() {
        translate([0, -6.65, -2.4]) cube([7.5, 13.3, 4.8]);
        for (y = [-5.08, -2.54, 0, 2.54, 5.08])
            translate([4.6, y - 0.72, -0.85]) cube([3, 1.44, 1.7]);
    }
    color("#dbdad7") translate([1, -4.1, 2.25]) cube([3.4, 8.2, 1.1]);
}

module battery_body() {
    corner = 1.8;
    color("#171923") hull()
        for (x = [-1, 1], y = [-1, 1], z = [corner, pack_size[2] - corner])
            translate([x * (pack_size[0] / 2 - corner),
                       y * (pack_size[1] / 2 - corner), z])
                sphere(r = corner);
    color("#202128") translate([-29, -16.5, pack_size[2] - 0.08])
        cube([58, 33, 0.18]);
    color("#efdc36") translate([-24, 2, pack_size[2] + 0.095])
        linear_extrude(0.1) text("DOGCOM UCELL", size = 4.7);
    color("white") translate([-24, -7, pack_size[2] + 0.095])
        linear_extrude(0.1) text("4S  1550 mAh  150C", size = 3.7);
}

module battery_model() {
    battery_body();
    for (side = [-1, 1])
        color(side < 0 ? "#212127" : "#c5252c")
            battery_wire([[38, 10 + side * 2.4, 22],
                          [46, 10 + side * 2.45, 22],
                          [54, 9 + side * 2.7, 21],
                          [64, 4 + side * 3.2, 18],
                          [75, side * 3.55, 14]], 4);
    translate([74.8, 0, 14]) battery_xt60();
    for (i = [0 : 4])
        color(i == 4 ? "#c5252c" : "#25252b")
            battery_wire([[38, 4 + i * 1.25, 18.5],
                          [45, 2 + i * 1.6, 18.5],
                          [51, -6 + i * 2.54, 15],
                          [57, -11 + i * 2.54, 14]], 1.25);
    translate([56.7, -5.92, 14]) battery_balance_plug();
}

module battery_keepout() {
    translate([-pack_size[0] / 2, -pack_size[1] / 2 - side_clearance, -pad_clearance])
        cube([pack_size[0] + lead_clearance,
              pack_size[1] + 2 * side_clearance,
              pack_size[2] + pad_clearance + strap_clearance]);
}

assert(len(pack_size) == 3 && min(pack_size) > 3.6);
assert(min(pad_clearance, strap_clearance, side_clearance, lead_clearance) >= 0);
assert(view == "reference" || view == "keepout");

if (view == "keepout")
    battery_keepout();
else
    battery_model();
