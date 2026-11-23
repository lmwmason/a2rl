# testFPV 5-inch retrofit

Open `assembly.scad` in OpenSCAD. The customizer exposes the assembled, exploded and propeller-clearance views and the 20×20 and 30.5×30.5 mm stack options. Use `show_battery_hatch` to see the battery service opening and `show_nano_pod` to inspect the Jetson mount. `canopy`, `battery-hatch`, `nano-guard`, `skid-shoes` and `stack-adapter` isolate individual parts. `carbon-plate` displays the flat DXF profile. Units are millimetres. The nose points along +X.

| Part | Geometry and fit status |
| --- | --- |
| Frame, four motors and 5-inch propellers | A single flat carbon plate includes the four arms and the extended central spine. The reference X frame retains its 220 mm diagonal wheelbase. Enter the existing frame's measured motor centres, plate outline and attachment locations before fabrication. Motor bodies, bolt pattern and propeller geometry are representative. |
| FC and ESC | Aikon F43030 FC with its published 30.5 mm pitch. The lower board represents the matching AK32 layout until the installed ESC variant is identified. The 20 mm stack uses representative boards. |
| Jetson Nano 4 GB | P3449 B01 reference built from the carrier specification drawings. The user's carrier revision is unconfirmed. Hole locations are drawing estimates. NVIDIA's official STEP download requires an NVIDIA sign-in. |
| AI camera | Waveshare IMX219-160, 25 × 24 mm CSI module. The tilt starts at an assumed 55° above the frame plane. Measure lens height and mounting holes and calibrate the captured field of view for the actual capture mode. |
| Battery and converter | DOGCOM UCELL DC0058 and DeviceMart SZH-PWSDF-036. See the component files for dimensions and photo-based estimates. |

The Jetson mounts inverted below the carbon plate inside a removable polymer pod. The floor is continuous beneath the heatsink. Raised side vents sit ahead of and behind the fins. The J25 plug sits within the sidewall envelope. Two replaceable skid shoes set the landing plane 18.7 mm below the heatsink tip in the nominal model. The skid holes are blind pilots in reinforced bosses. Match the screws to the printed boss depth.

The curved upper canopy follows a smooth longitudinal taper. The top hatch covers a nominal 89 × 44 mm opening over the pack and its straps, with a rear pocket for the XT60. Release the thumbscrew, lift the rear edge and slide the hatch rearward to disengage the front tabs. Disconnect and tuck the pack leads toward the pack, release the straps and lift the pack vertically. The converter sits behind the stack. Fasteners remain separate hardware.

Cable paths show the XT60 branch to the ESC and converter, the converter's 5 V branch to J25, motor wiring, the CSI ribbon and the FC signal path. They serve as mechanical routing references. Identify the FC and ESC revisions before selecting actual pins or cable pinouts.

The clearance view depicts the 127 mm diameter propeller sweep with a provisional ±4 mm vertical allowance. The canopy cutouts use an additional 3 mm radial and 1 mm vertical allowance. The model checks propeller disc overlap, converter height, the battery floor above the stack and the floor clearance below the heatsink. Before flight, verify hatch and skid retention, Jetson airflow and temperature under inference load, pad and pod deflection, loaded centre of gravity, and XL4015 voltage and temperature at a sustained 4 A load. Final fit depends on the measured frame, connector routing, fastener locations and camera field of view. Aerodynamic drag and crash loads have not been measured.

[Aikon F43030 specifications](https://aikon-electronics.com/index.php?id=25) · [Aikon AK32 35A specifications](https://aikon-electronics.com/index.php?id=20) · [NVIDIA P3449 B01 drawing](https://developer.nvidia.com/jetson-nano-developer-kit-carrier-board-p3449-b01-specification) · [Waveshare IMX219-160 specifications](https://www.waveshare.com/wiki/IMX219-160_Camera)
