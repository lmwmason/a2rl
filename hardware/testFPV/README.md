# testFPV 5-inch retrofit

Open `assembly.scad` in OpenSCAD. The customizer exposes the assembled, exploded and propeller-clearance views and the 20×20 and 30.5×30.5 mm stack options. `show_canopy` controls whether the shell covers the installed components. Select `canopy`, `nano-guard` or `stack-adapter` to inspect an individual part, or `carbon-plate` to inspect the flat DXF profile. Units are millimetres. The nose points along +X.

| Part | Geometry and fit status |
| --- | --- |
| Frame, four motors and 5-inch propellers | A single flat carbon plate includes the four arms and the extended central spine. The reference X frame retains its 220 mm diagonal wheelbase. Enter the existing frame's measured motor centres, plate outline and attachment locations before fabrication. Motor bodies, bolt pattern and propeller geometry are representative. |
| FC and ESC | Aikon F43030 FC with its published 30.5 mm pitch. The lower board represents the matching AK32 layout until the installed ESC variant is identified. The 20 mm stack uses representative boards. |
| Jetson Nano 4 GB | P3449 B01 reference built from the carrier specification drawings. The user's carrier revision is unconfirmed. Hole locations are drawing estimates. NVIDIA's official STEP download requires an NVIDIA sign-in. |
| AI camera | Waveshare IMX219-160, 25 × 24 mm CSI module. The tilt starts at an assumed 55° above the frame plane. Measure lens height and mounting holes and calibrate the captured field of view for the actual capture mode. |
| Battery and converter | DOGCOM UCELL DC0058 and DeviceMart SZH-PWSDF-036. See the component files for dimensions and photo-based estimates. |

The Jetson mounts inverted under the carbon plate with its heatsink fins and ports accessible. A polymer guard surrounds the underside and sets the landing plane below the heatsink. The battery sits in a polymer cradle above the flight stack. The converter sits behind the stack. The removable polymer canopy tapers toward the nose and wraps the battery, electronics and camera mount. The canopy geometry includes a lens opening, vents and propeller clearance scallops. Fasteners remain separate hardware.

Cable paths show the XT60 branch to the ESC and converter, the converter's 5 V branch to J25, motor wiring, the CSI ribbon and the FC signal path. They serve as mechanical routing references. Identify the FC and ESC revisions before selecting actual pins or cable pinouts.

The clearance view depicts the 127 mm diameter propeller sweep with a provisional ±4 mm vertical allowance. The canopy cutouts use an additional 3 mm radial and 1 mm vertical allowance. The model checks propeller disc overlap, converter height, the battery floor above the stack and the landing plane below the heatsink. Final fit depends on the measured frame, fastener locations, cable bend radii and unobstructed camera field of view. Before flight, verify guard deflection, loaded centre of gravity and XL4015 voltage and temperature at a sustained 4 A load.

[Aikon F43030 specifications](https://aikon-electronics.com/index.php?id=25) · [Aikon AK32 35A specifications](https://aikon-electronics.com/index.php?id=20) · [NVIDIA P3449 B01 drawing](https://developer.nvidia.com/jetson-nano-developer-kit-carrier-board-p3449-b01-specification) · [Waveshare IMX219-160 specifications](https://www.waveshare.com/wiki/IMX219-160_Camera)
