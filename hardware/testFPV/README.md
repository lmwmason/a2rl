# testFPV 5-inch retrofit

Open `assembly.scad` in OpenSCAD. The customizer exposes the assembled, exploded and propeller-clearance views and the 20×20 and 30.5×30.5 mm stack options. `stack-adapter` shows the dual-pattern adapter for isolated inspection. Units are millimetres. The nose points along +X.

| Part | Geometry and fit status |
| --- | --- |
| Frame, four motors and 5-inch propellers | Reference X frame with 220 mm diagonal wheelbase. Enter the existing frame's four motor centres, plate outline and standoff locations before fabrication. Motor bodies, bolt pattern and propeller geometry are representative. |
| FC and ESC | Aikon F43030 FC with its published 30.5 mm pitch. The lower board represents the matching AK32 layout until the installed ESC variant is identified. The 20 mm stack uses representative boards. |
| Jetson Nano 4 GB | P3449 B01 reference built from the carrier specification drawings. The user's carrier revision is unconfirmed. Hole locations are drawing estimates. NVIDIA's official STEP download requires an NVIDIA sign-in. |
| AI camera | Waveshare IMX219-160, 25 × 24 mm CSI module. The tilt starts at an assumed 55° above the frame plane. Measure lens height and mounting holes and calibrate the captured field of view for the actual capture mode. |
| Battery and converter | DOGCOM UCELL DC0058 and DeviceMart SZH-PWSDF-036. See the component files for dimensions and photo-based estimates. |

The layout places the Jetson and its slotted carrier above the propeller sweep, with a lower battery platform and landing feet. The converter sits at the rear. The cable paths show the XT60 branch to the ESC and the converter, the converter's 5 V output to J25, four sets of motor wires, the CSI ribbon route, and a signal path to the FC. Cable paths are mechanical routing references. Identify the FC and ESC revisions before selecting actual pin positions or cable pinouts.

The clearance view depicts the 127 mm diameter sweep with a provisional ±4 mm vertical allowance. OpenSCAD rejects overlapping propeller discs and checks vertical separation from the upper plate, converter and Jetson tray. Final fit requires the existing frame, motor and propeller dimensions, mounting locations and cable bend radii. Test the XL4015 at 5 V under a sustained 4 A load and verify installed clearances and balance before using the layout in flight.

[Aikon F43030 specifications](https://aikon-electronics.com/index.php?id=25) · [Aikon AK32 35A specifications](https://aikon-electronics.com/index.php?id=20) · [NVIDIA P3449 B01 drawing](https://developer.nvidia.com/jetson-nano-developer-kit-carrier-board-p3449-b01-specification) · [Waveshare IMX219-160 specifications](https://www.waveshare.com/wiki/IMX219-160_Camera)
