# testFPV 5-inch retrofit

Open `assembly.scad` in OpenSCAD. The customizer exposes the assembled, exploded, fastener and propeller-clearance views and the 20×20 and 30.5×30.5 mm stack options. Disable `show_battery_hatch` to see the battery opening, or disable `show_jetson_service_panel` to see the Jetson's connectors with the protective pod installed. `carbon-plate` displays the flat cutting profile. The other part views isolate the printable brackets, trays, enclosure, hatch, panel and skids. Units are millimetres. The nose points along +X.

| Part | Geometry and fit status |
| --- | --- |
| Frame, four motors and 5-inch propellers | A single flat carbon plate includes the four arms and the extended central spine. The reference X frame retains its 220 mm diagonal wheelbase. Enter the existing frame's measured motor centres, plate outline and attachment locations before fabrication. Motor bodies, bolt pattern and propeller geometry are representative. |
| FC and ESC | Aikon F43030 FC with its published 30.5 mm pitch. The lower board represents the matching AK32 layout until the installed ESC variant is identified. The 20 mm stack uses representative boards. |
| Jetson Nano 4 GB | P3449 B01 reference built from the carrier specification drawings. The user's carrier revision is unconfirmed. Hole locations are drawing estimates. NVIDIA's official STEP download requires an NVIDIA sign-in. |
| AI camera | Waveshare IMX219-160, 25 × 24 mm CSI module. The tilt starts at an assumed 55° above the frame plane. Measure lens height and mounting holes and calibrate the captured field of view for the actual capture mode. |
| Battery and converter | DOGCOM UCELL DC0058 and DeviceMart SZH-PWSDF-036. See the component files for dimensions and photo-based estimates. |

The Jetson mounts inverted below the carbon plate inside a removable polymer pod. Raised side vents sit ahead of and behind the fins. The J25 plug sits within the sidewall envelope. The I/O side panel comes off using two outside-facing M3 bolts. With the panel removed, the USB, Ethernet and J25 connectors are accessible while the pod and board remain mounted. Two replaceable skid shoes set the landing plane 18.7 mm below the heatsink tip in the nominal model. The four skid bolts pass through reinforced floor bosses into recessed nuts inside the pod.

The curved upper canopy follows a smooth longitudinal taper. The top hatch covers a nominal 89 × 44 mm opening over the pack and its straps, with a rear pocket for the XT60. Release the thumbscrew, lift the rear edge and slide the hatch rearward to disengage the front tabs. Disconnect and tuck the pack leads toward the pack, release the straps and lift the pack vertically. The converter sits behind the stack. Fasteners remain separate hardware.

Cable paths show the XT60 branch to the ESC and converter, the converter's 5 V branch to J25, motor wiring, the CSI ribbon and the FC signal path. They serve as mechanical routing references. Identify the FC and ESC revisions before selecting actual pins or cable pinouts.

## Mechanical joints

| Interface | Fastening in the model |
| --- | --- |
| Carbon plate to canopy, Nano carrier and pod | Four shared M3 through-bolts into recessed hex nuts in the pod. The battery shelf has tool access holes. Remove the hatch and pack to reach all four heads from above. |
| Flight stack to carbon | 30.5 mm pattern with M3 bolts and 20 mm pattern with M2 bolts, through the carbon and adapter. The two additional M3 holes retain the adapter. Verify the actual ESC hole sizes and grommets. |
| Battery shelf to carbon | Four M3 through-bolts, insulating spacers and recessed nuts below the battery contact plane. The straps loop through separate slots. |
| XL4015 to carbon | Four M3 through-bolts share centres with the PCB and isolating tray. The Nano carrier has relief around the bolt heads. Check solder tails and exposed copper against the carbon. |
| Camera to carbon | Four M3 through-bolts at the bracket feet. Four M2 through-bolts mate the PCB to the bracket, with nut access from the rear. Camera hole centres are provisional. |
| Nano board to carrier | Four M3 through-bolts with slotted captive-nut seats. The carrier design uses the provisional B01 hole coordinates. |
| Skids and battery hatch | M3 bolts with recessed nuts. The hatch has a hand-operated screw and two front locating tabs. |
| Jetson I/O panel | Two side-entry M3 bolts into hex nuts inside the fixed pod. Fit the nuts with the pod off the frame. |
| Motors and propellers | The motor base screws thread into the motors. The propellers mount on modelled M5 shafts with hex locknuts. Verify the installed motors' pattern, shaft and safe screw engagement depth. |

## Assembly and service

1. Fit the camera bracket, ESC and FC stack, XL4015 and battery cradle to the carbon plate while the underside is open. Route the camera ribbon through the rounded slot. Leave the battery out for tool access.
2. Install the recessed nuts in the Nano carrier, bolt the Nano to it and connect CSI and the power lead. Assemble the skid shoes and install the pod's recessed nuts before closing the pod.
3. Locate the carrier and pod beneath the plate. Install the canopy and insert its four shared bolts from above through the access holes in the empty battery cradle. Fit the side panel from the outside, then install the battery and hatch.
4. For bench access, remove only the two panel bolts and lift off the side panel. The board, pod and main fasteners stay installed.

The clearance view depicts the 127 mm diameter propeller sweep with a provisional ±4 mm vertical allowance. The canopy cutouts use an additional 3 mm radial and 1 mm vertical allowance. The model checks propeller disc overlap, converter height, the battery floor above the stack and the floor clearance below the heatsink. Before flight, verify hatch and skid retention, Jetson airflow and temperature under inference load, pad and pod deflection, loaded centre of gravity, and XL4015 voltage and temperature at a sustained 4 A load. Final fit depends on the measured frame, connector routing, fastener locations and camera field of view. Aerodynamic drag and crash loads have not been measured.

[Aikon F43030 specifications](https://aikon-electronics.com/index.php?id=25) · [Aikon AK32 35A specifications](https://aikon-electronics.com/index.php?id=20) · [NVIDIA P3449 B01 drawing](https://developer.nvidia.com/jetson-nano-developer-kit-carrier-board-p3449-b01-specification) · [Waveshare IMX219-160 specifications](https://www.waveshare.com/wiki/IMX219-160_Camera)
