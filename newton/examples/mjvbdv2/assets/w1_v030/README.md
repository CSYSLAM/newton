# W1 V030 with parallel grippers

`w1-030/` is copied from the user-provided
`AssembleW1-030/DexforceW1V030/w1-030/` asset directory. The source URDF identifies
DexForce Technology Co., Ltd., assembly manager V2.0, generated July 14, 2026.
Its original GLB/DAE visuals, OBJ collision meshes, two parallel grippers and
RealSense D405 camera modules are retained. No additional license is asserted
for these user-provided assets.
Mesh contents and URDF joint definitions are unchanged; trailing blank lines
in URDF text are normalized. Meshes are stored with Git LFS.

The demo loads `robot.urdf`. The older `DexforceW1V021_grip` robot from the same
download is not used. `--robot-urdf PATH` can select an equivalent V030 assembly
with the same link/joint names and its corresponding mesh directory.

The robot is a fixed-base, kinematically prescribed moving boundary. IK changes
only the 14 arm joints; both joints of each parallel gripper receive the same
opening, consistent with the source mimic relation. The original position
limits and finger collision meshes are preserved. No block attachment, pose
reset, extra grasp joint or hidden finger pad is used.

The cameras retain the original fixed transforms relative to `left_hand_base`
and `right_hand_base`: translation (0.100, 0, 0.060) m, pitch -0.5236 rad.
For side grasping, gripper +Z points forward and -X points upward, placing the
cameras above the hands without modifying the asset. The approach is horizontal;
release occurs above the bin rim so the horizontal palm clears the wall.

The two 48 × 48 × 90 mm blocks use an illustrative density of 450 kg/m³
(about 93 g each). Their height allows the original palm mesh to clear the
table during side grasping. Friction and contact stiffness are demonstration parameters,
not measured hardware properties. Each bin has a 200 × 240 mm clear interior,
an 18 mm floor and a 130 mm outer wall height, on an 860 mm worktop. Visible
table and bin geometry matches collision geometry.

MJVBDV2 solves the free blocks and their rigid contacts. The prescribed robot
does not react to contact forces; robot self-collision and static/kinematic
contact response are disabled. This is a simulation demonstration, not a
torque-controlled or hardware-qualified robot trajectory.
