"""Record motor commands and post-step actuator efforts without changing the simulation state."""

from __future__ import annotations

import csv
import json
import weakref
from pathlib import Path

import numpy as np
import warp as wp

COLUMNS = (
    "frame",
    "time_s",
    "joint_name",
    "joint_type",
    "position_unit",
    "effort_unit",
    "target_time_s",
    "target_position",
    "target_velocity",
    "feedforward_effort",
    "feedback_effort",
    "feedback_effort_mean",
    "feedback_effort_min",
    "feedback_effort_max",
    "feedback_valid",
)


@wp.kernel
def capture_motor_sample(
    substep: int,
    dt: float,
    motion_time: wp.array[float],
    target_indices: wp.array[int],
    dof_indices: wp.array[int],
    target_position: wp.array[float],
    target_velocity: wp.array[float],
    feedforward_effort: wp.array[float],
    feedback_effort: wp.array[float],
    samples: wp.array3d[float],
):
    motor = wp.tid()
    dof = dof_indices[motor]
    samples[substep, motor, 0] = target_position[target_indices[motor]]
    samples[substep, motor, 1] = target_velocity[dof]
    samples[substep, motor, 2] = feedforward_effort[dof]
    samples[substep, motor, 3] = feedback_effort[dof]
    # The command kernel advances this clock after evaluating the target for the current substep.
    samples[substep, motor, 4] = motion_time[0] - dt


class MotorTelemetryCSV:
    """Stream one row per motor per frame; unsolved initial feedback remains explicitly missing."""

    def __init__(self, path, joints, fps, substeps):
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.joints = joints
        self.fps = float(fps)
        self.substeps = int(substeps)
        self.frame_count = 0
        self.complete = False
        with self.path.open("w", newline="", encoding="utf-8") as output:
            csv.writer(output).writerow(COLUMNS)
        self.write_metadata()

    def write_metadata(self):
        metadata = {
            "schema_version": 1,
            "csv": self.path.name,
            "fps": self.fps,
            "physics_substeps_per_frame": self.substeps,
            "frame_count": self.frame_count if self.complete else None,
            "complete": self.complete,
            "rows_per_frame": len(self.joints),
            "joints": self.joints,
            "input": "Actual position/velocity targets and feedforward effort used by the last physics substep.",
            "feedback_sampling": "post_step",
            "feedback_model": "limited_pd_plus_feedforward",
            "feedback": (
                "Post-step actuator effort evaluated from each completed physics substep's joint position and velocity: "
                "feedforward plus the active PD drive response, with the motor effort limit applied. "
                "Positive means the joint's positive axis. This is a simulated actuator output computed separately "
                "from the nonlinear solver RHS; it is not a hardware measurement, contact/constraint reaction, "
                "or inverse-dynamics torque."
            ),
            "frame_sampling": (
                "time_s is the completed frame boundary; target_time_s is the motor command's actual float32 clock. "
                "feedback_effort is the final substep's post-step effort; mean/min/max cover the post-step effort "
                "of every substep in that frame."
            ),
            "initial_sample": "Frame 0 contains initialized commands; feedback is blank with feedback_valid=0.",
        }
        self.path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    def append(self, frame, samples, *, initial=False):
        if self.complete:
            raise ValueError("Cannot append frames to finalized motor telemetry")
        samples = np.asarray(samples, dtype=np.float64)
        expected_substeps = 1 if initial else self.substeps
        if samples.shape != (expected_substeps, len(self.joints), 5) or not np.isfinite(samples).all():
            raise ValueError("Motor telemetry must contain finite command/effort samples for every motor and substep")
        if frame != self.frame_count or initial != (frame == 0):
            raise ValueError("Motor telemetry frames must be contiguous from the unsolved initial frame")
        rows = []
        for motor, joint in enumerate(self.joints):
            command = samples[-1, motor]
            efforts = samples[:, motor, 3]
            feedback = (
                ("", "", "", "")
                if initial
                else (
                    float(efforts[-1]),
                    float(efforts.mean()),
                    float(efforts.min()),
                    float(efforts.max()),
                )
            )
            rows.append(
                (
                    frame,
                    frame / self.fps,
                    joint["name"],
                    joint["type"],
                    joint["position_unit"],
                    joint["effort_unit"],
                    float(command[4]),
                    *map(float, command[:3]),
                    *feedback,
                    int(not initial),
                )
            )
        # Close after each frame so a stopped interactive run still leaves a complete, immediately readable CSV.
        with self.path.open("a", newline="", encoding="utf-8") as output:
            csv.writer(output).writerows(rows)
        self.frame_count += 1

    def finish(self):
        self.complete = True
        self.write_metadata()


class MotorTelemetry:
    """Preallocate graph-compatible sample storage, then transfer only the small motor table once per frame."""

    def __init__(self, example, robot_joints, path):
        import newton

        self.example = weakref.proxy(example)
        model = example.model
        joint_types = model.joint_type.numpy()
        dofs = model.joint_qd_start.numpy()[robot_joints]
        self.dofs = wp.array(dofs, dtype=wp.int32, device=model.device)
        self.samples = wp.empty((example.sim_substeps, len(robot_joints), 5), dtype=wp.float32, device=model.device)
        joints = []
        for joint, dof in zip(robot_joints, dofs, strict=True):
            kind = int(joint_types[joint])
            if kind not in (int(newton.JointType.REVOLUTE), int(newton.JointType.PRISMATIC)):
                raise ValueError("The Piper telemetry expects single-axis revolute or prismatic motors")
            angular = kind == int(newton.JointType.REVOLUTE)
            joints.append(
                {
                    "name": model.joint_label[joint].split("/")[-1],
                    "joint_index": int(joint),
                    "dof_index": int(dof),
                    "type": "revolute" if angular else "prismatic",
                    "position_unit": "rad" if angular else "m",
                    "effort_unit": "Nm" if angular else "N",
                }
            )
        self.writer = MotorTelemetryCSV(path, joints, 1.0 / example.frame_dt, example.sim_substeps)
        # Enable before CUDA graph capture: feedback storage must never be allocated during replay.
        example.solver.enable_joint_drive_feedback()
        initial = np.zeros((1, len(robot_joints), 5), dtype=np.float32)
        initial[0, :, 0] = example.control.joint_target_q.numpy()[example.motor_targets.numpy()]
        initial[0, :, 1] = example.control.joint_target_qd.numpy()[dofs]
        initial[0, :, 2] = example.control.joint_f.numpy()[dofs]
        self.writer.append(0, initial, initial=True)

    def capture_substep(self, substep):
        """Capture the independent actuator output after solver.step completes the substep."""
        example = self.example
        wp.launch(
            capture_motor_sample,
            dim=len(self.dofs),
            inputs=[
                substep,
                example.sim_dt,
                example.motion_time,
                example.motor_targets,
                self.dofs,
                example.control.joint_target_q,
                example.control.joint_target_qd,
                example.control.joint_f,
                example.solver.joint_drive_effort,
            ],
            outputs=[self.samples],
            device=example.model.device,
        )

    def record_frame(self):
        self.writer.append(self.writer.frame_count, self.samples.numpy())

    def finish(self):
        self.writer.finish()
