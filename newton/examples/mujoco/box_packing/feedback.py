# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Retarget robot-only commands to the observed free carton pose during packing."""

import mujoco
import numpy as np

from .closure import closure_metrics
from .ik import solve_arm
from .slots import aligned_ear_angle
from .validation import rear_support_contact

FRONT_PRESS_ANGLE = 1.90
EAR_OVERFOLD_MARGIN = 0.08
EAR_PRESS_ANGLE = 1.70
EAR_FOLD_READY_ANGLE = 1.65
EAR_REBOUND_ANGLE = 1.35


def table_frame(data, height):
    """Track planar carton motion without commanding the hands to follow a tip or lift."""
    box = data.body("carton")
    yaw = np.arctan2(box.xmat[3], box.xmat[0])
    rotation = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    return np.array([box.xpos[0], box.xpos[1], height]), rotation


class CartonFrameController:
    """Keep contact targets attached to the carton frame, not an obsolete table point."""

    def __init__(self, model, replay):
        """Load robot contact targets and the reference carton coordinate frame."""
        self.model = model
        with np.load(replay) as source:
            self.replay_controls = source["ctrl"].copy()
            self.replay_times = source["times"].copy()
            self.replay_targets = source["target_qpos"].copy()
            self.plan = {
                name: source[name].copy()
                for name in (
                    "target_times",
                    "hand_position",
                    "hand_forward",
                    "hand_up",
                    "hand_opening",
                    "task_frame_position",
                    "task_frame_yaw",
                )
            }
        yaw = float(self.plan["task_frame_yaw"])
        self.reference_rotation = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        self.robot_addresses = [model.jnt_qposadr[model.actuator_trnid[i, 0]] for i in range(model.nu)]
        self.applied_times = []
        self.applied_controls = []
        self.previous_target = None
        self.support_origin = None
        self.support_rotation = None
        self.support_wait = 0.0
        self.support_inset = 0.0
        self.support_ready = False
        self.support_stage_wait = 0.0
        self.ear_stage_wait = 0.0
        self.ears_ready = False
        self.checked_ears = set()
        self.insertion_wait = 0.0
        self.insertion_ready = False
        self.front_press_inset = 0.0
        self.release_advance = 0.0
        self.support_height_offset = 0.0
        self.prefix_offset = np.zeros(model.nu)
        self.reference_data = mujoco.MjData(model)

    def update(self, data, time):
        """Compute robot actuator targets from the current simulated observations."""
        if time < 12:
            index = min(np.searchsorted(self.replay_times, time), len(self.replay_times) - 1)
            target_index = min(np.searchsorted(self.plan["target_times"], time), len(self.replay_targets) - 1)
            reference = self.replay_targets[target_index].copy()
            self.reference_data.qpos[:] = reference
            mujoco.mj_forward(self.model, self.reference_data)
            mujoco.mj_forward(self.model, data)
            origin, box_rotation = table_frame(data, self.plan["task_frame_position"][2])
            rotation = box_rotation @ self.reference_rotation.T
            target = reference.copy() if self.previous_target is None else self.previous_target.copy()
            target[16:] = data.qpos[16:]
            for side in ("left", "right"):
                # Retarget the reachable offline IK pose, not an idealized hand pose
                # that may demand an unreachable wrist orientation.
                site = self.reference_data.site(f"{side}/contact")
                position = origin + rotation @ (site.xpos - self.plan["task_frame_position"])
                frame = rotation @ site.xmat.reshape(3, 3)
                target = solve_arm(
                    self.model,
                    target,
                    side,
                    position,
                    frame[:, 0],
                    frame[:, 2],
                    max_joint_step=0.12,
                    allow_partial=True,
                )
            self.prefix_offset = target[self.robot_addresses] - reference[self.robot_addresses]
            self.prefix_offset[[6, 13]] = 0
            control = self.replay_controls[index] + self.prefix_offset
            self.previous_target = target.copy()
            self.applied_times.append(time)
            self.applied_controls.append(control.copy())
            return control
        mujoco.mj_forward(self.model, data)
        simulation_time = time
        time -= self.support_wait
        if 18.2 <= time < 19.2 and not self.ears_ready:
            angles = [
                direction * float(data.qpos[self.model.joint(f"crease/{side}_ear").qposadr[0]])
                for side, direction in (("left", 1), ("right", -1))
            ]
            if min(angles) < EAR_FOLD_READY_ANGLE:
                self.support_wait += 0.04
                self.ear_stage_wait += 0.04
                time = 18.2
                if self.ear_stage_wait > 3.0:
                    target = np.degrees(EAR_FOLD_READY_ANGLE)
                    raise RuntimeError(
                        f"locking-ear fold did not reach {target:.1f} degrees; measured {np.degrees(angles)}"
                    )
            else:
                self.ears_ready = True
        for side, deadline, direction in (("left", 20.4, 1), ("right", 20.4, -1)):
            if time >= deadline and side not in self.checked_ears:
                angle = direction * float(data.qpos[self.model.joint(f"crease/{side}_ear").qposadr[0]])
                if angle < EAR_REBOUND_ANGLE:
                    raise RuntimeError(
                        f"{side} locking ear rebounded to {np.degrees(angle):.1f} degrees; insertion cancelled"
                    )
                self.checked_ears.add(side)
        support_start = 22.6
        if support_start <= time < support_start + 1 and not self.support_ready:
            self.support_ready = rear_support_contact(self.model, data)
            if not self.support_ready:
                self.support_wait += 0.04
                self.support_stage_wait += 0.04
                self.support_inset = min(0.004, self.support_inset + 0.00008)
                time = support_start
                if self.support_stage_wait > 2.0:
                    raise RuntimeError("rear support did not establish contact within 2 seconds; front press cancelled")
        insertion_check_start = 28.8
        release_plan_start = 31.2
        if self.insertion_ready:
            time += self.release_advance
        elif insertion_check_start <= time < insertion_check_start + 1:
            metrics = closure_metrics(self.model, data)
            self.insertion_ready = all(metrics[side]["inserted"] for side in ("left", "right"))
            if self.insertion_ready:
                self.release_advance = release_plan_start - time
                time = release_plan_start
            else:
                self.support_wait += 0.04
                self.insertion_wait += 0.04
                self.front_press_inset = min(0.012, self.front_press_inset + 0.0002)
                if not rear_support_contact(self.model, data):
                    self.support_inset = min(0.008, self.support_inset + 0.0001)
                time = insertion_check_start
                if self.insertion_wait > 3.0:
                    depths = [1000.0 * metrics[side]["depth_m"] for side in ("left", "right")]
                    angles = [
                        float(data.qpos[self.model.joint(name).qposadr[0]])
                        for name in ("crease/front", "crease/left_ear", "crease/right_ear")
                    ]
                    raise RuntimeError(
                        f"locking ears did not enter both channels; depths are {depths} mm, "
                        f"front/left/right angles are {np.degrees(angles)} degrees"
                    )
        times = self.plan["target_times"]
        index = int(np.clip(np.searchsorted(times, time) - 1, 0, len(times) - 2))
        u = float(np.clip((time - times[index]) / (times[index + 1] - times[index]), 0, 1))
        sample = {
            name: (1 - u) * self.plan[name][index] + u * self.plan[name][index + 1]
            for name in ("hand_position", "hand_forward", "hand_up", "hand_opening")
        }
        origin, box_rotation = table_frame(data, self.plan["task_frame_position"][2])
        if time >= 20 and self.support_origin is None:
            self.support_origin = origin.copy()
            self.support_rotation = box_rotation.copy()
        rotation = box_rotation @ self.reference_rotation.T
        qpos = data.qpos.copy()
        # Keep the IK branch continuous. Seeding each target from the deflected
        # physical arm made contact disturbances appear as commanded velocities,
        # which the damping feedforward then amplified into large pushes.
        if self.previous_target is not None:
            qpos[:16] = self.previous_target[:16]
        for arm, side in enumerate(("left", "right")):
            position = origin + rotation @ (sample["hand_position"][arm] - self.plan["task_frame_position"])
            forward = rotation @ sample["hand_forward"][arm]
            up = rotation @ sample["hand_up"][arm]
            if side == "right" and 20 <= time <= 32.2:
                # The supporting hand holds its world pose, rather than following the
                # moving carton. Only real rear-wall contact can oppose the front push.
                support_transform = self.support_rotation @ self.reference_rotation.T
                position = self.support_origin + support_transform @ (
                    sample["hand_position"][arm] - self.plan["task_frame_position"]
                )
                forward = support_transform @ sample["hand_forward"][arm]
                up = support_transform @ sample["hand_up"][arm]
                if 22.6 <= time <= 31.2:
                    position += self.support_inset * forward / np.linalg.norm(forward)
                    # Keep the support low. Following the moving apron contact height
                    # raised this target to about 82 mm at the start of the press and
                    # let the rear fingertip lift the carton. The retained rollout's
                    # rear-wall contact is approximately 55 mm above the table.
                    supported_height = self.support_origin[2] + 0.052
                    self.support_height_offset = supported_height - position[2]
                    position[2] = supported_height
                elif 31.2 < time <= 32.2:
                    fraction = np.clip((time - 31.2) / 1.0, 0, 1)
                    blend = fraction**2 * (3 - 2 * fraction)
                    # Removing the contact-height correction in one update lifted the
                    # finger by several centimetres before it cleared the rear rim.
                    position[2] += (1 - blend) * self.support_height_offset
            if side == "left" and 23.2 <= time <= 31.2:
                sign = -1 if side == "left" else 1
                lid = data.qpos[self.model.joint("crease/lid").qposadr[0]]
                if time <= 25.2:
                    progress = np.clip((time - 23.2) / 2.0, 0, 1)
                    planned_front = 0.33 + 0.47 * progress**2 * (3 - 2 * progress)
                elif time <= 28.2:
                    progress = np.clip((time - 25.2) / 3.0, 0, 1)
                    planned_front = 0.8 + (FRONT_PRESS_ANGLE - 0.8) * progress**2 * (3 - 2 * progress)
                else:
                    planned_front = FRONT_PRESS_ANGLE
                # The target must advance through the flap. Tracking only the measured
                # angle plus a fixed offset stalls at the first contact equilibrium.
                front = min(
                    max(float(data.qpos[self.model.joint("crease/front").qposadr[0]]) + 0.10, planned_front),
                    FRONT_PRESS_ANGLE,
                )
                total = lid + front
                f = np.array((-sign * 0.4, -np.cos(total), -np.sin(total)))
                f /= np.linalg.norm(f)
                p = np.array(
                    (
                        0.0,
                        0.0784 - 0.1583 * np.sin(lid) - 0.055 * np.sin(total),
                        0.104 + 0.1583 * np.cos(lid) + 0.055 * np.cos(total),
                    )
                )
                position = origin + box_rotation @ (p - (0.004 - self.front_press_inset) * f)
                # Keep the pad normal to the moving apron. A fixed downward wrist
                # orientation only maintained contact at the first angle and produced
                # almost no hinge torque as the target advanced.
                forward = box_rotation @ f
                up = box_rotation @ np.array((0, np.sin(total), -np.cos(total)))
                sample["hand_opening"][arm] = 0.002
            ear_start = 15.2
            if ear_start <= time <= 20.0:
                sign = -1 if side == "left" else 1
                lid = float(data.qpos[self.model.joint("crease/lid").qposadr[0]])
                front = float(data.qpos[self.model.joint("crease/front").qposadr[0]])
                total = lid + front
                fraction = np.clip((time - ear_start) / 3.0, 0, 1)
                # Elastic bending can move the geometric tip target while it is being
                # pressed. Do not chase that motion into an ever larger plastic fold.
                overfold = min(aligned_ear_angle(self.model, data, side) + EAR_OVERFOLD_MARGIN, EAR_PRESS_ANGLE)
                scheduled_angle = 0.12 + (overfold - 0.12) * fraction**2 * (3 - 2 * fraction)
                angle = scheduled_angle
                f = np.array((-sign * np.sin(angle), -np.cos(angle) * np.cos(total), -np.cos(angle) * np.sin(total)))
                p = np.array(
                    (
                        sign * (0.11265 + 0.022 * np.cos(angle)),
                        0.0784 - 0.1583 * np.sin(lid) - 0.030 * np.sin(total) - 0.022 * np.sin(angle) * np.cos(total),
                        0.104 + 0.1583 * np.cos(lid) + 0.030 * np.cos(total) - 0.022 * np.sin(angle) * np.sin(total),
                    )
                )
                clearance = 0.004 + 0.056 * np.clip((time - ear_start - 4.0) / 0.8, 0, 1)
                position = origin + box_rotation @ (p - clearance * f)
                forward = box_rotation @ np.array((-sign, 0, -0.1))
                up = box_rotation @ np.array((0, 0, 1))
            qpos = solve_arm(self.model, qpos, side, position, forward, up, max_joint_step=0.12, allow_partial=True)
        control = qpos[self.robot_addresses].copy()
        for actuator in (*range(6), *range(7, 13)):
            joint = self.model.actuator_trnid[actuator, 0]
            dof = self.model.jnt_dofadr[joint]
            address = self.model.jnt_qposadr[joint]
            velocity = (
                0.0
                if self.previous_target is None
                else np.clip((qpos[address] - self.previous_target[address]) / 0.04, -3, 3)
            )
            control[actuator] += (
                data.qfrc_bias[dof] + self.model.dof_damping[dof] * velocity
            ) / self.model.actuator_gainprm[actuator, 0]
            control[actuator] = np.clip(control[actuator], *self.model.actuator_ctrlrange[actuator])
        control[[6, 13]] = sample["hand_opening"]
        self.applied_times.append(simulation_time)
        self.applied_controls.append(control.copy())
        self.previous_target = qpos.copy()
        return control
