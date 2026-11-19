"""Bounded joint targets; payloads move exclusively through simulated contact."""

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "demos/piper_bag_transport"))
from geometry import BAG_SIZE, GRASP_OFFSET
from piper_motion import PiperKinematics, loading_drop_rotation

GRIP_LOCAL = (0, 0, 0.105)
# Side clamping needs the loose film inside the pad faces rather than beyond the tapered fingertips.
GRIP_POINTS = (GRIP_LOCAL, GRIP_LOCAL)
# Leave lateral room between the envelopes during shaking and free fall.
GRASP_POINTS = np.array([(-0.21, -0.30, 0.030), (0.21, -0.30, 0.030)])
CLOSING_OPENINGS = (0.0, 0.0)
ROTATIONS = [np.eye(3), np.eye(3)]
# The native fingertips stop outside the payload: only the loose film enters the jaws.
END_GRASP_OFFSET = GRASP_OFFSET
CENTERS = [p + np.array((0, END_GRASP_OFFSET, BAG_SIZE[2] / 2 + 0.003 - p[2])) for p in GRASP_POINTS]
PARCEL_STARTS = (0.0, 15.5)
LIFT_START = 4.0
SHAKE_START, SHAKE_END = 7.5, 10.5
DROP_RELEASE_START, DROP_RELEASE_END = 10.5, 10.65
RELEASE_OPENING = 0.035
SHAKE_CYCLES = 3
SHAKE_DIRECTION = (0.0, 1.0, 0.0)
SHAKE_AMPLITUDE = 0.040
# Keep the full flick reachable at both ends of the horizontal stroke under native wrist limits.
SHAKE_PITCH_CENTER = 25.0
SHAKE_PITCH_AMPLITUDE = 15.0
LIFT_HEIGHT = 0.310
RETRACT_START, RETRACT_CLEAR, RETRACT_END = 10.5, 10.7, 13.3
INITIAL_TILT = 40.0
# The native slide axis is wrist-local Y. Roll before approach so the pads close along world X, not vertically.
JAW_FRAMES = (Rotation.from_euler("z", -90, degrees=True).as_matrix(),) * 2


def smoothstep(t):
    t = np.clip(t, 0, 1)
    return t**3 * (10 + t * (-15 + 6 * t))


def shake_state(time, grasp):
    """Return a smooth airborne shake that starts and ends at the same carry pose."""
    # Allow the full 280 mm envelope to hang clear of the floor while keeping the wrist in its reachable workspace.
    carry = grasp + np.array((0.0, 0.185, LIFT_HEIGHT))
    u = np.clip((time - SHAKE_START) / (SHAKE_END - SHAKE_START), 0.0, 1.0)
    envelope = smoothstep(u / 0.15) * smoothstep((1.0 - u) / 0.15)
    phase = 2.0 * np.pi * SHAKE_CYCLES * u
    offset = np.asarray(SHAKE_DIRECTION) * SHAKE_AMPLITUDE * np.sin(phase) * envelope
    # Pitch about world X keeps the jaws sideways; the forward stroke also flicks the fingertips upward.
    tilt = SHAKE_PITCH_CENTER - SHAKE_PITCH_AMPLITUDE * np.sin(phase) * envelope
    return carry + offset, np.deg2rad(tilt)


def cartesian_plan(fps=60, duration=30, second_offset=None):
    """Lift each parcel, shake it three times in the air, and release it for a free drop."""
    keys = []
    for parcel, start in enumerate(PARCEL_STARTS):
        grasp = GRASP_POINTS[parcel].copy()
        if parcel == 1 and second_offset is not None:
            grasp += second_offset
        carry, carry_tilt = shake_state(SHAKE_START, grasp)
        release_point, release_tilt = shake_state(DROP_RELEASE_START, grasp)
        # Extract the opened fingers promptly so loose film cannot remain draped over their tapered tips.
        withdrawal = np.array((0.0, -0.100, 0.0))
        for t, p, tilt, opening, yaw_scale in (
            (0, grasp + (0, 0, 0.13), 40, 0.035, 0),
            (1.5, grasp + (0, 0, 0.06), 40, 0.035, 0),
            (3, grasp, INITIAL_TILT, 0.035, 0),
            (LIFT_START, grasp, INITIAL_TILT, CLOSING_OPENINGS[parcel], 0),
            (7.1, carry, np.rad2deg(carry_tilt), 0, 0),
            (SHAKE_START, carry, np.rad2deg(carry_tilt), 0, 0),
            (DROP_RELEASE_START, release_point, np.rad2deg(release_tilt), 0, 0),
            (DROP_RELEASE_END, release_point, np.rad2deg(release_tilt), RELEASE_OPENING, 0),
            (RETRACT_CLEAR, release_point + withdrawal, np.rad2deg(release_tilt), RELEASE_OPENING, 0),
            (RETRACT_END, release_point + withdrawal, np.rad2deg(release_tilt), RELEASE_OPENING, 0),
        ):
            keys.append((start + t, np.asarray(p), np.deg2rad(tilt), opening, yaw_scale))
    if keys[-1][0] < duration:
        keys.append((duration, *keys[-1][1:]))
    times = np.arange(round(duration * fps) + 1, dtype=float) / fps
    positions, rotations, openings = [], [], []
    key_times = [k[0] for k in keys]
    for time in times:
        i = min(max(np.searchsorted(key_times, time, side="right") - 1, 0), len(keys) - 2)
        a, b = keys[i], keys[i + 1]
        blend = smoothstep((time - a[0]) / (b[0] - a[0]))
        p = a[1] + blend * (b[1] - a[1])
        tilt, opening, yaw_scale = np.array(a[2:]) + blend * (np.array(b[2:]) - a[2:])
        for start in PARCEL_STARTS:
            local_time = time - start
            if DROP_RELEASE_START <= local_time <= RETRACT_CLEAR:
                # Withdraw independently of the opening ramp so falling film clears both opened pads promptly.
                parcel = int(time >= PARCEL_STARTS[1])
                grasp = GRASP_POINTS[parcel].copy()
                if parcel == 1 and second_offset is not None:
                    grasp += second_offset
                release_point = shake_state(DROP_RELEASE_START, grasp)[0]
                p = (
                    release_point
                    + smoothstep((local_time - RETRACT_START) / (RETRACT_CLEAR - RETRACT_START)) * withdrawal
                )
        for parcel, start in enumerate(PARCEL_STARTS):
            local_time = time - start
            if SHAKE_START <= local_time <= DROP_RELEASE_START:
                grasp = GRASP_POINTS[parcel].copy()
                if parcel == 1 and second_offset is not None:
                    grasp += second_offset
                p, tilt = shake_state(local_time, grasp)
                yaw_scale = 0.0
        yaw = -np.arctan2(p[0], p[1] + 0.52) * yaw_scale
        parcel = int(time >= PARCEL_STARTS[1])
        positions.append(p)
        rotations.append(loading_drop_rotation(tilt, yaw) @ JAW_FRAMES[parcel])
        openings.append(opening)
    return np.asarray(positions), np.asarray(rotations), np.asarray(openings)


def create_plan(fps=60, duration=30, second_offset=None):
    arm = PiperKinematics()
    positions, rotations, openings = cartesian_plan(fps, duration, second_offset)
    poses, seed, pitch = [], None, np.deg2rad(INITIAL_TILT)
    carry_pitch = None
    transfer = (round(RETRACT_END * fps), round(PARCEL_STARTS[1] * fps))
    for frame, (p, rotation, opening) in enumerate(zip(positions, rotations, openings)):
        time = frame / fps
        if transfer[0] < frame < transfer[1]:
            poses.append(None)
            continue
        if frame == transfer[1]:
            seed, pitch = None, np.deg2rad(INITIAL_TILT)
            carry_pitch = None
        parcel = int(time >= PARCEL_STARTS[1])
        local_time = time - PARCEL_STARTS[parcel]
        grip_local = GRIP_POINTS[parcel]
        try:
            if 0 < local_time <= RETRACT_END:
                previous = seed[:6]
                target_pitch = np.arctan2(-rotation[2, 2], rotation[1, 2])
                pitch_weight = 0.0
                if local_time >= 6.5:
                    # Blend the angle from the reachable lift pose, not just its residual weight: even a small
                    # weight can abruptly override the otherwise free pitch and create a joint-velocity spike.
                    if carry_pitch is None:
                        carry_pitch = pitch
                    blend = smoothstep((local_time - 6.5) / 0.6)
                    target_pitch = carry_pitch + blend * (target_pitch - carry_pitch)
                    pitch_weight = 0.15

                def residual(x):
                    pose = arm.fk(x[:6], grip_local)
                    target = loading_drop_rotation(x[6], 0) @ JAW_FRAMES[parcel]
                    error = Rotation.from_matrix(target.T @ pose[:3, :3]).as_rotvec()
                    return np.r_[
                        pose[:3, 3] - p,
                        0.15 * error,
                        1e-5 * (x[:6] - previous),
                        1e-5 * (x[6] - np.deg2rad(INITIAL_TILT)),
                        pitch_weight * (x[6] - target_pitch),
                    ]

                fit = least_squares(
                    residual,
                    np.r_[previous, pitch],
                    bounds=(np.r_[arm.limits[:6, 0], np.deg2rad(5)], np.r_[arm.limits[:6, 1], np.deg2rad(60)]),
                    max_nfev=120,
                    ftol=1e-10,
                    xtol=1e-10,
                    gtol=1e-10,
                )
                error = residual(fit.x)
                if np.linalg.norm(error[:3]) > 2e-4 or np.linalg.norm(error[3:6]) > 3e-4:
                    raise ValueError("No continuous pitch solution within the native joint limits")
                seed, pitch = np.r_[fit.x[:6], opening, -opening], fit.x[6]
            elif local_time > RETRACT_END:
                seed = seed.copy()
            else:
                seed = arm.solve(p, rotation, seed, opening=opening, grip_point_local=grip_local)
        except ValueError as error:
            raise ValueError(f"Unreachable trajectory at {frame / fps:.3f}s, point {p}: {error}") from error
        poses.append(seed)
    if transfer[1] < len(poses):
        # Transfer in joint space only after the open fingers have withdrawn and cleared the parcel vertically.
        for frame in range(transfer[0] + 1, transfer[1]):
            blend = smoothstep((frame - transfer[0]) / (transfer[1] - transfer[0]))
            poses[frame] = poses[transfer[0]] + blend * (poses[transfer[1]] - poses[transfer[0]])
    else:
        for frame in range(transfer[0] + 1, len(poses)):
            poses[frame] = poses[transfer[0]].copy()
    return np.asarray(poses, dtype=np.float32)
