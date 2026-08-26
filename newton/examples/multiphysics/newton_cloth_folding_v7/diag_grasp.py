"""Diagnostic harness: run cloth_folding_coupled --first-grasp-only and print
the tracked patch height, TCP pose, gripper finger targets, and contact counts
throughout the run so we can see *why* the grasp fails.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import warp as wp

# Make the example importable.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import example_cloth_folding_coupled as ex_mod

import newton
import newton.examples


def main() -> None:
    parser = ex_mod.Example.create_parser()
    parser.set_defaults(num_frames=720)
    # Use null viewer, test mode off (we drive the loop ourselves).
    parser.set_defaults(viewer="null", headless=True)
    # Match the README reproduction flags.
    parser.set_defaults(first_grasp_only=True)
    argv = [
        "--first-grasp-only",
        "--num-frames",
        "720",
        "--no-graph-capture",
        "--viewer",
        "null",
        "--no-cloth-self-contact",
    ]
    args = parser.parse_args(argv)
    # Tell newton.examples.init not to start a GL viewer; we bypass it entirely.
    args.test = False
    args.benchmark = False

    # We need a viewer object for the Example; create a null one.
    viewer = newton.viewer.ViewerNull(num_frames=args.num_frames, benchmark=False, benchmark_timeout=None)
    ex = ex_mod.Example(viewer, args)

    # Tables we need from the example.
    p_start = ex.shirt_particle_start
    p_end = ex.shirt_particle_end
    patch_global = np.asarray(ex.first_grasp_particle_ids, dtype=np.int64)
    patch_local = patch_global - p_start
    table_top = float(ex_mod.TABLE_POS[2] + ex_mod.TABLE_HALF_EXTENTS[2])

    # IK target buffer to read current commanded target.
    # Read the actual commanded task target each frame from the GPU buffer.
    ik_targets = ex.ik_target_positions
    finger_buf = ex.finger_pos_buf
    hand_body = None
    for i, lbl in enumerate(ex.model.body_label):
        if str(lbl).endswith("fr3_link7"):
            hand_body = i
            break
    finger_bodies = [i for i, lbl in enumerate(ex.model.body_label) if "finger" in str(lbl)]
    # FK of full Franka from current joint_q for actual TCP.
    fk_state = ex.model.state()

    print(f"[diag] table_top={table_top:.4f} m, patch_local_count={len(patch_local)}")
    print(f"[diag] finger_bodies={[ex.model.body_label[i] for i in finger_bodies]}")
    # Print the keyframe target table.
    print(f"[diag] key_times={ex.key_times.tolist()}")
    print("[diag] targets (xyz, grip):")
    for k, tgt in enumerate(ex.targets):
        print(
            f"[diag]   [{k}] t_end={ex.key_times[k]:.3f}  xyz=({tgt[0]:.4f},{tgt[1]:.4f},{tgt[2]:.4f})  grip={tgt[-1]:.4f}"
        )
    print(
        "[diag] header: frame t(s)  patch_p75_z patch_min_z  ik_tgt_z  actual_tcp_z  actual_tcp_xy  grip  finger_z  n_contacts"
    )

    n_frames = int(args.num_frames)
    log_every = max(1, n_frames // 24)
    for f in range(n_frames):
        ex.step()
        if f % log_every == 0 or f == n_frames - 1:
            pq = ex.state_0.particle_q.numpy()
            patch_z = pq[p_start + patch_local, 2]
            p75 = float(np.percentile(patch_z, 75.0))
            pmin = float(patch_z.min())
            ik_tgt = ik_targets.numpy()[0]
            newton.eval_fk(ex.model, ex.state_0.joint_q, ex.state_0.joint_qd, fk_state)
            bq = fk_state.body_q.numpy()
            hand_q = bq[hand_body]
            hpos = np.array([hand_q[0], hand_q[1], hand_q[2]], dtype=np.float64)
            hrot = wp.quat(float(hand_q[3]), float(hand_q[4]), float(hand_q[5]), float(hand_q[6]))
            tcp = hpos + np.array(wp.quat_rotate(hrot, wp.vec3(0.0, 0.0, 0.22)), dtype=np.float64)
            grip = float(finger_buf.numpy()[0])
            finger_z = [float(bq[i, 2]) for i in finger_bodies]
            try:
                n_con = int(ex.contacts.rigid_contact_count.numpy()[0])
            except Exception:
                n_con = -1
            print(
                f"[diag] {f:4d} {ex.sim_time:6.2f}  p75={p75:.4f} pmin={pmin:.4f}  "
                f"ik_z={float(ik_tgt[2]):.4f}  tcp_z={tcp[2]:.4f}  tcp_xy=({tcp[0]:.3f},{tcp[1]:.3f})  "
                f"grip={grip:.4f}  fz={finger_z}  nc={n_con}"
            )

    # Final report.
    pq = ex.state_0.particle_q.numpy()
    patch_z = pq[p_start + patch_local, 2]
    p75 = float(np.percentile(patch_z, 75.0))
    print(f"\n[diag] FINAL patch_p75_z={p75:.4f} m, table_top={table_top:.4f}, clearance={p75 - table_top:.4f}")

    # Contact detail: are fingers actually in contact with cloth at the end?
    try:
        rc = ex.contacts
        body_a = rc.rigid_contact_body0.numpy()
        # count contacts involving shirt particles
        print(f"[diag] contacts object type: {type(rc).__name__}")
    except Exception as e:
        print(f"[diag] contact introspection failed: {e}")


if __name__ == "__main__":
    main()
