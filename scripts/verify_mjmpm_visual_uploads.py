# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Compare unchanged and cached instance attributes on identical rendered physics states."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import warp as wp

from newton.examples.mjmpm.example_mjmpm_w1_rice_cooker import Example
from newton.examples.mjmpm.support.rice_cooker_visuals import RiceCookerVisuals
from newton.viewer import ViewerGL


def main():
    from PIL import Image

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    opts = parser.parse_args()
    opts.output_dir.mkdir(parents=True, exist_ok=True)
    wp.set_device("cuda:0")
    args = Example.create_parser().parse_args(["--viewer", "null"])
    args.test = True
    viewer = ViewerGL(width=1920, height=1080, headless=True, num_frames=990)
    example = Example(viewer, args)
    batched = example.visuals
    pose = batched.cooker_pose.numpy()[0]
    reference = RiceCookerVisuals(
        viewer,
        device=example.model.device,
        cooker_pose=wp.transform(wp.vec3(*pose[:3]), wp.quat(*pose[3:])),
        rice_count=example.rice_end - example.rice_begin,
        batch_opaque=False,
        prefix="/unbatched_reference",
    )
    results = []
    try:
        for frame in range(516):
            example.step()
            example.test_post_step()
            if frame not in (180, 300, 515):
                continue
            images = []
            timings = {}
            previous = None
            for name, visuals, cache in (
                ("reference", reference, False),
                ("cached", reference, True),
                ("batched", batched, True),
            ):
                other = reference if visuals is batched else batched
                for part, *_ in other.parts:
                    instancer = viewer.objects.get(part + "/instances")
                    if instancer is not None:
                        instancer.hidden = True
                if previous is not visuals:
                    visuals._static_logged = False
                previous = visuals
                example.visuals = visuals
                visuals._reuse_instance_properties = cache
                example.render()
                viewer.renderer.gl.glFinish()
                images.append(viewer.get_frame().numpy().copy())
                Image.fromarray(images[-1]).save(opts.output_dir / f"frame_{frame}_{name}.png")
                samples = []
                for _ in range(10):
                    wp.synchronize_device(example.model.device)
                    start = time.perf_counter()
                    example.render()
                    viewer.renderer.gl.glFinish()
                    wp.synchronize_device(example.model.device)
                    samples.append(1000 * (time.perf_counter() - start))
                timings[name + "_ms"] = float(np.median(samples))
            count = int(np.count_nonzero(images[0] != images[1]))
            batched_count = int(np.count_nonzero(images[0] != images[2]))
            results.append(
                {"frame": frame, "different_components": count, "batched_differences": batched_count, **timings}
            )
            print(results[-1], flush=True)
            if count or batched_count:
                raise AssertionError(f"Rendering changed frame {frame}: {count}, {batched_count} components")
        (opts.output_dir / "report.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    finally:
        viewer.close()


if __name__ == "__main__":
    main()
