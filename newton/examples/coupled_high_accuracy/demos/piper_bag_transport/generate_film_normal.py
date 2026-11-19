"""Generate the seamless OpenGL film normal map and, optionally, a synthetic studio HDRI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

DEFAULT_OUTPUT = Path(__file__).resolve().parent / "textures" / "crumpled_normal.png"


def make_studio_environment(output, width=1024, height=512):
    """Write a portable linear-light RGBE environment with broad softboxes and narrow reflection strips."""
    longitude = ((np.arange(width, dtype=np.float32) + 0.5) / width - 0.5) * (2.0 * np.pi)
    latitude = (0.5 - (np.arange(height, dtype=np.float32) + 0.5) / height) * np.pi
    radiance = np.empty((height, width, 3), dtype=np.float32)
    ambient = 0.06 + 0.06 * np.maximum(np.sin(latitude), 0.0)
    radiance[:] = ambient[:, None, None] * np.array((0.76, 0.86, 1.0), dtype=np.float32)
    for azimuth, elevation, half_width, half_height, intensity, color in (
        (-75.0, 25.0, 18.0, 42.0, 8.0, (1.0, 0.97, 0.92)),
        (35.0, 52.0, 65.0, 22.0, 1.5, (0.90, 0.95, 1.0)),
        (150.0, 22.0, 7.0, 47.0, 5.0, (1.0, 1.0, 1.0)),
    ):
        delta = np.angle(np.exp(1j * (longitude - np.deg2rad(azimuth))))
        x = delta[None, :] / np.deg2rad(half_width)
        y = (latitude[:, None] - np.deg2rad(elevation)) / np.deg2rad(half_height)
        # Smooth rectangular lights leave natural elongated highlights in the crumpled film.
        softbox = np.exp(-np.minimum(x**8 + y**8, 80.0))
        radiance += softbox[..., None] * intensity * np.asarray(color, dtype=np.float32)
    maximum = radiance.max(axis=-1)
    _, exponent = np.frexp(maximum)
    rgbe = np.empty((height, width, 4), dtype=np.uint8)
    rgbe[..., :3] = np.floor(radiance * np.exp2(8.0 - exponent[..., None])).clip(0, 255).astype(np.uint8)
    rgbe[..., 3] = (exponent + 128).astype(np.uint8)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # RGBE permits original uncompressed scanlines as well as run-length encoding.
    header = f"#?RADIANCE\n# Synthetic product studio, linear RGB\nFORMAT=32-bit_rle_rgbe\n\n-Y {height} +X {width}\n"
    output.write_bytes(header.encode("ascii") + rgbe.tobytes())
    return {"studio_hdri": str(output), "size": [width, height], "peak_radiance": float(maximum.max())}


def add_fold(height, rng, tile_size, length, width, amplitude):
    """Add a tapered, slightly bent crease, including all of its periodic images."""
    size = height.shape[0]
    spacing = tile_size / size
    angle = rng.uniform(0.0, 2.0 * np.pi)
    cosine, sine = np.cos(angle), np.sin(angle)
    center = rng.uniform(0.0, tile_size, 2)
    bend = rng.uniform(-0.55, 0.55) * width
    skew = rng.uniform(-0.3, 0.3) * width
    half_length = 0.5 * length
    extent_u = half_length
    extent_v = 4.0 * width + abs(bend) + abs(skew)
    extent_x = abs(cosine) * extent_u + abs(sine) * extent_v
    extent_y = abs(sine) * extent_u + abs(cosine) * extent_v
    ix = np.arange(int(np.floor((center[0] - extent_x) / spacing)), int(np.ceil((center[0] + extent_x) / spacing)))
    iy = np.arange(int(np.floor((center[1] - extent_y) / spacing)), int(np.ceil((center[1] + extent_y) / spacing)))
    x = (ix.astype(np.float32)[None, :] + 0.5) * spacing - center[0]
    y = (iy.astype(np.float32)[:, None] + 0.5) * spacing - center[1]
    u = cosine * x + sine * y
    v = -sine * x + cosine * y
    t = np.clip(u / half_length, -1.0, 1.0)
    taper = np.square(1.0 - np.square(t))
    distance = v - (bend + skew * t) * (1.0 - np.square(t))
    # A rounded cusp gives narrow highlights with broad, almost planar shoulders.
    # Compact smooth support avoids the hard rectangle boundaries of a clipped ridge.
    radius = np.abs(distance) / width
    rounded_radius = np.sqrt(np.square(radius) + 0.035**2) - 0.035
    cutoff = np.square(np.maximum(1.0 - np.square(radius / 4.0), 0.0))
    ridge = amplitude * taper * np.exp(-rounded_radius) * cutoff
    # Wrapped indices make folds crossing a tile edge exactly periodic; add.at also
    # handles the rare wide patch whose periodic copies overlap in the same tile.
    np.add.at(height, (iy[:, None] % size, ix[None, :] % size), ridge)


def make_height(size, tile_size, seed):
    rng = np.random.default_rng(seed)
    height = np.zeros((size, size), dtype=np.float32)
    # Unequal lengths, widths and signed amplitudes prevent a regular cellular or
    # orange-peel pattern. The smallest creases remain sparse and subordinate.
    families = (
        (72, (0.040, 0.140), (0.0020, 0.0070), (0.00016, 0.00055)),
        (155, (0.015, 0.065), (0.00055, 0.0020), (0.000035, 0.00017)),
        (175, (0.006, 0.030), (0.00020, 0.00060), (0.000009, 0.000040)),
    )
    for count, length_range, width_range, amplitude_range in families:
        for _ in range(count):
            length = rng.uniform(*length_range)
            width = np.exp(rng.uniform(*np.log(width_range)))
            amplitude = rng.uniform(*amplitude_range) * rng.choice((-1.0, 1.0))
            add_fold(height, rng, tile_size, length, width, amplitude)
    # A subpixel periodic filter removes sampling spikes without rounding away the
    # crumpled facets. Macro folds continue to come from the simulated bag geometry.
    return gaussian_filter(height, sigma=0.65, mode="wrap")


def normal_image(height, tile_size):
    spacing = tile_size / height.shape[0]
    dx = (np.roll(height, -1, axis=1) - np.roll(height, 1, axis=1)) / (2.0 * spacing)
    dy = (np.roll(height, -1, axis=0) - np.roll(height, 1, axis=0)) / (2.0 * spacing)
    normals = np.stack((-dx, -dy, np.ones_like(height)), axis=-1)
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
    # Array rows increase in +V, whereas PNG rows run from top to bottom. Flip the
    # image, not the green channel: Blender consumes OpenGL (+Y) tangent normals.
    pixels = np.rint((normals[::-1] * 0.5 + 0.5) * 255.0).astype(np.uint8)
    return Image.fromarray(pixels), normals


def report(height, normals, tile_size, output):
    encoded = np.asarray(Image.open(output), dtype=np.float32) / 127.5 - 1.0
    angles = np.degrees(np.arccos(np.clip(normals[..., 2], -1.0, 1.0)))
    step_x = np.abs(np.roll(height, -1, axis=1) - height)
    step_y = np.abs(np.roll(height, -1, axis=0) - height)
    seam_steps = np.concatenate((step_x[:, -1], step_y[-1, :]))
    all_steps = np.concatenate((step_x.ravel(), step_y.ravel()))
    return {
        "output": str(output),
        "size": list(height.shape),
        "tile_size_m": tile_size,
        "normal_convention": "OpenGL +Y",
        "normal_angle_degrees_p50_p95_p99_max": np.percentile(angles, (50, 95, 99, 100)).round(4).tolist(),
        "quantized_normal_length_max_error": float(np.max(np.abs(np.linalg.norm(encoded, axis=-1) - 1.0))),
        "height_seam_step_max_m": float(seam_steps.max()),
        "height_interior_step_max_m": float(all_steps.max()),
        "height_seam_step_rms_m": float(np.sqrt(np.mean(np.square(seam_steps)))),
        "height_interior_step_rms_m": float(np.sqrt(np.mean(np.square(all_steps)))),
        "recommended_strength": 1.0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=2048)
    parser.add_argument("--tile-size", type=float, default=0.18, help="Physical repeat length in meters.")
    parser.add_argument("--seed", type=int, default=7031)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--studio-output", type=Path, help="Also regenerate the optional synthetic studio .hdr environment."
    )
    args = parser.parse_args()
    if args.size < 32 or args.tile_size <= 0.0:
        parser.error("size must be at least 32 and tile-size must be positive")
    height = make_height(args.size, args.tile_size, args.seed)
    image, normals = normal_image(height, args.tile_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output, compress_level=9)
    print(json.dumps(report(height, normals, args.tile_size, args.output), indent=2))
    if args.studio_output is not None:
        print(json.dumps(make_studio_environment(args.studio_output), indent=2))


if __name__ == "__main__":
    main()
