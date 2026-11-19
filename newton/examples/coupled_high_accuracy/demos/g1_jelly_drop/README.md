
# G1 Jelly Drop

Drop a Unitree G1 onto a stack of deformable jelly cubes, export the simulation as USD,
and build an editable Blender Cycles project for offline rendering.

The main entry is [`../demo/demo_g1_jelly_drop.py`](../demo/demo_g1_jelly_drop.py).
This directory contains scene dependencies, render configuration, documentation, and the demo preview image.
All locally generated artifacts go into `outputs/`, which is excluded by the local `.gitignore`. Blender projects,
USD caches, videos, and original frames remain available for further editing.
Shared rendering, encoding, and USD import validation tools live in [`../utils/`](../utils/README.md) and accept
project and input/output paths as arguments.

```text
g1_jelly_drop/
|-- .gitignore
|-- README.md
|-- g1_jelly_drop.png
|-- render_config.json
|-- g1_robot.py
|-- jelly_stack.py
|-- export_simulation.py
|-- build_render_scene.py
|-- validate_simulation.py
`-- outputs/                    # Generated locally; do not commit
    |-- simulation.usdc
    |-- simulation_states.npz
    |-- simulation_metadata.json
    |-- g1_jelly_drop.blend
    |-- g1_jelly_drop.blend1
    |-- g1_jelly_drop.mp4
    |-- frames/
    |-- previews/
    |-- contact_sheet.jpg
    |-- *.log
    `-- *_validation.json, *.video.json
```

Start these PowerShell commands from the repository root. Adjust the Blender executable path for your machine.

```powershell
$blenderExe = 'D:\Program Files\Blender Foundation\Blender 5.2\blender.exe'
Set-Location .\demos\g1_jelly_drop
New-Item -ItemType Directory -Force .\outputs | Out-Null
```

Run the centralized entry from this dependency directory:

```powershell
uv run python ..\demo\demo_g1_jelly_drop.py
```

The G1 assembly and jelly generation scripts live in this directory. Robot assets remain in
`assets/g1_jelly_drop/` at the repository root and are resolved independently of the working directory.

### Jelly material and rebound

The default material uses `--jelly-mu 4000`, `--jelly-lambda 80000`, and `--jelly-damping 12`.
Compared with the old shear modulus of 2000 and damping of 180, it recovers shape faster and visibly
oscillates after impact. The volume stiffness stays at 80000 so the stack can still support the robot.
`--contact-kd 25` reduces normal contact damping; `--substeps 8` resolves the rebound at 480 Hz with the
current 60 FPS simulation. Increasing `--jelly-damping` suppresses oscillation; increasing `--jelly-mu`
makes shape recovery firmer. These options are also accepted by `export_simulation.py` and saved in metadata.

A gravity-free, contact-free cube released from 20% volume-preserving compression recovered through its
rest height in 0.05 s and overshot by 11.8% with this preset. The old material recovered monotonically,
with no meaningful overshoot at the same timestep. This is a visual tuning preset, not measured food material data.
Saved USD animations contain baked motion: export again and rebuild the Blender project to see material changes.

Export the current example's physics simulation. Unspecified options use the current defaults from
`../demo/demo_g1_jelly_drop.py`. You can append `--jelly-width`, `--layers`, `--drop-height`, `--layer-gap`,
`--base-height`, and other example options. The current simulation samples at 50 Hz; `--num-frames 250` saves 251 samples,
including the initial state. The exporter derives USD FPS, NPZ timestamps, and metadata times from the actual
`Example.frame_dt`; physics substeps do not change the output sample rate.

```powershell
uv run --with usd-core python .\export_simulation.py --num-frames 250 --substeps 4
uv run --with usd-core python .\validate_simulation.py
```

The exporter writes `outputs/simulation.usdc`, `outputs/simulation_states.npz`, and `outputs/simulation_metadata.json`.
Metadata records the actual parameters and source hashes. `validate_simulation.py` compares the USD with the raw states.
If you change the simulation source after exporting, the source hash check will fail; export again to validate that version.
Exports made before source files were moved or renamed retain their original provenance and remain usable for rendering.
Export again to validate a new simulation against the current source paths and hashes.

Build an editable Blender project from the USD:

```powershell
& $blenderExe -b --python-exit-code 1 --python .\build_render_scene.py -- --config .\render_config.json
& $blenderExe -b .\outputs\g1_jelly_drop.blend --python-exit-code 1 --python ..\utils\validate_import.py -- `
    --usd .\outputs\simulation.usdc
```

The builder checks `simulation_fps` against USD `timeCodesPerSecond` before importing or saving a project.
If they differ after changing the simulation rate, re-export with the corrected exporter before rebuilding.
The `usd` and `project` paths in `render_config.json` are relative to this directory and point into `outputs/` by default.
If you change the cache length, update `frame_end` in the configuration and `--end` in the encoding command.
Check camera framing after changing scene dimensions. The build script overwrites `outputs/g1_jelly_drop.blend`,
including any manual edits in that file.

`camera.fov` sets the horizontal field of view in degrees (strictly between 0 and 180).
The default 41.11209 degrees matches the previous 48 mm lens on a 36 mm sensor. Larger values show more of the
scene; smaller values frame the subject more tightly. Vertical FOV follows the render aspect ratio.
Rebuild the Blender project after changing FOV; the USD animation does not need to be regenerated.
Older configs can still use `camera.lens` in millimeters; `camera.fov` takes precedence when both are present.

After changing example parameters and exporting again, always run `build_render_scene.py` again before rendering.
`render_config.json` controls G1 metallic with `robot_metallic` and G1 surface roughness with `robot_roughness` (0.3).
`robot_roughness` applies to all G1 body materials: 0 gives sharp reflections and 1 gives broad, matte reflections.
Configs without this key retain the previous roughness values (0.30 for dark parts, 0.26 for other parts).
`floor_metallic` and `floor_roughness` control the studio floor's material.
Rebuild the Blender project after changing these material settings; no new simulation is needed.
The viewer and new USD exports omit the ground rings. Rebuilding also hides rings from older USD exports,
so removing them from a rendered scene does not require another simulation.
Replacing `simulation.usdc` does not recreate the objects or static geometry already imported into the `.blend`.
For example, reducing the number of jelly layers can leave old cubes visible in an unreconstructed project.
The builder starts from an empty scene and records the USD fingerprint; rendering rejects changed caches or missing
USD object paths and asks you to rebuild. Back up any manually edited project before rebuilding.

The default preset uses Cycles / OptiX GPU, 1920 x 1080 resolution, up to 256 samples, an adaptive threshold of 0.01,
OpenImageDenoise, ACES 2.0 tone mapping, depth of field, and motion blur. Jelly materials use transmission and volume absorption with
2 levels of render subdivision. The USD retains the original simulation mesh.

Color management uses Blender's native OCIO view transform. Set `view_transform` (default `ACES 2.0`),
`look` (default `None`), `exposure`, and `gamma` (1.0) in `render_config.json`, then rebuild the Blender project.
`gamma` is an additional artistic adjustment, not the output encoding gamma. The OCIO display transform already
handles sRGB output encoding; keep this adjustment at 1.0 for a neutral result with ACES.
ACES 2.0 requires a Blender installation whose color configuration provides that view (the supplied 5.2 preset does).
Tone mapping applies when viewing/rendering images; the USD simulation data remains in its original color representation.

The four studio area lights are twice as far from their respective targets as in the previous preset.
Their power is multiplied by four and their size by two, preserving direct illumination and angular size at each target.
To repeat this adjustment, use `position = target + scale * (position - target)`, multiply `energy` by `scale**2`,
and multiply `size` by `scale`. This reduces lighting variation across the floor; indirect light and shadows may still change.

Preview selected frames in `outputs/previews/`:

```powershell
& $blenderExe -b .\outputs\g1_jelly_drop.blend --python-exit-code 1 --python ..\utils\render_frames.py -- `
    --frames 0,30,60,120,240 --preview --config .\render_config.json --overwrite
```

Render the complete sequence and encode the video:

```powershell
& $blenderExe -b .\outputs\g1_jelly_drop.blend --python-exit-code 1 --python ..\utils\render_frames.py -- `
    --animation --overwrite
uv run --with imageio-ffmpeg --with pillow python ..\utils\encode_video.py `
    --output-dir .\outputs --output g1_jelly_drop.mp4 --start 0 --end 240 --contact-sheet
```

Final frames are saved as 16-bit RGB PNGs from `outputs/frames/frame_0000.png` through `frame_0250.png`.
The video is written to `outputs/g1_jelly_drop.mp4` with H.264, CRF 17, 30 fps, BT.709, and yuv420p by default.
Encoding all 251 samples from a 50 Hz simulation at 30 fps gives 0.6x playback speed and about 8.37 seconds of video.
Use `--fps 50` for real-time playback or `--crf 15` for higher encoding quality.
The encoder checks frame continuity, dimensions, and full video decoding. It preserves all PNGs and saves verification
results in `outputs/*.video.json`.
Relative `--input` and `--output` paths for `encode_video.py` resolve under `--output-dir`; absolute paths are supported.
`render_frames.py` defaults to `frames/` or `previews/` beside the `.blend`; use `--output-dir` to override this location.

To adjust only the rendering, open `outputs/g1_jelly_drop.blend`, edit materials, cameras, lights, or Cycles settings,
and save the project. Then run the preview, rendering, and encoding commands above.
`render_frames.py` uses the saved project settings and skips existing frames without `--overwrite`, allowing interrupted
renders to resume. After changing render parameters, use `--overwrite` to avoid mixing old and new frames.
`--overwrite` replaces only the requested PNGs. It does not rebuild the scene or remove other preview frames.
Use a fresh `--output-dir` if you need a directory containing only the newly selected frames.

The project reads its animation through the relative path `//simulation.usdc`, so move the entire `outputs/` together.
Redirect logs into this directory as well, for example with `> .\outputs\render.log 2>&1`.
Existing logs, previews, videos, caches, validation reports, and Blender backups are all stored in `outputs/`.
