Reusable offline rendering tools for demos. These scripts do not depend on the G1 scene and do not write render
artifacts into `utils/`. Explicit relative paths resolve from the current working directory, with the exceptions below.

| Script | Purpose and path rules |
| --- | --- |
| `render_frames.py` | Render a loaded Blender project. Frames go beside the `.blend`; override with `--output-dir`. |
| `encode_video.py` | Encode PNG frames to H.264 and verify the video. Input/output paths resolve under `--output-dir`. |
| `validate_import.py` | Compare a loaded Blender project with `--usd`. Reports default to the `.blend` directory. |
| `usd_cache.py` | Shared USD fingerprint and cache-reference checks used by scene builders and rendering tools. |
| `particle_forces.py` | Shared W1 particle-force kernels; scene entries do not import each other for these forces. |

Run these PowerShell examples from a demo directory. Adjust the Blender executable path for your machine.

```powershell
$blenderExe = 'D:\Program Files\Blender Foundation\Blender 5.2\blender.exe'
& $blenderExe -b .\outputs\scene.blend --python-exit-code 1 --python ..\utils\render_frames.py -- `
    --frames 0,60 --preview --config .\render_config.json --overwrite
& $blenderExe -b .\outputs\scene.blend --python-exit-code 1 --python ..\utils\render_frames.py -- --animation
uv run --with imageio-ffmpeg --with pillow python ..\utils\encode_video.py `
    --output-dir .\outputs --output video.mp4 --start 0 --end 240 --contact-sheet
& $blenderExe -b .\outputs\scene.blend --python-exit-code 1 --python ..\utils\validate_import.py -- `
    --usd .\outputs\simulation.usdc --frames 0,60,240
```

Rendering uses the settings saved in the project. Images default to `frames/` or `previews/` beside the `.blend`.
`--preview` temporarily changes sampling and resolution to 32 samples and 60% by default. Use `--config` to read
`preview_samples` and `preview_percentage` from a JSON file.
Cycles GPU projects automatically select an available local backend, falling back to CPU if none is available.
Use `--device CPU`, `--device OPTIX`, or another supported backend to select a device explicitly.
Existing frames are skipped unless `--overwrite` is supplied. Use it after changing render settings to avoid mixing
old and new frames. The render script does not save changes to the project.
`--overwrite` only replaces the requested image files; it does not rebuild imported objects or remove other frames.

Rendering and import validation reject cache references whose objects no longer exist in the USD. Scene builders can
call `record_usd_sources(scene)` from `usd_cache.py` after making cache paths relative and before the final save.
This records SHA-256 hashes of the imported USD files so rendering also rejects changed caches, including changes to
static geometry that Blender does not reload. Legacy projects without these hashes only receive the reference checks.
After exporting a new simulation, rebuild the project before rendering; back up manual scene edits first.

Encoding requires an `--output` filename. Relative `--input` and `--output` paths resolve under `--output-dir`,
which defaults to `./outputs`; absolute paths are also supported. `--fps` controls playback speed and defaults to 30.
Specify the inclusive last frame with `--end` to avoid treating an unfinished sequence prefix as a complete video.
The encoder checks frame continuity, dimensions, and full video decoding, preserves the PNGs, and writes a matching
`.video.json` report. `--contact-sheet` saves a six-frame overview beside the video.

Import validation reads the USD directly without simulation metadata. It checks mesh vertices, transforms, face counts,
and external cache paths at four times across the USD range by default. Use `--frames` to select sample times.
USD mesh basenames must be unique and match Blender object names; duplicate names across different hierarchies are rejected.
Subdivision modifiers are temporarily disabled to compare the original simulation surfaces. These changes are not saved.
External references must use resolvable relative paths within the project directory. Use `--report` to choose a report path.

Keep physics exporters, scene-specific materials and cameras, and validators tied to a particular state or metadata format
in the corresponding demo directory.
