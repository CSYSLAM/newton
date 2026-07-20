# Newton Gear Crusher Demo

Standalone reproduction of the **Gear Crusher** scene from Fig. 8 of *Divide and Truncate* using Newton's current `SolverVBD`, whose particle self-contact path contains Planar-DAT truncation.

## Run

Install Newton with examples support, then run from this directory:

```bash
pip install "newton[examples]>=1.3.0"
python example_gear_crusher.py --viewer gl --device cuda:0 --num-frames 360
```

From a Newton source checkout, you can also copy this directory anywhere and run it through the checkout environment:

```bash
uv run --extra examples python /path/to/gear_crusher_newton/example_gear_crusher.py \
  --viewer gl --device cuda:0 --num-frames 360
```

For USD output:

```bash
python example_gear_crusher.py --viewer usd --output-path gear_crusher.usd \
  --device cuda:0 --num-frames 360
```

## Included assets

- `assets/crusher_gear.npz`: collision/render triangle mesh used by the demo.
- `assets/crusher_gear.obj`: the same gear in a standard interchange format.
- `assets/armadillo_proxy_tet.npz`: 12,559 vertices and 59,016 positive-volume tetrahedra.
- `assets/armadillo_proxy_surface.obj`: extracted surface for inspection/rendering.

The armadillo-like proxy and gear are original procedural assets. Run `python generate_assets.py` to regenerate them; only NumPy is required.

## Paper-matched parameters

The demo uses `dt=1/600 s`, 10 VBD iterations, Lamé parameters `lambda=1e6`, `mu=1e5`, contact stiffness `kc=1e6`, and friction `mu_f=0.2`. The supplied proxy is coarser than the paper's production Armadillo, so its self-contact radius is set to 14 mm rather than the paper's 5 mm contact radius.
