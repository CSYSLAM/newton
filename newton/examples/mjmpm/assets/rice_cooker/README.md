# Rice cooker for MJMPM

`body.obj` and `lid.obj` were modelled in Blender 5.1.2 for
`mjmpm_w1_rice_cooker`; `model_rice_cooker.py` regenerates them. Units are
metres with the origin at the bottom centre and +z up.

- Body: 24 cm diameter, 15 cm tall, with an 18 cm wide, 11 cm deep inner pot
  and a control panel on the robot side (-x). The walls (3 cm) and the bottom
  (4 cm) are at least two MPM cells of the example's 1.5 cm grid thick, so rice
  grains cannot tunnel through them.
- Lid: a shallow dome hinged at the back rim (+x) and opened 100 degrees, clear
  of the cup during the pour.

Both meshes are closed (no non-manifold edges), as the MPM collider's signed
distance needs. They are original work for this example; no external assets
were used.
