"""Cut the paddle panel region out of the modified backplate for a quick fit test print."""
from pathlib import Path

import trimesh
ROOT = Path(__file__).resolve().parent.parent
SRC = str(ROOT / "output" / "Xbox_One_Backplate_Elite_Paddles.stl")
OUT = str(ROOT / "output" / "Xbox_One_Backplate_Elite_Paddles_TEST_COUPON.stl")
m = trimesh.load(SRC); m.merge_vertices(); print("volume:", m.is_volume, "watertight:", m.is_watertight, "winding:", m.is_winding_consistent)
box = trimesh.creation.box(extents=(66, 20, 30))
box.apply_translation((30.5, 20.0, 12.0))   # x -2.5..63.5, y 10..30, z -3..27
c = trimesh.boolean.intersection([m, box], engine="manifold")
print("watertight", c.is_watertight, "bounds", c.bounds.round(1).tolist())
c.export(OUT)
