"""Add the Elite paddle interface to Xbox_One_Backplate_Electronic.stl:
hook openings (+ retaining ramp block inside), button holes with 6x6 tact-switch seats.
Magnet pockets are disabled (MAG_Y = ()): magnets are glued on by hand.

All dimensions in mm, model coordinates (exterior = +Z, panel centre X = CX, Y grows from the
battery cage towards the grip bottom). Values come from caliper measures, photo estimates and
coupon-test shifts: see "Provenance" in README.md before changing them.

Run (from the project root):  uv run --with numpy --with trimesh --with scipy --with networkx --with rtree --with manifold3d --with shapely --with mapbox-earcut python tools/build_backplate.py
"""
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parent.parent
SRC = str(ROOT / "src" / "Xbox_One_Backplate_Electronic.stl")
OUT = str(ROOT / "output" / "Xbox_One_Backplate_Elite_Paddles.stl")

# ---------------- parameters ----------------
CX = 30.5                  # panel centre line (stub / cage pin)
ROW_Y = (15.8, 23.5)       # paddle rows (P1/P3 upper, P2/P4 lower), model Y

# round holes (button column), measured from photo
BTN_D = 3.9                # hole diameter
BTN_Y = (16.2, 24.1)

# hook openings: straight rectangular through-slots (as in the Elite / Elite V2 shells)
HOOK_DX = 18.55            # distance from centre line to the opening centre (23.05 original estimate, minus 3.5 and 1.0 mm shifts from coupon tests)
OPEN_W, OPEN_L = 4.0, 6.0  # opening width (X) / length (Y), caliper-measured on the real Elite plate
CLEAR = 0.25               # extra clearance on every opening side (3D-print tolerance)
NOTCH_L, NOTCH_H = 2.0, 2.0  # 2x2 mm slit towards the button column, CENTRED on the opening's long side
WALL_TO_BTN = 1.0          # minimum plastic between the slit tip and the button hole
# retaining ramp block on the INSIDE face, covers the outer part of each opening
BLK_W = 2.0                # how far the block reaches over the opening (from the outer edge) = length of the ~2 mm slit
BLK_Y_OVER = 0.4           # block overshoots the opening ends (avoids coplanar faces -> non-manifold edges)
BLK_GAP_MARGIN = 0.15      # block keeps this much clear of the slit walls
BLK_ATTACH = 1.6           # how far it is anchored under the plate beyond the opening
RAMP_H = 2.4               # block depth below the inside face (at the outer wall)
RAMP_TIP = 0.5             # remaining depth at the free end (chamfer)

BTN_DX = HOOK_DX + OPEN_W / 2 + CLEAR + NOTCH_L + CLEAR + WALL_TO_BTN + BTN_D / 2 + CLEAR

# tact-switch seats (6x6 mm body glued on the inside face, plunger goes up through the round hole)
SW_SIZE = 6.0              # body side
SW_CLEAR = 0.3             # total clearance in the pocket
SW_WALL = 1.1              # frame wall thickness
SW_HEIGHT = 5.0            # chosen switch total height (user test: 5 mm)
SW_BODY_H = 3.0            # body height implied by the user: a 5 mm switch ends flush with the exterior face (plunger = SW_HEIGHT - SW_BODY_H)
SW_DEPTH = 2.0             # frame depth below the inside face
# the side facing the hook opening stays OPEN so the paddle hook never hits a wall

# magnets 8x3 neodymium
MAG_D, MAG_H = 8.0, 3.0
MAG_FIT = 0.25             # diametral clearance
MAG_DX = 13.8              # distance from centre line (Elite magnet pocket centre)
MAG_Y = ()                 # magnet pockets removed: magnets are glued on and positioned by hand (was (19.1,))
SKIN = 0.6                 # plastic left between magnet and paddle
BOSS_D = 11.4              # boss outside diameter (inside face)
PANEL_T = 2.03             # panel thickness along the normal
# --------------------------------------------


def ray_z(mesh, x, y):
    locs, _, _ = mesh.ray.intersects_location([[x, y, 100.0]], [[0, 0, -1.0]])
    z = np.sort(locs[:, 2])[::-1]
    return z[0], z[1]  # exterior, interior


def fit_surface(mesh):
    """Exterior surface height by ray casting, smoothed over a 3x3 mm patch."""
    cache = {}

    def z(x, y):
        k = (round(x, 3), round(y, 3))
        if k not in cache:
            vals = [ray_z(mesh, x + dx, y + dy)[0] for dx in (-0.75, 0, 0.75) for dy in (-0.75, 0, 0.75)]
            cache[k] = float(np.mean(vals))
        return cache[k]

    return z


def pocket_frame(mesh, x, y, r=4.8):
    """Frame on the *lowest* exterior surface (groove bottoms) over a disc of radius r,
    so decorative grooves never pierce the skin above a magnet pocket."""
    pts = []
    for dx in np.arange(-r, r + 0.01, 0.4):
        for dy in np.arange(-r, r + 0.01, 0.4):
            if dx * dx + dy * dy <= r * r:
                pts.append((x + dx, y + dy, ray_z(mesh, x + dx, y + dy)[0]))
    P = np.array(pts)
    A = np.c_[np.ones(len(P)), P[:, 0], P[:, 1]]
    c = np.linalg.lstsq(A, P[:, 2], rcond=None)[0]
    shift = (P[:, 2] - A @ c).min()          # <= 0
    n = np.array([-c[1], -c[2], 1.0]); n /= np.linalg.norm(n)
    ex = np.cross([0, 1, 0], n); ex /= np.linalg.norm(ex)
    ey = np.cross(n, ex)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2] = ex, ey, n
    T[:3, 3] = [x, y, c[0] + c[1] * x + c[2] * y + shift]
    print(f"pocket ({x:.1f},{y:.1f}): groove depth {-shift:.2f} mm, tilt {np.degrees(np.arccos(n[2])):.1f} deg")
    return T


def local_frame(zf, x, y):
    e = 1.5
    dzdx = (zf(x + e, y) - zf(x - e, y)) / (2 * e)
    dzdy = (zf(x, y + e) - zf(x, y - e)) / (2 * e)
    n = np.array([-dzdx, -dzdy, 1.0]); n /= np.linalg.norm(n)
    ex = np.cross([0, 1, 0], n); ex /= np.linalg.norm(ex)
    ey = np.cross(n, ex)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = ex, ey, n, [x, y, zf(x, y)]
    return T


def box(sx, sy, sz, cx, cy, cz, T):
    b = trimesh.creation.box(extents=(sx, sy, sz))
    b.apply_translation((cx, cy, cz))
    b.apply_transform(T)
    return b


def cyl(d, z0, z1, T, cx=0.0, cy=0.0):
    c = trimesh.creation.cylinder(radius=d / 2, height=z1 - z0, sections=64)
    c.apply_translation((cx, cy, (z0 + z1) / 2))
    c.apply_transform(T)
    return c


def ramp_block(T, side, outer, y0, y1):
    """Wedge below the inside face: full depth at the outer wall, chamfered towards the opening."""
    zi = -PANEL_T
    prof = [(-BLK_ATTACH, zi + 0.4), (-BLK_ATTACH, zi - RAMP_H), (0.0, zi - RAMP_H),
            (BLK_W, zi - RAMP_TIP), (BLK_W, zi + 0.4)]          # (u, z), u grows towards the centre line
    from shapely.geometry import Polygon
    wedge = trimesh.creation.extrude_polygon(Polygon(prof), height=y1 - y0)   # x=u, y=z, extrusion=y
    v = wedge.vertices.copy()
    out = np.c_[outer - side * v[:, 0], y0 + v[:, 2], v[:, 1]]
    b = trimesh.Trimesh(out, wedge.faces.copy(), process=True)
    b.fix_normals()
    b.apply_transform(T)
    return b


def switch_frame(T, side):
    """3-sided locating frame for a 6x6 tact switch below the inside face.
    s axis = outwards (away from the centre line); the inner side (s<0) is open."""
    half = (SW_SIZE + SW_CLEAR) / 2
    z0, z1 = -(PANEL_T + SW_DEPTH), -PANEL_T + 0.4
    h = z1 - z0
    cz = (z0 + z1) / 2
    parts = []
    # outer wall
    parts.append(box(SW_WALL, 2 * half + 2 * SW_WALL, h, side * (half + SW_WALL / 2), 0, cz, T))
    # upper / lower walls, from the open inner edge to the outer wall
    inset = 0.4            # keep the wall ends clear of the slit tip (avoids sliver geometry)
    L = 2 * half + SW_WALL - inset
    for sy in (-1, 1):
        parts.append(box(L, SW_WALL, h, side * (half + SW_WALL - L / 2), sy * (half + SW_WALL / 2), cz, T))
    return parts


def union(meshes):
    return trimesh.boolean.union(meshes, engine="manifold")


def main():
    m = trimesh.load(SRC)
    m.merge_vertices()
    assert m.is_watertight, "source mesh not watertight"
    zf = fit_surface(m)

    bosses, cutters, blocks = [], [], []
    for side in (-1, 1):                       # -1 = left (x < CX), +1 = right
        # --- button holes
        for by in BTN_Y:
            T = local_frame(zf, CX + side * BTN_DX, by)
            cutters.append(cyl(BTN_D + 2 * CLEAR, -6, 4, T))
            bosses.extend(switch_frame(T, side))
        # --- hook openings (straight rectangles) + retaining ramp blocks on the inside
        for ry in ROW_Y:
            T = local_frame(zf, CX + side * HOOK_DX, ry)
            w, l = OPEN_W + 2 * CLEAR, OPEN_L + 2 * CLEAR
            cutters.append(box(w, l, 10, 0, 0, -1, T))
            outer = side * w / 2
            # centred slit: through-cut extension of the opening towards the button column
            nl, nh = NOTCH_L + CLEAR, NOTCH_H + 2 * CLEAR
            cutters.append(box(nl + 0.1, nh, 10, side * (w / 2 + nl / 2 - 0.05), 0, -1, T))
            # ramp block below the inside face, split so the slit band stays clear
            blocks.append(ramp_block(T, side, outer, -l / 2 - BLK_Y_OVER, -nh / 2 - BLK_GAP_MARGIN))
            blocks.append(ramp_block(T, side, outer, nh / 2 + BLK_GAP_MARGIN, l / 2 + BLK_Y_OVER))
        # --- magnet pockets + bosses
        for my in MAG_Y:
            T = pocket_frame(m, CX + side * MAG_DX, my)
            bosses.append(cyl(BOSS_D, -(PANEL_T + 1.8), -1.0, T))
            cutters.append(cyl(MAG_D + MAG_FIT, -(PANEL_T + 1.8) - 0.5, -SKIN, T))

    tip = SW_HEIGHT - SW_BODY_H - PANEL_T
    print(f"switch {SW_HEIGHT:g} mm: plunger tip {tip:+.2f} mm vs exterior face ({'proud' if tip > 0 else 'recessed'})")
    print("adding bosses...")
    body = union([m] + bosses)
    print("cutting...")
    body = trimesh.boolean.difference([body, union(cutters)], engine="manifold")
    print("adding ramp blocks...")
    body = union([body] + blocks)
    print("watertight:", body.is_watertight, "volume:", round(body.volume, 1),
          "delta vs source:", round(body.volume - m.volume, 1))
    body.export(OUT)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
