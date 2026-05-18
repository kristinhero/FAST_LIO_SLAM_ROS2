"""
Export the gravity vector (from mat_out.txt cols 22-24) as a PLY mesh arrow
(thin cylinder shaft + cone arrowhead) that CloudCompare renders as a solid 3D object.
"""

import os
import sys
import numpy as np

_dir  = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Log_square_tank_fast')
T_MAX = None   # set to e.g. 235.0 to only use data before that time; None = use all
if len(sys.argv) > 1:
    _dir = sys.argv[1]
if len(sys.argv) > 2:
    T_MAX = float(sys.argv[2])

ARROW_LENGTH   = 2.0    # total length in metres
HEAD_FRACTION  = 0.20   # cone is this fraction of total length
SHAFT_RADIUS   = 0.02   # shaft cylinder radius in metres
HEAD_RADIUS    = 0.10   # cone base radius in metres
N_SIDES        = 12     # polygon approximation for circles


def perp_basis(axis_unit):
    ref = np.array([1.0, 0.0, 0.0]) if abs(axis_unit[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    p1 = np.cross(axis_unit, ref);  p1 /= np.linalg.norm(p1)
    p2 = np.cross(axis_unit, p1);   p2 /= np.linalg.norm(p2)
    return p1, p2


def make_cylinder(start, end, radius, n):
    axis = end - start
    p1, p2 = perp_basis(axis / np.linalg.norm(axis))
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    offsets = [radius * (np.cos(a) * p1 + np.sin(a) * p2) for a in angles]
    bot = [start + o for o in offsets]
    top = [end   + o for o in offsets]
    verts = bot + top
    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces += [(i, j, n + i), (j, n + j, n + i)]
    return verts, faces


def make_cone(base_center, tip, radius, n):
    axis = tip - base_center
    p1, p2 = perp_basis(axis / np.linalg.norm(axis))
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = [base_center + radius * (np.cos(a) * p1 + np.sin(a) * p2) for a in angles]
    # vertex 0 = tip, 1..n = base ring
    verts = [tip] + ring
    faces = [(0, 1 + i, 1 + (i + 1) % n) for i in range(n)]
    return verts, faces


def make_disk(center, outward, radius, n):
    """Flat cap (facing outward direction)."""
    p1, p2 = perp_basis(outward / np.linalg.norm(outward))
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = [center + radius * (np.cos(a) * p1 + np.sin(a) * p2) for a in angles]
    # vertex 0 = center, 1..n = ring
    verts = [center] + ring
    faces = [(0, 1 + i, 1 + (i + 1) % n) for i in range(n)]
    return verts, faces


# --- Load data ---
mat_out = np.loadtxt(os.path.join(_dir, 'mat_out.txt'))
rows    = mat_out[mat_out[:, 0] < T_MAX] if T_MAX is not None else mat_out
grav    = np.median(rows[:, 22:25], axis=0)
g_mag   = np.linalg.norm(grav)
g_unit  = grav / g_mag
print(f"Gravity vector (world frame): {grav}")
print(f"Magnitude: {g_mag:.4f} m/s²  direction: {g_unit}")

origin    = np.zeros(3)
cone_base = g_unit * ARROW_LENGTH * (1.0 - HEAD_FRACTION)
arrow_tip = g_unit * ARROW_LENGTH

# --- Build mesh parts ---
all_verts = []
all_faces = []

def add_part(verts, faces):
    offset = len(all_verts)
    all_verts.extend(verts)
    all_faces.extend([(f[0]+offset, f[1]+offset, f[2]+offset) for f in faces])

add_part(*make_cylinder(origin, cone_base, SHAFT_RADIUS, N_SIDES))
add_part(*make_disk(origin,    -g_unit,  SHAFT_RADIUS, N_SIDES))  # bottom cap
add_part(*make_disk(cone_base,  g_unit,  SHAFT_RADIUS, N_SIDES))  # top cap (between shaft and cone)
add_part(*make_cone(cone_base, arrow_tip, HEAD_RADIUS, N_SIDES))
add_part(*make_disk(cone_base, -g_unit,  HEAD_RADIUS, N_SIDES))   # cone base cap

# Cyan colour
R, G, B = 0, 200, 200

out_path = os.path.join(_dir, 'gravity_vector.ply')
with open(out_path, 'w') as f:
    f.write("ply\n")
    f.write("format ascii 1.0\n")
    f.write(f"element vertex {len(all_verts)}\n")
    f.write("property float x\n")
    f.write("property float y\n")
    f.write("property float z\n")
    f.write("property uchar red\n")
    f.write("property uchar green\n")
    f.write("property uchar blue\n")
    f.write(f"element face {len(all_faces)}\n")
    f.write("property list uchar int vertex_indices\n")
    f.write("end_header\n")
    for v in all_verts:
        f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {R} {G} {B}\n")
    for face in all_faces:
        f.write(f"3 {face[0]} {face[1]} {face[2]}\n")

print(f"Saved: {out_path}  ({len(all_verts)} vertices, {len(all_faces)} faces)")
