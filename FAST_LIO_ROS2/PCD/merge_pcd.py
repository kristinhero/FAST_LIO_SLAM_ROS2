import glob
import os

path = "/home/aidit/helmet_scans/no_motion_deskew_curved_tank_fast"
pcd_files = sorted(glob.glob(os.path.join(path, "scans_*.pcd")))

if not pcd_files:
    print("No scan PCD files found!")
    exit(1)

print(f"Found {len(pcd_files)} files")

CHUNK = 4 * 1024 * 1024  # 4 MB read buffer


def read_header(filepath):
    """Read PCD header, return (fields dict, byte offset where data starts)."""
    fields = {}
    offset = 0
    with open(filepath, 'rb') as f:
        while True:
            line = f.readline()
            offset += len(line)
            line_str = line.decode('ascii', errors='replace').strip()
            if not line_str or line_str.startswith('#'):
                continue
            parts = line_str.split()
            fields[parts[0].upper()] = parts[1:]
            if parts[0].upper() == 'DATA':
                break
    return fields, offset


# ── Pass 1: read all headers (no point data loaded) ───────────────────────────
print("Reading headers...")
all_offsets = []
all_points  = []
ref         = None

for fp in pcd_files:
    fields, offset = read_header(fp)
    all_offsets.append(offset)
    all_points.append(int(fields['POINTS'][0]))
    if ref is None:
        ref = fields

total_points = sum(all_points)
data_type    = ref['DATA'][0].lower()
print(f"Total points: {total_points:,}  data type: {data_type}")

if data_type == 'binary_compressed':
    print("ERROR: binary_compressed is not supported by this script.")
    exit(1)

if data_type == 'binary':
    sizes      = [int(s) for s in ref['SIZE']]
    counts     = [int(c) for c in ref['COUNT']]
    point_size = sum(s * c for s, c in zip(sizes, counts))
    print(f"Bytes per point: {point_size}")

# ── Build output header ────────────────────────────────────────────────────────
def hline(key, vals):
    return f"{key} {' '.join(vals)}\n"

header = (
    "# .PCD v0.7 - Point Cloud Data file format\n"
    + hline("VERSION",  ref.get("VERSION",  ["0.7"]))
    + hline("FIELDS",   ref["FIELDS"])
    + hline("SIZE",     ref["SIZE"])
    + hline("TYPE",     ref["TYPE"])
    + hline("COUNT",    ref["COUNT"])
    + f"WIDTH {total_points}\n"
    + "HEIGHT 1\n"
    + hline("VIEWPOINT", ref.get("VIEWPOINT", ["0","0","0","1","0","0","0"]))
    + f"POINTS {total_points}\n"
    + hline("DATA", ref["DATA"])
)

# ── Pass 2: stream data from each file ────────────────────────────────────────
merged = os.path.join(path, "merged.pcd")
print(f"Writing {merged} ...")

with open(merged, 'wb') as out:
    out.write(header.encode('ascii'))

    for i, (fp, offset, n_pts) in enumerate(zip(pcd_files, all_offsets, all_points)):
        print(f"  [{i+1}/{len(pcd_files)}] {os.path.basename(fp)}  ({n_pts:,} pts)")
        with open(fp, 'rb') as f:
            f.seek(offset)
            if data_type == 'binary':
                remaining = n_pts * point_size
                while remaining > 0:
                    chunk = f.read(min(CHUNK, remaining))
                    if not chunk:
                        break
                    out.write(chunk)
                    remaining -= len(chunk)
            else:  # ascii
                for _ in range(n_pts):
                    out.write(f.readline())

print(f"Done. {total_points:,} points -> {merged}")
