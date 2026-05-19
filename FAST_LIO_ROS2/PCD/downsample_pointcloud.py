#!/usr/bin/env python3
"""
Memory-efficient point cloud downsampler.

For binary PCD files with voxel method: streams in chunks — peak RAM is
bounded by the OUTPUT size, not the input size.
All other formats/methods fall back to open3d (loads fully into RAM).
"""

import argparse
import sys
import numpy as np
from pathlib import Path

CHUNK_PTS    = 500_000       # points read per iteration
VOXEL_OFFSET = 100_000       # supports ±5000 m at 5 cm voxel size (5000/0.05)
VOXEL_STRIDE = np.int64(2 * VOXEL_OFFSET + 1)  # 200 001


# ── PCD helpers ───────────────────────────────────────────────────────────────

def read_pcd_header(path):
    fields = {}
    offset = 0
    with open(path, 'rb') as f:
        while True:
            line = f.readline()
            offset += len(line)
            s = line.decode('ascii', errors='replace').strip()
            if not s or s.startswith('#'):
                continue
            parts = s.split()
            fields[parts[0].upper()] = parts[1:]
            if parts[0].upper() == 'DATA':
                break
    return fields, offset


def pcd_numpy_dtype(fields):
    type_map = {'F': 'f', 'I': 'i', 'U': 'u'}
    dt = []
    for name, ftype, fsize, fcount in zip(
            fields['FIELDS'], fields['TYPE'], fields['SIZE'], fields['COUNT']):
        np_type = type_map[ftype] + fsize
        n = int(fcount)
        dt.append((name, np_type) if n == 1 else (name, np_type, (n,)))
    return np.dtype(dt)


def build_pcd_header(ref_fields, n_points, data_type):
    def h(k, vals): return f"{k} {' '.join(vals)}\n"
    return (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        + h("VERSION",   ref_fields.get("VERSION", ["0.7"]))
        + h("FIELDS",    ref_fields["FIELDS"])
        + h("SIZE",      ref_fields["SIZE"])
        + h("TYPE",      ref_fields["TYPE"])
        + h("COUNT",     ref_fields["COUNT"])
        + f"WIDTH {n_points}\nHEIGHT 1\n"
        + h("VIEWPOINT", ref_fields.get("VIEWPOINT", ["0","0","0","1","0","0","0"]))
        + f"POINTS {n_points}\n"
        + h("DATA", [data_type])
    )


# ── Streaming voxel downsampler (binary PCD only) ─────────────────────────────

def voxel_downsample_pcd(input_path, output_path, voxel_size):
    header_fields, data_offset = read_pcd_header(input_path)
    data_type = header_fields['DATA'][0].lower()

    if data_type == 'binary_compressed':
        sys.exit("ERROR: binary_compressed PCD is not supported — convert to binary first.")

    n_total    = int(header_fields['POINTS'][0])
    dtype      = pcd_numpy_dtype(header_fields)
    point_size = dtype.itemsize

    seen   = set()   # occupied voxel keys (Python int, one per output point)
    chunks = []      # accumulated output point arrays

    print(f"Input:  {n_total:,} points  |  voxel_size={voxel_size} m  |  {point_size} bytes/pt")

    with open(input_path, 'rb') as f:
        f.seek(data_offset)
        n_done = 0

        while n_done < n_total:
            n_read = min(CHUNK_PTS, n_total - n_done)

            if data_type == 'binary':
                raw   = f.read(n_read * point_size)
                chunk = np.frombuffer(raw, dtype=dtype).copy()
            else:
                rows  = [f.readline() for _ in range(n_read)]
                chunk = np.array([tuple(r.split()) for r in rows], dtype=dtype)

            # Voxel indices
            vx = np.floor(chunk['x'] / voxel_size).astype(np.int64)
            vy = np.floor(chunk['y'] / voxel_size).astype(np.int64)
            vz = np.floor(chunk['z'] / voxel_size).astype(np.int64)

            if (np.any(np.abs(vx) > VOXEL_OFFSET) or
                    np.any(np.abs(vy) > VOXEL_OFFSET) or
                    np.any(np.abs(vz) > VOXEL_OFFSET)):
                sys.exit("ERROR: coordinates exceed ±5000 m — increase VOXEL_OFFSET in the script.")

            flat = ((vx + VOXEL_OFFSET) * VOXEL_STRIDE * VOXEL_STRIDE
                    + (vy + VOXEL_OFFSET) * VOXEL_STRIDE
                    + (vz + VOXEL_OFFSET))

            # One representative per voxel within this chunk
            unique_keys, first_idx = np.unique(flat, return_index=True)

            # Keep only voxels not yet seen globally
            keep = []
            for k, i in zip(unique_keys.tolist(), first_idx.tolist()):
                if k not in seen:
                    seen.add(k)
                    keep.append(i)

            if keep:
                chunks.append(chunk[keep])

            n_done += n_read
            print(f"\r  {n_done:,}/{n_total:,}  →  {len(seen):,} output points", end='', flush=True)

    print()

    out   = np.concatenate(chunks) if chunks else np.zeros(0, dtype=dtype)
    n_out = len(out)
    print(f"Output: {n_out:,} points  ({n_out / n_total * 100:.1f}% of input)")

    hdr = build_pcd_header(header_fields, n_out, data_type)
    with open(output_path, 'wb') as f:
        f.write(hdr.encode('ascii'))
        f.write(out.tobytes())

    print(f"Saved:  {output_path}")


# ── open3d fallback for non-PCD formats or random method ──────────────────────

def downsample_open3d(input_path, output_path, method, voxel_size, target_points):
    try:
        import open3d as o3d
    except ImportError:
        sys.exit("open3d is required for non-PCD files: pip install open3d")

    print(f"Loading (open3d): {input_path}")
    pcd    = o3d.io.read_point_cloud(input_path)
    before = len(pcd.points)
    print(f"Points before: {before:,}")

    if method == 'voxel':
        result = pcd.voxel_down_sample(voxel_size=voxel_size)
    else:
        ratio  = min(target_points / before, 1.0)
        result = pcd.random_down_sample(ratio)

    after = len(result.points)
    print(f"Points after:  {after:,}  ({after / before * 100:.1f}%)")
    o3d.io.write_point_cloud(output_path, result)
    print(f"Saved: {output_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Memory-efficient point cloud downsampler.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python downsample_pointcloud.py merged.pcd out.pcd --method voxel --voxel-size 0.05
  python downsample_pointcloud.py scan.ply   out.ply --method voxel --voxel-size 0.10
  python downsample_pointcloud.py scan.pcd   out.pcd --method random --target-points 1000000
        """,
    )
    parser.add_argument("input",  help="Input point cloud (.pcd, .ply, .las, …)")
    parser.add_argument("output", help="Output file")
    parser.add_argument("--method", choices=["voxel", "random"], default="voxel")
    parser.add_argument("--voxel-size",    type=float, default=0.05,
                        help="Voxel grid size in metres (default: 0.05)")
    parser.add_argument("--target-points", type=int,   default=1_000_000,
                        help="Target count for random method (default: 1 000 000)")
    args = parser.parse_args()

    ext = Path(args.input).suffix.lower()
    if ext == '.pcd' and args.method == 'voxel':
        voxel_downsample_pcd(args.input, args.output, args.voxel_size)
    else:
        downsample_open3d(args.input, args.output,
                          args.method, args.voxel_size, args.target_points)


if __name__ == "__main__":
    main()
