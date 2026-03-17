#!/usr/bin/env python3
"""Fix the mesh topology by merging duplicate vertices while preserving UVs.

The original mesh has 176,397 vertices but only 31,868 unique positions.
Each face has its own copy of vertices, breaking mesh connectivity.
This script merges spatially coincident vertices and rebuilds face indices.
"""

import numpy as np
from PIL import Image
import os
import trimesh

SCAN_DIR = "data/scan_20260317_132224/scan_20260317_132224"
OUT_DIR = "data/scan_fixed"
ANALYSIS_DIR = "data/scan_analysis"
os.makedirs(OUT_DIR, exist_ok=True)


def parse_obj(path):
    verts, normals, uvs = [], [], []
    faces_v, faces_vt, faces_vn = [], [], []
    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                p = line.split()
                verts.append([float(p[1]), float(p[2]), float(p[3])])
            elif line.startswith("vn "):
                p = line.split()
                normals.append([float(p[1]), float(p[2]), float(p[3])])
            elif line.startswith("vt "):
                p = line.split()
                uvs.append([float(p[1]), float(p[2])])
            elif line.startswith("f "):
                parts = line.split()[1:]
                fv, ft, fn = [], [], []
                for p in parts:
                    idx = p.split("/")
                    fv.append(int(idx[0]) - 1)
                    ft.append(int(idx[1]) - 1 if len(idx) > 1 and idx[1] else 0)
                    fn.append(int(idx[2]) - 1 if len(idx) > 2 and idx[2] else 0)
                faces_v.append(fv)
                faces_vt.append(ft)
                faces_vn.append(fn)
    return (np.array(verts), np.array(normals), np.array(uvs),
            np.array(faces_v), np.array(faces_vt), np.array(faces_vn))


def merge_vertices(verts, faces_v, tolerance=1e-6):
    """Merge vertices at the same spatial position, rebuild face indices."""
    print("Merging duplicate vertices...")
    # Quantize positions for grouping
    quantized = np.round(verts / tolerance).astype(np.int64)

    # Map each quantized position to a unique index
    unique_map = {}
    new_verts = []
    old_to_new = np.zeros(len(verts), dtype=np.int32)

    for i in range(len(verts)):
        key = tuple(quantized[i])
        if key not in unique_map:
            unique_map[key] = len(new_verts)
            new_verts.append(verts[i])
        old_to_new[i] = unique_map[key]

    new_verts = np.array(new_verts)

    # Rebuild face indices
    new_faces = old_to_new[faces_v]

    # Remove degenerate faces (where two or more vertices collapsed to same)
    valid = []
    for i in range(len(new_faces)):
        f = new_faces[i]
        if f[0] != f[1] and f[1] != f[2] and f[0] != f[2]:
            valid.append(i)
    valid = np.array(valid)
    new_faces = new_faces[valid]

    print(f"  {len(verts)} -> {len(new_verts)} vertices")
    print(f"  {len(faces_v)} -> {len(new_faces)} faces")
    print(f"  Removed {len(faces_v) - len(new_faces)} degenerate faces")

    return new_verts, new_faces, old_to_new, valid


def write_obj_with_uvs(path, verts, faces_v, uvs, faces_vt, mtl_name="model.mtl"):
    """Write OBJ preserving separate vertex/UV indexing."""
    print(f"Writing {path}...")
    with open(path, 'w') as f:
        f.write("# Fixed mesh with merged vertices\n")
        f.write(f"mtllib {mtl_name}\n")
        f.write("usemtl textured_material\n\n")
        for v in verts:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        f.write("\n")
        for vt in uvs:
            f.write(f"vt {vt[0]} {vt[1]}\n")
        f.write("\n")
        for i in range(len(faces_v)):
            fv = faces_v[i]
            ft = faces_vt[i]
            f.write(f"f {fv[0]+1}/{ft[0]+1} "
                    f"{fv[1]+1}/{ft[1]+1} "
                    f"{fv[2]+1}/{ft[2]+1}\n")
    print(f"  {len(verts)} vertices, {len(uvs)} UVs, {len(faces_v)} faces")


def verify_fixed_mesh(path):
    """Load and verify the fixed mesh."""
    mesh = trimesh.load(path, process=False)
    print(f"\n=== FIXED MESH VERIFICATION ===")
    print(f"  Vertices: {len(mesh.vertices):,}")
    print(f"  Faces: {len(mesh.faces):,}")
    print(f"  Connected components: {mesh.body_count}")
    print(f"  Watertight: {mesh.is_watertight}")
    print(f"  Winding consistent: {mesh.is_winding_consistent}")

    # Vertex sharing
    unique_in_faces = len(np.unique(mesh.faces))
    avg_sharing = mesh.faces.size / unique_in_faces
    print(f"  Avg faces per vertex: {avg_sharing:.1f}")

    bb = mesh.bounds
    dims = bb[1] - bb[0]
    print(f"  Dimensions: {dims[0]:.2f} x {dims[1]:.2f} x {dims[2]:.2f} m")
    print(f"  Surface area: {mesh.area:.2f} m²")

    return mesh


def main():
    print("=" * 60)
    print("FIXING MESH TOPOLOGY")
    print("=" * 60)

    # Parse original
    obj_path = os.path.join(SCAN_DIR, "model.obj")
    verts, normals, uvs, faces_v, faces_vt, faces_vn = parse_obj(obj_path)
    print(f"Original: {len(verts)} verts, {len(uvs)} UVs, {len(faces_v)} faces")

    # Merge vertices (keep UVs separate since they need per-face-vertex values)
    new_verts, new_faces_v, old_to_new, valid_faces = merge_vertices(
        verts, faces_v
    )

    # Keep original UVs and UV face indices for valid faces
    new_faces_vt = faces_vt[valid_faces]

    # Write fixed OBJ
    fixed_obj = os.path.join(OUT_DIR, "model_fixed.obj")
    write_obj_with_uvs(fixed_obj, new_verts, new_faces_v, uvs, new_faces_vt)

    # Copy MTL and texture
    import shutil
    shutil.copy2(os.path.join(SCAN_DIR, "model.mtl"),
                 os.path.join(OUT_DIR, "model.mtl"))
    shutil.copy2(os.path.join(SCAN_DIR, "texture.jpg"),
                 os.path.join(OUT_DIR, "texture.jpg"))

    # Verify
    mesh = verify_fixed_mesh(fixed_obj)

    # Also try opening with macOS Quick Look
    print(f"\nFixed files saved to: {OUT_DIR}/")
    print("You can preview with: open data/scan_fixed/model_fixed.obj")

    print("\n" + "=" * 60)
    print("DIAGNOSIS SUMMARY")
    print("=" * 60)
    print("  Problem: Each triangle had its own 3 vertices (no sharing)")
    print("           176,397 verts = 58,799 faces × 3 = all isolated triangles")
    print(f"  Fix:     Merged to {len(new_verts)} unique vertices")
    print(f"           Now {mesh.body_count} connected component(s)")
    print("  Root cause: iOS scanner's TexturedMeshExporter duplicates")
    print("              vertices per-face for UV seams but doesn't merge")
    print("              vertices that share the same position.")


if __name__ == "__main__":
    main()
