篇； ；‘/’ ；/#!/usr/bin/env python3
"""Optimize UV unwrapping for the scanned mesh using xatlas Atlas API.

Re-unwraps the mesh with proper chart segmentation and packing to eliminate
UV overlaps, then re-projects the texture from camera frames.
"""

import json
import os
import numpy as np
import xatlas
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt

SCAN_DIR = "data/scan_20260317_132224/scan_20260317_132224"
OUT_DIR = "data/scan_optimized"
ANALYSIS_DIR = "data/scan_analysis"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(ANALYSIS_DIR, exist_ok=True)

ATLAS_SIZE = 4096


def parse_obj(path):
    """Parse OBJ file."""
    vertices, normals, uvs = [], [], []
    faces_v = []
    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                p = line.split()
                vertices.append([float(p[1]), float(p[2]), float(p[3])])
            elif line.startswith("vn "):
                p = line.split()
                normals.append([float(p[1]), float(p[2]), float(p[3])])
            elif line.startswith("vt "):
                p = line.split()
                uvs.append([float(p[1]), float(p[2])])
            elif line.startswith("f "):
                parts = line.split()[1:]
                faces_v.append([int(p.split("/")[0]) - 1 for p in parts])
    return (np.array(vertices, dtype=np.float32),
            np.array(normals, dtype=np.float32),
            np.array(uvs, dtype=np.float32),
            np.array(faces_v, dtype=np.uint32))


def unwrap_with_xatlas(vertices, normals, faces):
    """Re-unwrap mesh using xatlas Atlas API with proper packing."""
    print("Running xatlas Atlas UV unwrap...")

    atlas = xatlas.Atlas()
    atlas.add_mesh(vertices, faces, normals)

    # Configure chart options for better segmentation
    chart_opts = xatlas.ChartOptions()
    chart_opts.max_iterations = 4
    chart_opts.normal_deviation_weight = 2.0
    chart_opts.normal_seam_weight = 4.0

    # Configure pack options for non-overlapping layout
    pack_opts = xatlas.PackOptions()
    pack_opts.resolution = ATLAS_SIZE
    pack_opts.padding = 2  # 2-pixel padding between charts
    pack_opts.bilinear = True
    pack_opts.bruteForce = True  # Better packing quality
    pack_opts.create_image = True

    atlas.generate(chart_opts, pack_opts, verbose=True)

    print(f"  Atlas: {atlas.atlas_count} atlas(es), "
          f"{atlas.chart_count} charts, "
          f"size {atlas.width}x{atlas.height}")

    # Get utilization
    for i in range(atlas.atlas_count):
        util = atlas.get_utilization(i)
        print(f"  Atlas {i} utilization: {util*100:.1f}%")

    # Extract mesh data
    vmapping, indices, new_uvs = atlas[0]  # first (only) mesh

    print(f"  Output: {len(vmapping)} vertices, {len(indices)} faces")
    print(f"  UV range: U[{new_uvs[:,0].min():.4f}, {new_uvs[:,0].max():.4f}] "
          f"V[{new_uvs[:,1].min():.4f}, {new_uvs[:,1].max():.4f}]")

    # UVs from Atlas API are already normalized to [0, 1]

    # Map vertices back
    new_verts = vertices[vmapping]
    new_normals = normals[vmapping] if len(normals) > 0 else None

    return new_verts, new_normals, new_uvs, indices, vmapping, atlas


def load_camera_data():
    """Load camera intrinsics and poses."""
    with open(os.path.join(SCAN_DIR, "intrinsics.json")) as f:
        intr = json.load(f)
    with open(os.path.join(SCAN_DIR, "poses.json")) as f:
        poses = json.load(f)
    return intr, poses


def project_point(point, pose_matrix, intr):
    """Project 3D world point to 2D pixel coords. Returns (px, py) or None."""
    view = np.linalg.inv(pose_matrix)
    p_cam = view @ np.append(point, 1.0)
    if p_cam[2] >= 0:
        return None
    neg_z = -p_cam[2]
    px = intr['fx'] * (p_cam[0] / neg_z) + intr['cx']
    py = intr['fy'] * (-p_cam[1] / neg_z) + intr['cy']
    if 0 <= px < intr['width'] and 0 <= py < intr['height']:
        return (px, py)
    return None


def select_best_frame(face_center, face_normal, frames, pose_matrices, intr):
    """Select best camera frame for a face."""
    best_idx, best_score = -1, 0
    for i in range(len(frames)):
        cam_pos = pose_matrices[i][:3, 3]
        to_cam = cam_pos - face_center
        dist = np.linalg.norm(to_cam)
        if dist < 1e-10:
            continue
        view_dir = to_cam / dist
        dot = np.dot(face_normal, view_dir)
        if dot <= 0:
            continue
        if project_point(face_center, pose_matrices[i], intr) is None:
            continue
        score = dot / (dist * dist)
        if score > best_score:
            best_score = score
            best_idx = i
    return best_idx, best_score


def render_texture_atlas(new_verts, new_normals, new_uvs, new_faces,
                         poses_data, intr):
    """Render texture atlas by projecting camera images onto UV space."""
    print("Rendering texture atlas...")
    frames = poses_data["frames"]

    # Precompute pose matrices
    pose_matrices = []
    for frame in frames:
        t = np.array(frame["transform"], dtype=np.float64).reshape(4, 4, order='F')
        pose_matrices.append(t)

    # Load all images
    images = []
    for frame in frames:
        img_path = os.path.join(SCAN_DIR, frame["imageFile"])
        img = np.array(Image.open(img_path).convert("RGB"), dtype=np.float64)
        images.append(img)

    atlas = np.zeros((ATLAS_SIZE, ATLAS_SIZE, 3), dtype=np.float64)
    atlas_weight = np.zeros((ATLAS_SIZE, ATLAS_SIZE), dtype=np.float64)

    total = len(new_faces)
    assigned = 0

    for fi in range(total):
        if fi % 10000 == 0:
            print(f"  Processing face {fi}/{total}...")

        face = new_faces[fi]
        v0, v1, v2 = new_verts[face[0]], new_verts[face[1]], new_verts[face[2]]
        center = (v0 + v1 + v2) / 3.0
        edge1, edge2 = v1 - v0, v2 - v0
        cross = np.cross(edge1, edge2)
        cross_len = np.linalg.norm(cross)
        if cross_len < 1e-10:
            continue
        normal = cross / cross_len

        best_idx, _ = select_best_frame(center, normal, frames, pose_matrices, intr)
        if best_idx < 0:
            continue

        pose = pose_matrices[best_idx]
        img = images[best_idx]
        h, w = img.shape[:2]

        # UV coords for this face
        uv0, uv1, uv2 = new_uvs[face[0]], new_uvs[face[1]], new_uvs[face[2]]
        px_uvs = np.array([uv0, uv1, uv2]) * (ATLAS_SIZE - 1)

        u_min = max(0, int(np.floor(px_uvs[:, 0].min())))
        u_max = min(ATLAS_SIZE - 1, int(np.ceil(px_uvs[:, 0].max())))
        v_min = max(0, int(np.floor(px_uvs[:, 1].min())))
        v_max = min(ATLAS_SIZE - 1, int(np.ceil(px_uvs[:, 1].max())))

        if u_min >= u_max or v_min >= v_max:
            continue

        assigned += 1
        a, b, c = px_uvs[0], px_uvs[1], px_uvs[2]
        v0b, v1b = c - a, b - a
        dot00 = np.dot(v0b, v0b)
        dot01 = np.dot(v0b, v1b)
        dot11 = np.dot(v1b, v1b)
        denom = dot00 * dot11 - dot01 * dot01
        if abs(denom) < 1e-10:
            continue
        inv_denom = 1.0 / denom

        for py in range(v_min, v_max + 1):
            for px in range(u_min, u_max + 1):
                p = np.array([px, py], dtype=np.float64)
                v2b = p - a
                dot02 = np.dot(v0b, v2b)
                dot12 = np.dot(v1b, v2b)
                u_bary = (dot11 * dot02 - dot01 * dot12) * inv_denom
                v_bary = (dot00 * dot12 - dot01 * dot02) * inv_denom

                if u_bary < -0.01 or v_bary < -0.01 or (u_bary + v_bary) > 1.01:
                    continue

                w0 = 1.0 - u_bary - v_bary
                w1 = v_bary
                w2 = u_bary
                world_pt = w0 * v0 + w1 * v1 + w2 * v2

                proj = project_point(world_pt, pose, intr)
                if proj is None:
                    continue

                ix, iy = int(proj[0]), int(proj[1])
                if 0 <= ix < w and 0 <= iy < h:
                    atlas[py, px] += img[iy, ix]
                    atlas_weight[py, px] += 1.0

    # Normalize
    mask = atlas_weight > 0
    for ch in range(3):
        atlas[:, :, ch][mask] /= atlas_weight[mask]

    # Fill empty pixels with nearest neighbor
    for ch in range(3):
        channel = atlas[:, :, ch]
        empty = ~mask
        if empty.any() and mask.any():
            _, indices = distance_transform_edt(empty, return_distances=True,
                                                return_indices=True)
            channel[empty] = channel[indices[0][empty], indices[1][empty]]

    atlas_img = np.clip(atlas, 0, 255).astype(np.uint8)
    print(f"  Assigned {assigned}/{total} faces")
    return atlas_img


def write_obj(path, vertices, normals, uvs, faces, mtl_name="optimized.mtl"):
    """Write OBJ file."""
    print(f"Writing OBJ to {path}...")
    with open(path, 'w') as f:
        f.write("# Optimized mesh with xatlas UV unwrap\n")
        f.write(f"mtllib {mtl_name}\n")
        f.write("usemtl textured_material\n\n")
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        f.write("\n")
        for vt in uvs:
            f.write(f"vt {vt[0]} {vt[1]}\n")
        f.write("\n")
        if normals is not None:
            for vn in normals:
                f.write(f"vn {vn[0]} {vn[1]} {vn[2]}\n")
            f.write("\n")
        for face in faces:
            i0, i1, i2 = face[0]+1, face[1]+1, face[2]+1
            if normals is not None:
                f.write(f"f {i0}/{i0}/{i0} {i1}/{i1}/{i1} {i2}/{i2}/{i2}\n")
            else:
                f.write(f"f {i0}/{i0} {i1}/{i1} {i2}/{i2}\n")
    print(f"  Written: {len(vertices)} vertices, {len(faces)} faces")


def write_mtl(path, texture_filename="texture_optimized.jpg"):
    """Write MTL material file."""
    with open(path, 'w') as f:
        f.write("newmtl textured_material\n")
        f.write("Ka 1.0 1.0 1.0\n")
        f.write("Kd 1.0 1.0 1.0\n")
        f.write(f"map_Kd {texture_filename}\n")


def verify_result(new_uvs, new_faces, atlas_img):
    """Verify optimized UV mapping and compare with original."""
    print("\nVerifying optimized UV mapping...")
    in_range = np.all((new_uvs >= 0) & (new_uvs <= 1), axis=1)
    print(f"  UVs in [0,1]: {in_range.sum()}/{len(new_uvs)} ({in_range.sum()/len(new_uvs)*100:.1f}%)")

    # Overlap check
    res = 1024
    grid = np.zeros((res, res), dtype=np.int32)
    for fi in range(len(new_faces)):
        face = new_faces[fi]
        tri_uvs = new_uvs[face]
        if np.any(tri_uvs < 0) or np.any(tri_uvs > 1):
            continue
        uc = (tri_uvs[:, 0] * (res - 1)).astype(int)
        vc = (tri_uvs[:, 1] * (res - 1)).astype(int)
        grid[vc.min():vc.max()+1, uc.min():uc.max()+1] += 1

    used = grid > 0
    overlapping = grid > 1
    overlap_pct = overlapping.sum() / max(used.sum(), 1) * 100
    print(f"  UV overlaps: {overlapping.sum()} ({overlap_pct:.1f}%)")
    print(f"  Max overlap: {grid.max()}x")

    # Texture sampling
    h, w = atlas_img.shape[:2]
    n = min(5000, len(new_faces))
    idx = np.random.choice(len(new_faces), n, replace=False)
    black = 0
    colors = []
    for i in idx:
        face = new_faces[i]
        centroid = new_uvs[face].mean(axis=0)
        px = int(np.clip(centroid[0], 0, 1) * (w - 1))
        py = int(np.clip(centroid[1], 0, 1) * (h - 1))
        c = atlas_img[py, px]
        colors.append(c)
        if np.all(c < 5):
            black += 1
    colors = np.array(colors)
    black_pct = black / len(colors) * 100

    print(f"  Black texels: {black} ({black_pct:.1f}%)")
    print(f"  Mean color: R={colors[:,0].mean():.1f} G={colors[:,1].mean():.1f} B={colors[:,2].mean():.1f}")

    print(f"\n  {'Metric':<25} {'Original':>12} {'Optimized':>12}")
    print(f"  {'-'*25} {'-'*12} {'-'*12}")
    print(f"  {'UV overlaps %':<25} {'32.7%':>12} {f'{overlap_pct:.1f}%':>12}")
    print(f"  {'Black texels %':<25} {'0.1%':>12} {f'{black_pct:.1f}%':>12}")
    print(f"  {'Max overlap':<25} {'4x':>12} {f'{grid.max()}x':>12}")

    return overlap_pct, grid


def plot_comparison(new_uvs, new_faces, overlap_grid, atlas_img):
    """Generate comparison visualization."""
    fig = plt.figure(figsize=(20, 10))

    # New UV layout
    ax1 = fig.add_subplot(2, 3, 1)
    n_draw = min(5000, len(new_faces))
    draw_idx = np.random.choice(len(new_faces), n_draw, replace=False)
    for idx in draw_idx:
        face = new_faces[idx]
        tri = new_uvs[face]
        tri_c = np.vstack([tri, tri[0]])
        ax1.plot(tri_c[:, 0], tri_c[:, 1], 'b-', linewidth=0.1, alpha=0.3)
    ax1.set_xlim(0, 1); ax1.set_ylim(0, 1)
    ax1.set_aspect('equal')
    ax1.set_title('Optimized UV Layout (xatlas)')

    # UV on texture
    ax2 = fig.add_subplot(2, 3, 2)
    tex_small = Image.fromarray(atlas_img).resize((512, 512))
    ax2.imshow(tex_small, extent=[0, 1, 0, 1], origin='lower')
    for idx in draw_idx[:2000]:
        face = new_faces[idx]
        tri = new_uvs[face]
        tri_c = np.vstack([tri, tri[0]])
        ax2.plot(tri_c[:, 0], tri_c[:, 1], 'w-', linewidth=0.1, alpha=0.4)
    ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
    ax2.set_aspect('equal')
    ax2.set_title('UV on Optimized Texture')

    # Overlap heatmap
    ax3 = fig.add_subplot(2, 3, 3)
    im = ax3.imshow(np.clip(overlap_grid, 0, 5), cmap='hot', origin='lower')
    plt.colorbar(im, ax=ax3, label='Overlap count')
    ax3.set_title('Overlap Heatmap')
    ax3.axis('off')

    # Original vs optimized texture
    ax4 = fig.add_subplot(2, 3, 4)
    orig = Image.open(os.path.join(SCAN_DIR, "texture.jpg")).resize((512, 512))
    ax4.imshow(orig); ax4.set_title('Original Texture'); ax4.axis('off')

    ax5 = fig.add_subplot(2, 3, 5)
    ax5.imshow(Image.fromarray(atlas_img).resize((512, 512)))
    ax5.set_title('Optimized Texture'); ax5.axis('off')

    ax6 = fig.add_subplot(2, 3, 6)
    ax6.hist2d(new_uvs[:, 0], new_uvs[:, 1], bins=64, cmap='viridis')
    ax6.set_xlim(0, 1); ax6.set_ylim(0, 1)
    ax6.set_aspect('equal')
    ax6.set_title('UV Density')

    plt.tight_layout()
    out = os.path.join(ANALYSIS_DIR, "uv_optimization_comparison.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


def main():
    print("=" * 60)
    print("UV OPTIMIZATION WITH XATLAS (Atlas API)")
    print("=" * 60)

    # 1. Parse original OBJ
    print("\nStep 1: Loading original mesh...")
    obj_path = os.path.join(SCAN_DIR, "model.obj")
    vertices, normals, old_uvs, faces = parse_obj(obj_path)
    print(f"  {len(vertices)} verts, {len(normals)} normals, "
          f"{len(old_uvs)} UVs, {len(faces)} faces")

    # 2. Re-unwrap with xatlas Atlas API
    print("\nStep 2: Re-unwrapping with xatlas...")
    new_verts, new_normals, new_uvs, new_faces, vmapping, atlas = \
        unwrap_with_xatlas(vertices, normals, faces)

    # 3. Load camera data
    print("\nStep 3: Loading camera data...")
    intr, poses_data = load_camera_data()

    # 4. Render new texture atlas
    print("\nStep 4: Rendering texture atlas...")
    atlas_img = render_texture_atlas(
        new_verts, new_normals, new_uvs, new_faces, poses_data, intr
    )

    # 5. Save results
    print("\nStep 5: Saving optimized files...")
    tex_path = os.path.join(OUT_DIR, "texture_optimized.jpg")
    Image.fromarray(atlas_img).save(tex_path, quality=95)
    print(f"  Texture: {tex_path}")

    write_obj(os.path.join(OUT_DIR, "model_optimized.obj"),
              new_verts, new_normals, new_uvs, new_faces)
    write_mtl(os.path.join(OUT_DIR, "optimized.mtl"))

    # 6. Verify
    print("\nStep 6: Verification...")
    overlap_pct, overlap_grid = verify_result(new_uvs, new_faces, atlas_img)

    # 7. Comparison plot
    print("\nStep 7: Generating comparison...")
    plot_comparison(new_uvs, new_faces, overlap_grid, atlas_img)

    print("\n" + "=" * 60)
    print("DONE! Files saved to:", OUT_DIR)
    print("=" * 60)


if __name__ == "__main__":
    main()
