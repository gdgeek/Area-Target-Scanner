#!/usr/bin/env python3
"""Verify model + texture mapping correctness."""

import numpy as np
import trimesh
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import json

SCAN_DIR = "data/scan_20260317_132224/scan_20260317_132224"
OUT_DIR = "data/scan_analysis"
os.makedirs(OUT_DIR, exist_ok=True)


def parse_obj_uvs(path):
    """Parse UV coordinates and face UV indices from OBJ."""
    uvs = []
    face_uv_indices = []
    with open(path) as f:
        for line in f:
            if line.startswith("vt "):
                parts = line.strip().split()
                uvs.append((float(parts[1]), float(parts[2])))
            elif line.startswith("f "):
                parts = line.strip().split()[1:]
                fi = []
                for p in parts:
                    indices = p.split("/")
                    if len(indices) >= 2 and indices[1]:
                        fi.append(int(indices[1]) - 1)  # 0-indexed
                face_uv_indices.append(fi)
    return np.array(uvs), face_uv_indices


def check_uv_range(uvs):
    """Check if UVs are in valid [0,1] range."""
    print("=" * 60)
    print("UV COORDINATE VALIDATION")
    print("=" * 60)
    print(f"  Total UV coords:  {len(uvs):,}")
    print(f"  U range:          [{uvs[:,0].min():.6f}, {uvs[:,0].max():.6f}]")
    print(f"  V range:          [{uvs[:,1].min():.6f}, {uvs[:,1].max():.6f}]")

    in_range = np.all((uvs >= 0) & (uvs <= 1), axis=1)
    pct = in_range.sum() / len(uvs) * 100
    print(f"  In [0,1] range:   {in_range.sum():,} / {len(uvs):,} ({pct:.1f}%)")

    out_of_range = ~in_range
    if out_of_range.any():
        print(f"  OUT OF RANGE:     {out_of_range.sum():,} UVs")
        bad = uvs[out_of_range]
        print(f"    U outliers:     min={bad[:,0].min():.4f}, max={bad[:,0].max():.4f}")
        print(f"    V outliers:     min={bad[:,1].min():.4f}, max={bad[:,1].max():.4f}")
    else:
        print("  ✓ All UVs within valid range")

    return pct


def check_uv_coverage(uvs, face_uv_indices):
    """Check how well UVs cover the texture space."""
    print("\n" + "=" * 60)
    print("UV COVERAGE ANALYSIS")
    print("=" * 60)

    # Create a coverage map
    resolution = 512
    coverage = np.zeros((resolution, resolution), dtype=bool)

    for fi in face_uv_indices:
        if len(fi) < 3:
            continue
        tri_uvs = uvs[fi[:3]]
        # Rasterize triangle into coverage map
        u_coords = np.clip(tri_uvs[:, 0], 0, 1) * (resolution - 1)
        v_coords = np.clip(tri_uvs[:, 1], 0, 1) * (resolution - 1)

        # Simple bounding box fill for coverage estimation
        u_min, u_max = int(u_coords.min()), int(u_coords.max())
        v_min, v_max = int(v_coords.min()), int(v_coords.max())
        coverage[v_min:v_max+1, u_min:u_max+1] = True

    pct = coverage.sum() / coverage.size * 100
    print(f"  UV space coverage: {pct:.1f}%")
    if pct < 30:
        print("  ⚠ Low coverage — texture space is underutilized")
    elif pct < 70:
        print("  ~ Moderate coverage")
    else:
        print("  ✓ Good coverage")

    return coverage


def check_texture_sampling(uvs, face_uv_indices, texture_path):
    """Sample texture at UV positions and check for valid colors."""
    print("\n" + "=" * 60)
    print("TEXTURE SAMPLING VALIDATION")
    print("=" * 60)

    tex = Image.open(texture_path)
    tex_arr = np.array(tex)
    h, w = tex_arr.shape[:2]
    print(f"  Texture size:     {w} x {h}")

    # Sample at face centroids
    n_samples = min(5000, len(face_uv_indices))
    indices = np.random.choice(len(face_uv_indices), n_samples, replace=False)

    sampled_colors = []
    black_count = 0
    for idx in indices:
        fi = face_uv_indices[idx]
        if len(fi) < 3:
            continue
        tri_uvs = uvs[fi[:3]]
        centroid = tri_uvs.mean(axis=0)
        px = int(np.clip(centroid[0], 0, 1) * (w - 1))
        py = int(np.clip(1 - centroid[1], 0, 1) * (h - 1))  # flip V
        color = tex_arr[py, px]
        sampled_colors.append(color)
        if np.all(color < 5):
            black_count += 1

    sampled_colors = np.array(sampled_colors)
    black_pct = black_count / len(sampled_colors) * 100

    print(f"  Sampled faces:    {len(sampled_colors):,}")
    print(f"  Mean color:       R={sampled_colors[:,0].mean():.1f} "
          f"G={sampled_colors[:,1].mean():.1f} B={sampled_colors[:,2].mean():.1f}")
    print(f"  Black texels:     {black_count} ({black_pct:.1f}%)")

    if black_pct > 50:
        print("  ✗ PROBLEM: Most faces sample black — UV mapping likely broken")
    elif black_pct > 20:
        print("  ⚠ WARNING: Many faces sample black — partial UV issues")
    else:
        print("  ✓ Texture sampling looks reasonable")

    # Color variance check
    var = sampled_colors.std(axis=0).mean()
    print(f"  Color std dev:    {var:.1f}")
    if var < 5:
        print("  ⚠ Very low color variance — texture may be uniform/blank")
    else:
        print("  ✓ Good color variance")

    return sampled_colors, black_pct


def check_uv_overlaps(uvs, face_uv_indices):
    """Check for UV triangle overlaps (indicates bad unwrap)."""
    print("\n" + "=" * 60)
    print("UV OVERLAP CHECK")
    print("=" * 60)

    # Sample-based overlap detection using a raster grid
    resolution = 1024
    grid = np.zeros((resolution, resolution), dtype=np.int32)

    for fi in face_uv_indices:
        if len(fi) < 3:
            continue
        tri_uvs = uvs[fi[:3]]
        if np.any(tri_uvs < 0) or np.any(tri_uvs > 1):
            continue
        u_coords = (tri_uvs[:, 0] * (resolution - 1)).astype(int)
        v_coords = (tri_uvs[:, 1] * (resolution - 1)).astype(int)
        u_min, u_max = u_coords.min(), u_coords.max()
        v_min, v_max = v_coords.min(), v_coords.max()
        grid[v_min:v_max+1, u_min:u_max+1] += 1

    used = grid > 0
    overlapping = grid > 1
    if used.sum() > 0:
        overlap_pct = overlapping.sum() / used.sum() * 100
    else:
        overlap_pct = 0

    print(f"  Used texels:      {used.sum():,} / {grid.size:,}")
    print(f"  Overlapping:      {overlapping.sum():,} ({overlap_pct:.1f}% of used)")
    max_overlap = grid.max()
    print(f"  Max overlap:      {max_overlap}x")

    if overlap_pct > 30:
        print("  ⚠ Significant UV overlaps — may cause texture bleeding")
    elif overlap_pct > 5:
        print("  ~ Some overlaps present (common for projection-based mapping)")
    else:
        print("  ✓ Minimal overlaps")

    return grid, overlap_pct


def plot_results(uvs, face_uv_indices, coverage, overlap_grid, sampled_colors, texture_path):
    """Generate visualization of UV mapping quality."""
    fig = plt.figure(figsize=(20, 15))

    # 1. UV layout
    ax1 = fig.add_subplot(2, 3, 1)
    # Draw UV triangles (subsample for speed)
    n_draw = min(5000, len(face_uv_indices))
    draw_idx = np.random.choice(len(face_uv_indices), n_draw, replace=False)
    for idx in draw_idx:
        fi = face_uv_indices[idx]
        if len(fi) < 3:
            continue
        tri = uvs[fi[:3]]
        tri_closed = np.vstack([tri, tri[0]])
        ax1.plot(tri_closed[:, 0], tri_closed[:, 1], 'b-', linewidth=0.1, alpha=0.3)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_aspect('equal')
    ax1.set_title(f'UV Layout ({n_draw} faces sampled)')
    ax1.set_xlabel('U')
    ax1.set_ylabel('V')

    # 2. UV layout overlaid on texture
    ax2 = fig.add_subplot(2, 3, 2)
    tex = Image.open(texture_path).resize((512, 512))
    ax2.imshow(tex, extent=[0, 1, 0, 1], origin='lower')
    for idx in draw_idx[:2000]:
        fi = face_uv_indices[idx]
        if len(fi) < 3:
            continue
        tri = uvs[fi[:3]]
        tri_closed = np.vstack([tri, tri[0]])
        ax2.plot(tri_closed[:, 0], tri_closed[:, 1], 'w-', linewidth=0.1, alpha=0.4)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_aspect('equal')
    ax2.set_title('UV Layout on Texture')

    # 3. Coverage map
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.imshow(coverage, cmap='Greens', origin='lower')
    ax3.set_title(f'UV Coverage ({coverage.sum()/coverage.size*100:.1f}%)')
    ax3.axis('off')

    # 4. Overlap heatmap
    ax4 = fig.add_subplot(2, 3, 4)
    im = ax4.imshow(np.clip(overlap_grid, 0, 5), cmap='hot', origin='lower')
    plt.colorbar(im, ax=ax4, label='Overlap count')
    ax4.set_title('UV Overlap Heatmap')
    ax4.axis('off')

    # 5. Sampled color distribution
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.hist(sampled_colors[:, 0], bins=50, alpha=0.5, color='red', label='R')
    ax5.hist(sampled_colors[:, 1], bins=50, alpha=0.5, color='green', label='G')
    ax5.hist(sampled_colors[:, 2], bins=50, alpha=0.5, color='blue', label='B')
    ax5.set_title('Sampled Texture Color Distribution')
    ax5.set_xlabel('Intensity')
    ax5.set_ylabel('Count')
    ax5.legend()

    # 6. UV density histogram
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.hist2d(uvs[:, 0], uvs[:, 1], bins=64, cmap='viridis')
    ax6.set_xlim(0, 1)
    ax6.set_ylim(0, 1)
    ax6.set_aspect('equal')
    ax6.set_title('UV Density')
    ax6.set_xlabel('U')
    ax6.set_ylabel('V')

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "texture_verification.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"\nSaved: {out}")


def check_mtl_consistency():
    """Verify MTL file references correct texture."""
    print("\n" + "=" * 60)
    print("MATERIAL FILE CHECK")
    print("=" * 60)
    mtl_path = os.path.join(SCAN_DIR, "model.mtl")
    with open(mtl_path) as f:
        content = f.read()
    print(f"  MTL content:\n{content}")

    if "map_Kd texture.jpg" in content:
        tex_exists = os.path.exists(os.path.join(SCAN_DIR, "texture.jpg"))
        print(f"  Texture file exists: {tex_exists}")
        if tex_exists:
            print("  ✓ MTL correctly references texture.jpg")
        else:
            print("  ✗ MTL references texture.jpg but file is missing!")
    else:
        print("  ⚠ No diffuse texture map found in MTL")


def check_face_uv_consistency(uvs, face_uv_indices):
    """Check that all face UV indices are valid."""
    print("\n" + "=" * 60)
    print("FACE-UV INDEX CONSISTENCY")
    print("=" * 60)
    max_idx = len(uvs) - 1
    invalid = 0
    degenerate = 0
    for fi in face_uv_indices:
        for idx in fi:
            if idx < 0 or idx > max_idx:
                invalid += 1
        if len(fi) >= 3:
            tri_uvs = uvs[fi[:3]]
            area = 0.5 * abs(
                (tri_uvs[1][0]-tri_uvs[0][0])*(tri_uvs[2][1]-tri_uvs[0][1]) -
                (tri_uvs[2][0]-tri_uvs[0][0])*(tri_uvs[1][1]-tri_uvs[0][1])
            )
            if area < 1e-10:
                degenerate += 1

    print(f"  Total faces:      {len(face_uv_indices):,}")
    print(f"  Invalid UV refs:  {invalid}")
    print(f"  Degenerate UVs:   {degenerate} ({degenerate/len(face_uv_indices)*100:.1f}%)")
    if invalid == 0:
        print("  ✓ All UV indices valid")
    else:
        print("  ✗ Found invalid UV index references!")
    if degenerate > len(face_uv_indices) * 0.1:
        print("  ⚠ Many degenerate UV triangles (zero area)")


def main():
    print("Verifying model + texture mapping...\n")

    obj_path = os.path.join(SCAN_DIR, "model.obj")
    tex_path = os.path.join(SCAN_DIR, "texture.jpg")

    # Parse UVs
    print("Parsing OBJ file...")
    uvs, face_uv_indices = parse_obj_uvs(obj_path)
    print(f"Loaded {len(uvs):,} UVs, {len(face_uv_indices):,} faces\n")

    # Run checks
    check_mtl_consistency()
    check_uv_range(uvs)
    check_face_uv_consistency(uvs, face_uv_indices)
    coverage = check_uv_coverage(uvs, face_uv_indices)
    sampled_colors, black_pct = check_texture_sampling(uvs, face_uv_indices, tex_path)
    overlap_grid, overlap_pct = check_uv_overlaps(uvs, face_uv_indices)

    # Summary
    print("\n" + "=" * 60)
    print("OVERALL VERDICT")
    print("=" * 60)
    issues = []
    if black_pct > 20:
        issues.append("High black texel ratio")
    if overlap_pct > 30:
        issues.append("Significant UV overlaps")
    coverage_pct = coverage.sum() / coverage.size * 100
    if coverage_pct < 30:
        issues.append("Low UV coverage")

    if not issues:
        print("  ✓ Model + texture mapping looks CORRECT")
    else:
        print("  ⚠ Potential issues found:")
        for iss in issues:
            print(f"    - {iss}")

    # Generate visualization
    print("\nGenerating verification plots...")
    plot_results(uvs, face_uv_indices, coverage, overlap_grid, sampled_colors, tex_path)


if __name__ == "__main__":
    main()
