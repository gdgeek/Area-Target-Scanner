#!/usr/bin/env python3
"""Render textured mesh using painter's algorithm with face sorting.

Sorts faces back-to-front by depth, then draws them as filled polygons
using PIL. This gives correct occlusion for convex-ish scenes.
"""

import numpy as np
from PIL import Image, ImageDraw
import os

SCAN_DIR = "data/scan_20260317_132224/scan_20260317_132224"
OUT_DIR = "data/scan_analysis"
os.makedirs(OUT_DIR, exist_ok=True)

WIDTH, HEIGHT = 1600, 1200


def load_mesh():
    verts, uvs, faces_v, faces_vt = [], [], [], []
    with open(os.path.join(SCAN_DIR, "model.obj")) as f:
        for line in f:
            if line.startswith("v "):
                p = line.split()
                verts.append([float(p[1]), float(p[2]), float(p[3])])
            elif line.startswith("vt "):
                p = line.split()
                uvs.append([float(p[1]), float(p[2])])
            elif line.startswith("f "):
                parts = line.split()[1:]
                fv = [int(p.split("/")[0]) - 1 for p in parts]
                ft = [int(p.split("/")[1]) - 1 for p in parts]
                faces_v.append(fv)
                faces_vt.append(ft)
    return (np.array(verts, dtype=np.float32),
            np.array(uvs, dtype=np.float32),
            np.array(faces_v, dtype=np.int32),
            np.array(faces_vt, dtype=np.int32))


def sample_face_colors(uvs, faces_vt, tex):
    """Get per-face color by sampling texture at face centroid UV."""
    h, w = tex.shape[:2]
    colors = np.zeros((len(faces_vt), 3), dtype=np.uint8)
    for i in range(len(faces_vt)):
        tri_uv = uvs[faces_vt[i]]
        cu, cv = tri_uv.mean(axis=0)
        px = int(np.clip(cu, 0, 0.999) * w)
        py = int(np.clip(1.0 - cv, 0, 0.999) * h)
        colors[i] = tex[py, px]
    return colors


def compute_face_normals(verts, faces_v):
    v0 = verts[faces_v[:, 0]]
    v1 = verts[faces_v[:, 1]]
    v2 = verts[faces_v[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    lengths[lengths < 1e-10] = 1
    return normals / lengths


def project(verts, eye, target, up=np.array([0, 1, 0])):
    """Project 3D vertices to 2D screen coords using perspective."""
    # View matrix
    fwd = target - eye
    fwd = fwd / np.linalg.norm(fwd)
    right = np.cross(fwd, up)
    right = right / np.linalg.norm(right)
    up2 = np.cross(right, fwd)

    # Transform to camera space
    v_centered = verts - eye
    x = v_centered @ right
    y = v_centered @ up2
    z = v_centered @ fwd  # positive = in front

    # Perspective projection
    fov = 60
    f = 1.0 / np.tan(np.radians(fov / 2))
    aspect = WIDTH / HEIGHT

    # Avoid division by zero
    z_safe = np.where(z > 0.01, z, 0.01)

    sx = (f / aspect * x / z_safe + 1) * 0.5 * WIDTH
    sy = (1 - f * y / z_safe) * 0.5 * HEIGHT

    return sx, sy, z


def render_view(verts, faces_v, face_colors, face_normals, eye, target, title):
    """Render one view using painter's algorithm."""
    sx, sy, z = project(verts, eye, target)

    # Face centroids in camera z
    face_z = z[faces_v].mean(axis=1)

    # Backface culling: skip faces facing away from camera
    view_dir = target - eye
    view_dir = view_dir / np.linalg.norm(view_dir)
    dots = face_normals @ view_dir
    # Keep faces roughly facing the camera (dot > -0.1 for some tolerance)

    # Lighting
    light_dir = np.array([0.3, 0.8, 0.5])
    light_dir = light_dir / np.linalg.norm(light_dir)
    diffuse = np.abs(face_normals @ light_dir)
    shade = np.clip(0.3 + 0.7 * diffuse, 0, 1)

    # Sort faces back-to-front (painter's algorithm)
    order = np.argsort(-face_z)

    # Draw
    img = Image.new('RGB', (WIDTH, HEIGHT), (30, 30, 35))
    draw = ImageDraw.Draw(img)

    drawn = 0
    for fi in order:
        fz = face_z[fi]
        if fz < 0.01:  # behind camera
            continue

        fv = faces_v[fi]
        pts = [(float(sx[fv[j]]), float(sy[fv[j]])) for j in range(3)]

        # Skip if all points off screen
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        if max(xs) < 0 or min(xs) > WIDTH or max(ys) < 0 or min(ys) > HEIGHT:
            continue

        # Apply shading to face color
        s = shade[fi]
        c = face_colors[fi]
        r = int(np.clip(c[0] * s, 0, 255))
        g = int(np.clip(c[1] * s, 0, 255))
        b = int(np.clip(c[2] * s, 0, 255))

        draw.polygon(pts, fill=(r, g, b))
        drawn += 1

    print(f"  {title}: drew {drawn} faces")
    return img


def main():
    print("Loading mesh...")
    verts, uvs, faces_v, faces_vt = load_mesh()
    tex = np.array(Image.open(os.path.join(SCAN_DIR, "texture.jpg")))
    print(f"  {len(verts)} verts, {len(faces_v)} faces")

    print("Sampling face colors...")
    face_colors = sample_face_colors(uvs, faces_vt, tex)

    print("Computing normals...")
    face_normals = compute_face_normals(verts, faces_v)

    # Compute mesh center and extent for camera placement
    center = (verts.min(axis=0) + verts.max(axis=0)) / 2
    extent = (verts.max(axis=0) - verts.min(axis=0)).max()
    dist = extent * 1.2

    print(f"  Center: {center}, extent: {extent:.2f}m, cam dist: {dist:.2f}m")

    views = [
        (center + np.array([0, dist * 0.3, dist]), center, "Front"),
        (center + np.array([dist, dist * 0.3, 0]), center, "Right"),
        (center + np.array([0, dist * 0.3, -dist]), center, "Back"),
        (center + np.array([-dist, dist * 0.3, 0]), center, "Left"),
        (center + np.array([0, dist, 0.01]), center, "Top"),
        (center + np.array([dist * 0.7, dist * 0.3, dist * 0.7]), center, "3/4 View"),
    ]

    print("Rendering views...")
    images = []
    for eye, target, title in views:
        img = render_view(verts, faces_v, face_colors, face_normals,
                          eye.astype(np.float32), target.astype(np.float32), title)
        images.append((img, title))

    # Compose into grid
    grid_w, grid_h = 3, 2
    margin = 40
    cell_w, cell_h = WIDTH, HEIGHT
    canvas = Image.new('RGB',
                       (grid_w * cell_w, grid_h * cell_h + margin),
                       (20, 20, 25))

    from PIL import ImageFont
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 28)
    except Exception:
        font = ImageFont.load_default()

    draw = ImageDraw.Draw(canvas)
    for i, (img, title) in enumerate(images):
        col, row = i % grid_w, i // grid_w
        x, y = col * cell_w, row * cell_h + margin
        canvas.paste(img, (x, y))
        draw.text((x + 10, y + 5), title, fill=(255, 255, 255), font=font)

    draw.text((10, 5), "Textured Mesh Render - 6 Views", fill=(200, 200, 200), font=font)

    out = os.path.join(OUT_DIR, "model_rendered_views.png")
    canvas.save(out)
    print(f"\nSaved: {out}")

    # Hero shot
    eye = center + np.array([dist * 0.8, dist * 0.4, dist * 0.6])
    hero = render_view(verts, faces_v, face_colors, face_normals,
                       eye.astype(np.float32), center.astype(np.float32), "Hero")
    hero_path = os.path.join(OUT_DIR, "model_hero_shot.png")
    hero.save(hero_path)
    print(f"Saved: {hero_path}")


if __name__ == "__main__":
    main()
