"""UV unwrapping module using xatlas for server-side processing.

Takes an OBJ mesh with per-triangle UV (from iPad fast stub) and re-unwraps
it with proper chart segmentation and packing, then re-projects texture from
camera frames.
"""

import json
import logging
import os
import shutil

import numpy as np
import xatlas
from PIL import Image
from scipy.ndimage import distance_transform_edt

logger = logging.getLogger(__name__)

ATLAS_SIZE = 4096


def parse_obj(path):
    """Parse OBJ file into vertices, normals, UVs, and face indices."""
    vertices, normals, uvs = [], [], []
    faces_v, faces_vt = [], []
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
                fv, ft = [], []
                for p in parts:
                    idx = p.split("/")
                    fv.append(int(idx[0]) - 1)
                    ft.append(int(idx[1]) - 1 if len(idx) > 1 and idx[1] else 0)
                faces_v.append(fv)
                faces_vt.append(ft)
    return (
        np.array(vertices, dtype=np.float32),
        np.array(normals, dtype=np.float32) if normals else np.zeros((len(vertices), 3), dtype=np.float32),
        np.array(uvs, dtype=np.float32) if uvs else None,
        np.array(faces_v, dtype=np.uint32),
        np.array(faces_vt, dtype=np.int32) if faces_vt else None,
    )


def unwrap_with_xatlas(vertices, normals, faces):
    """Re-unwrap mesh using xatlas Atlas API with proper chart packing."""
    logger.info("Running xatlas UV unwrap: %d verts, %d faces", len(vertices), len(faces))

    atlas = xatlas.Atlas()
    atlas.add_mesh(vertices, faces, normals)

    chart_opts = xatlas.ChartOptions()
    chart_opts.max_iterations = 4
    chart_opts.normal_deviation_weight = 2.0
    chart_opts.normal_seam_weight = 4.0

    pack_opts = xatlas.PackOptions()
    pack_opts.resolution = ATLAS_SIZE
    pack_opts.padding = 2
    pack_opts.bilinear = True
    pack_opts.bruteForce = False
    pack_opts.create_image = True

    atlas.generate(chart_opts, pack_opts, verbose=False)

    logger.info("xatlas result: %d charts, %dx%d atlas", atlas.chart_count, atlas.width, atlas.height)

    vmapping, indices, new_uvs = atlas[0]
    new_verts = vertices[vmapping]
    new_normals = normals[vmapping] if len(normals) > 0 else None

    return new_verts, new_normals, new_uvs, indices, vmapping


def _vectorized_assign_frames(centers, normals, pose_matrices, intr):
    """Vectorized best-frame assignment for all faces at once.

    For each face, find the camera with highest score = dot(normal, viewDir) / dist^2,
    where the face center must project into the image bounds.

    Returns:
        assignments: (N,) int array, -1 if no valid frame found
    """
    n_faces = len(centers)
    n_frames = len(pose_matrices)

    # Camera positions: (n_frames, 3)
    cam_positions = np.array([p[:3, 3] for p in pose_matrices], dtype=np.float64)

    # Precompute view matrices for projection check
    view_matrices = np.array([np.linalg.inv(p) for p in pose_matrices], dtype=np.float64)

    fx, fy, cx, cy = intr['fx'], intr['fy'], intr['cx'], intr['cy']
    img_w, img_h = intr['width'], intr['height']

    # to_cam: (n_faces, n_frames, 3)
    to_cam = cam_positions[np.newaxis, :, :] - centers[:, np.newaxis, :]

    # dist: (n_faces, n_frames)
    dist_sq = np.sum(to_cam ** 2, axis=2)
    dist = np.sqrt(dist_sq)
    dist_safe = np.where(dist > 1e-10, dist, 1.0)

    # view_dir: (n_faces, n_frames, 3)
    view_dir = to_cam / dist_safe[:, :, np.newaxis]

    # dot product with face normals: (n_faces, n_frames)
    dot = np.sum(normals[:, np.newaxis, :] * view_dir, axis=2)

    # score = dot / dist^2, only where dot > 0
    score = np.where(dot > 0, dot / np.maximum(dist_sq, 1e-20), -1.0)

    # Projection check: transform centers to camera space for each frame
    # centers_h: (n_faces, 4) homogeneous
    centers_h = np.hstack([centers, np.ones((n_faces, 1), dtype=np.float64)])

    # For each frame, project all centers
    for i in range(n_frames):
        # p_cam: (n_faces, 4)
        p_cam = (view_matrices[i] @ centers_h.T).T
        # Must have negative Z (camera looks along -Z)
        behind = p_cam[:, 2] >= 0
        neg_z = np.where(behind, 1.0, -p_cam[:, 2])
        px = fx * (p_cam[:, 0] / neg_z) + cx
        py = fy * (-p_cam[:, 1] / neg_z) + cy
        out_of_bounds = behind | (px < 0) | (px >= img_w) | (py < 0) | (py >= img_h)
        score[out_of_bounds, i] = -1.0

    # Best frame per face
    assignments = np.argmax(score, axis=1).astype(np.int32)
    # Mark faces with no valid frame
    best_scores = score[np.arange(n_faces), assignments]
    assignments[best_scores <= 0] = -1

    return assignments


def render_texture_atlas(new_verts, new_uvs, new_faces, scan_dir, intr, poses_data):
    """Render texture atlas with vectorized frame assignment and per-face rasterization."""
    import time
    frames = poses_data["frames"]
    n_frames = len(frames)
    logger.info("Rendering texture atlas from %d camera frames (vectorized)...", n_frames)

    # Precompute pose and view matrices
    pose_matrices = []
    view_matrices = []
    for frame in frames:
        t = np.array(frame["transform"], dtype=np.float64).reshape(4, 4, order='F')
        pose_matrices.append(t)
        view_matrices.append(np.linalg.inv(t))

    # Load all images
    images = []
    for frame in frames:
        img_path = os.path.join(scan_dir, frame["imageFile"])
        img = np.array(Image.open(img_path).convert("RGB"), dtype=np.float64)
        images.append(img)

    fx, fy, cx, cy = intr['fx'], intr['fy'], intr['cx'], intr['cy']
    img_w, img_h = intr['width'], intr['height']

    # Step 1: Batch compute face centers and normals
    t0 = time.time()
    v0 = new_verts[new_faces[:, 0]].astype(np.float64)
    v1 = new_verts[new_faces[:, 1]].astype(np.float64)
    v2 = new_verts[new_faces[:, 2]].astype(np.float64)
    centers = (v0 + v1 + v2) / 3.0
    edge1 = v1 - v0
    edge2 = v2 - v0
    cross = np.cross(edge1, edge2)
    cross_len = np.linalg.norm(cross, axis=1, keepdims=True)
    valid_faces = (cross_len.ravel() > 1e-10)
    normals = np.zeros_like(cross)
    normals[valid_faces] = cross[valid_faces] / cross_len[valid_faces]
    logger.info("  Computed %d face normals in %.1fs", valid_faces.sum(), time.time() - t0)

    # Step 2: Vectorized frame assignment
    t1 = time.time()
    assignments = _vectorized_assign_frames(centers, normals, pose_matrices, intr)
    assigned_count = (assignments >= 0).sum()
    logger.info("  Assigned %d/%d faces in %.1fs", assigned_count, len(new_faces), time.time() - t1)

    # Step 3: Per-face rasterization with vectorized pixel processing
    t2 = time.time()
    atlas = np.zeros((ATLAS_SIZE, ATLAS_SIZE, 3), dtype=np.float64)
    atlas_weight = np.zeros((ATLAS_SIZE, ATLAS_SIZE), dtype=np.float64)

    total = len(new_faces)
    for fi in range(total):
        if fi % 20000 == 0:
            logger.info("  Rasterizing face %d/%d...", fi, total)

        if assignments[fi] < 0 or not valid_faces[fi]:
            continue

        frame_idx = assignments[fi]
        view = view_matrices[frame_idx]
        img = images[frame_idx]
        h, w = img.shape[:2]

        face = new_faces[fi]
        uv0, uv1, uv2 = new_uvs[face[0]], new_uvs[face[1]], new_uvs[face[2]]
        px_uvs = np.array([uv0, uv1, uv2], dtype=np.float64) * (ATLAS_SIZE - 1)

        u_min = max(0, int(np.floor(px_uvs[:, 0].min())))
        u_max = min(ATLAS_SIZE - 1, int(np.ceil(px_uvs[:, 0].max())))
        v_min = max(0, int(np.floor(px_uvs[:, 1].min())))
        v_max = min(ATLAS_SIZE - 1, int(np.ceil(px_uvs[:, 1].max())))

        if u_min >= u_max or v_min >= v_max:
            continue

        # Vectorized barycentric for all pixels in bounding box
        a, b, c = px_uvs[0], px_uvs[1], px_uvs[2]
        v0b, v1b = c - a, b - a
        dot00 = np.dot(v0b, v0b)
        dot01 = np.dot(v0b, v1b)
        dot11 = np.dot(v1b, v1b)
        denom = dot00 * dot11 - dot01 * dot01
        if abs(denom) < 1e-10:
            continue
        inv_denom = 1.0 / denom

        # Create pixel grid
        px_range = np.arange(u_min, u_max + 1, dtype=np.float64)
        py_range = np.arange(v_min, v_max + 1, dtype=np.float64)
        grid_x, grid_y = np.meshgrid(px_range, py_range)
        # (n_pixels, 2)
        pts_x = grid_x.ravel()
        pts_y = grid_y.ravel()

        v2b_x = pts_x - a[0]
        v2b_y = pts_y - a[1]
        dot02 = v0b[0] * v2b_x + v0b[1] * v2b_y
        dot12 = v1b[0] * v2b_x + v1b[1] * v2b_y
        u_bary = (dot11 * dot02 - dot01 * dot12) * inv_denom
        v_bary = (dot00 * dot12 - dot01 * dot02) * inv_denom

        # Filter pixels inside triangle
        inside = (u_bary >= -0.01) & (v_bary >= -0.01) & ((u_bary + v_bary) <= 1.01)
        if not inside.any():
            continue

        u_b = u_bary[inside]
        v_b = v_bary[inside]
        w0 = 1.0 - u_b - v_b
        w1 = v_b
        w2 = u_b

        # 3D world points: (n_inside, 3)
        fv0, fv1, fv2 = v0[fi], v1[fi], v2[fi]
        world_pts = w0[:, np.newaxis] * fv0 + w1[:, np.newaxis] * fv1 + w2[:, np.newaxis] * fv2

        # Project to camera: vectorized
        ones = np.ones((len(world_pts), 1), dtype=np.float64)
        world_h = np.hstack([world_pts, ones])  # (n, 4)
        p_cam = (view @ world_h.T).T  # (n, 4)

        visible = p_cam[:, 2] < 0
        if not visible.any():
            continue

        neg_z = -p_cam[visible, 2]
        ix = (fx * (p_cam[visible, 0] / neg_z) + cx).astype(np.int32)
        iy = (fy * (-p_cam[visible, 1] / neg_z) + cy).astype(np.int32)

        in_bounds = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
        if not in_bounds.any():
            continue

        # Atlas pixel coords for valid pixels
        atlas_px = pts_x[inside][visible][in_bounds].astype(np.int32)
        atlas_py = pts_y[inside][visible][in_bounds].astype(np.int32)
        sample_ix = ix[in_bounds]
        sample_iy = iy[in_bounds]

        # Sample from image and accumulate
        colors = img[sample_iy, sample_ix]  # (n_valid, 3)
        # Use np.add.at for safe accumulation (handles duplicate indices)
        np.add.at(atlas, (atlas_py, atlas_px), colors)
        np.add.at(atlas_weight, (atlas_py, atlas_px), 1.0)

    logger.info("  Rasterized in %.1fs", time.time() - t2)

    # Normalize
    mask = atlas_weight > 0
    for ch in range(3):
        atlas[:, :, ch][mask] /= atlas_weight[mask]

    # Fill empty pixels with nearest neighbor
    for ch in range(3):
        channel = atlas[:, :, ch]
        empty = ~mask
        if empty.any() and mask.any():
            _, indices = distance_transform_edt(empty, return_distances=True, return_indices=True)
            channel[empty] = channel[indices[0][empty], indices[1][empty]]

    filled_pct = mask.sum() / mask.size * 100
    logger.info("Atlas rendered: %.1f%% pixels filled (total %.1fs)", filled_pct, time.time() - t0)
    return np.clip(atlas, 0, 255).astype(np.uint8)


def write_obj(path, vertices, normals, uvs, faces, mtl_name="model.mtl"):
    """Write OBJ file with vertices, normals, UVs, and faces."""
    with open(path, 'w') as f:
        f.write("# UV-unwrapped mesh (xatlas)\n")
        f.write(f"mtllib {mtl_name}\n")
        f.write("usemtl textured_material\n\n")
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        f.write("\n")
        for vt in uvs:
            f.write(f"vt {vt[0]} {vt[1]}\n")
        f.write("\n")
        if normals is not None and len(normals) > 0:
            for vn in normals:
                f.write(f"vn {vn[0]} {vn[1]} {vn[2]}\n")
            f.write("\n")
        for face in faces:
            i0, i1, i2 = face[0] + 1, face[1] + 1, face[2] + 1
            if normals is not None and len(normals) > 0:
                f.write(f"f {i0}/{i0}/{i0} {i1}/{i1}/{i1} {i2}/{i2}/{i2}\n")
            else:
                f.write(f"f {i0}/{i0} {i1}/{i1} {i2}/{i2}\n")


def write_mtl(path, texture_filename="texture.jpg"):
    """Write MTL material file."""
    with open(path, 'w') as f:
        f.write("newmtl textured_material\n")
        f.write("Ka 1.0 1.0 1.0\n")
        f.write("Kd 1.0 1.0 1.0\n")
        f.write(f"map_Kd {texture_filename}\n")


def uv_unwrap_scan(scan_dir):
    """Run full UV unwrap pipeline on a scan directory.

    Reads model.obj, runs xatlas UV unwrap, re-projects texture from camera
    frames, and overwrites model.obj + texture.jpg + model.mtl in place.

    Args:
        scan_dir: Path to extracted scan directory containing model.obj,
                  texture.jpg, poses.json, intrinsics.json, images/

    Returns:
        dict with stats: chart_count, vertex_count, face_count
    """
    obj_path = os.path.join(scan_dir, "model.obj")
    poses_path = os.path.join(scan_dir, "poses.json")
    intrinsics_path = os.path.join(scan_dir, "intrinsics.json")

    # Validate required files
    for p in [obj_path, poses_path, intrinsics_path]:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"UV unwrap requires: {p}")

    # 1. Parse OBJ
    logger.info("UV unwrap: loading mesh from %s", obj_path)
    vertices, normals, old_uvs, faces_v, faces_vt = parse_obj(obj_path)
    logger.info("  %d vertices, %d faces", len(vertices), len(faces_v))

    # 2. Run xatlas
    new_verts, new_normals, new_uvs, new_faces, vmapping = unwrap_with_xatlas(
        vertices, normals, faces_v
    )

    # 3. Load camera data
    with open(intrinsics_path) as f:
        intr = json.load(f)
    with open(poses_path) as f:
        poses_data = json.load(f)

    # 4. Render texture atlas
    atlas_img = render_texture_atlas(new_verts, new_uvs, new_faces, scan_dir, intr, poses_data)

    # 5. Backup originals and write new files
    backup_dir = os.path.join(scan_dir, "_backup_pre_unwrap")
    os.makedirs(backup_dir, exist_ok=True)
    for fname in ["model.obj", "texture.jpg", "model.mtl"]:
        src = os.path.join(scan_dir, fname)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(backup_dir, fname))

    # Write new OBJ
    write_obj(obj_path, new_verts, new_normals, new_uvs, new_faces)
    logger.info("  Written: %s (%d verts, %d faces)", obj_path, len(new_verts), len(new_faces))

    # Write new texture
    tex_path = os.path.join(scan_dir, "texture.jpg")
    Image.fromarray(atlas_img).save(tex_path, quality=95)
    logger.info("  Written: %s (%dx%d)", tex_path, atlas_img.shape[1], atlas_img.shape[0])

    # Write new MTL
    mtl_path = os.path.join(scan_dir, "model.mtl")
    write_mtl(mtl_path)

    stats = {
        "vertex_count": len(new_verts),
        "face_count": len(new_faces),
        "atlas_size": ATLAS_SIZE,
    }
    logger.info("UV unwrap complete: %s", stats)
    return stats
