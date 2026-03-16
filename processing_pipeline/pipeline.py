"""Main reconstruction pipeline orchestrating all processing steps."""

from __future__ import annotations

import json
import logging
import os
import shutil
from datetime import datetime, timezone
from typing import List

import numpy as np
import open3d as o3d

from .models import (
    FeatureDatabase,
    KeyframeData,
    ProcessedCloud,
    TexturedMesh,
)


class ReconstructionPipeline:
    """End-to-end pipeline for processing scan data into an Area Target asset bundle.

    Orchestrates point cloud preprocessing, mesh reconstruction, simplification,
    texture mapping, feature extraction, and asset bundle export.
    """

    def process_point_cloud(self, input_ply: str) -> ProcessedCloud:
        """Preprocess a raw point cloud: denoise, downsample, and estimate normals.

        Applies statistical outlier removal (nb_neighbors=20, std_ratio=2.0),
        voxel downsampling (voxel_size=0.02m), KNN normal estimation (K=30),
        and consistent normal orientation (K=15).

        Args:
            input_ply: Path to the input PLY file.

        Returns:
            Preprocessed point cloud with normals and colors.

        Raises:
            ValueError: If the PLY file is invalid or contains fewer than 1000 points.
        """
        import os

        import numpy as np

        # --- Input validation ---
        if not os.path.isfile(input_ply):
            raise ValueError(f"PLY file does not exist: {input_ply}")

        cloud = o3d.io.read_point_cloud(input_ply)
        if cloud is None or cloud.is_empty():
            raise ValueError(f"Invalid or empty PLY file: {input_ply}")

        if len(cloud.points) < 1000:
            raise ValueError(
                f"Point cloud has {len(cloud.points)} points, minimum 1000 required"
            )

        # --- Step 1: Statistical outlier removal ---
        cloud, _ = cloud.remove_statistical_outlier(
            nb_neighbors=20, std_ratio=2.0
        )

        # --- Step 2: Voxel downsampling ---
        cloud = cloud.voxel_down_sample(voxel_size=0.02)

        # --- Step 3: Normal estimation (KNN K=30) ---
        cloud.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30)
        )

        # --- Step 4: Consistent normal orientation (K=15) ---
        cloud.orient_normals_consistent_tangent_plane(k=15)

        # --- Build result ---
        points = np.asarray(cloud.points)
        normals = np.asarray(cloud.normals)
        colors = (
            np.asarray(cloud.colors)
            if cloud.has_colors()
            else np.zeros((len(points), 3))
        )

        return ProcessedCloud(
            points=points,
            normals=normals,
            colors=colors,
            point_count=len(points),
        )

    def reconstruct_mesh(self, cloud: ProcessedCloud) -> o3d.geometry.TriangleMesh:
        """Reconstruct a triangle mesh from a preprocessed point cloud.

        Uses Poisson surface reconstruction (depth=9) with density-based
        cropping (removes vertices below the 1st percentile density).
        Falls back to depth=7 on failure, and raises on second failure.

        Args:
            cloud: Preprocessed point cloud with normals.

        Returns:
            Reconstructed triangle mesh.

        Raises:
            RuntimeError: If mesh reconstruction fails after retry.
        """
        import numpy as np

        # Convert ProcessedCloud back to Open3D PointCloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cloud.points)
        pcd.normals = o3d.utility.Vector3dVector(cloud.normals)
        if cloud.colors is not None and len(cloud.colors) > 0:
            pcd.colors = o3d.utility.Vector3dVector(cloud.colors)

        # Try Poisson reconstruction, first at depth=9, then depth=7 on failure
        for depth in (9, 7):
            try:
                mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                    pcd, depth=depth, width=0, scale=1.1, linear_fit=False
                )
                densities = np.asarray(densities)

                if len(densities) == 0 or len(mesh.vertices) == 0:
                    raise RuntimeError(
                        f"Poisson reconstruction at depth={depth} produced an empty mesh"
                    )

                # Density-based cropping: remove vertices below 1st percentile
                density_threshold = np.quantile(densities, 0.01)
                vertices_to_remove = np.where(densities < density_threshold)[0]
                mesh.remove_vertices_by_index(vertices_to_remove.tolist())

                # Clean up: remove degenerate triangles and unreferenced vertices
                mesh.remove_degenerate_triangles()
                mesh.remove_unreferenced_vertices()

                # Validate the result
                if len(mesh.triangles) == 0:
                    raise RuntimeError(
                        f"Poisson reconstruction at depth={depth} produced "
                        f"a mesh with no triangles after cleanup"
                    )

                return mesh

            except Exception as exc:
                if depth == 7:
                    # Both attempts failed — produce error report
                    raise RuntimeError(
                        f"Mesh reconstruction failed after retry (depth=9 and depth=7). "
                        f"The point cloud may be too noisy or sparse for surface reconstruction. "
                        f"Please consider rescanning the area with better coverage. "
                        f"Last error: {exc}"
                    ) from exc
                # depth=9 failed, will retry with depth=7
                continue

    def simplify_mesh(
        self,
        mesh: o3d.geometry.TriangleMesh,
        target_faces: int = 50000,
    ) -> o3d.geometry.TriangleMesh:
        """Simplify a mesh to a target face count using quadric error metrics.

        Removes unreferenced vertices and degenerate triangles after
        simplification.

        Args:
            mesh: Input triangle mesh.
            target_faces: Maximum number of triangles in the output.

        Returns:
            Simplified triangle mesh with at most *target_faces* triangles.
        """
        import numpy as np

        # Step 1: Quadric decimation if face count exceeds target
        if len(mesh.triangles) > target_faces:
            mesh = mesh.simplify_quadric_decimation(
                target_number_of_triangles=target_faces
            )

        # Step 2: Remove unreferenced vertices
        mesh.remove_unreferenced_vertices()

        # Step 3: Remove degenerate triangles
        mesh.remove_degenerate_triangles()

        # Step 4: Verify constraint
        assert len(mesh.triangles) <= target_faces, (
            f"Simplified mesh has {len(mesh.triangles)} triangles, "
            f"expected at most {target_faces}"
        )

        return mesh

    def generate_texture(
        self,
        mesh: o3d.geometry.TriangleMesh,
        images: List[dict],
    ) -> TexturedMesh:
        """Project images onto the mesh surface to generate a texture atlas.

        Tries MVS-Texturing first; falls back to a simple vertex-color
        projection when the external tool is unavailable.  Applies global
        colour correction and records a quality score.

        Args:
            mesh: Simplified triangle mesh.
            images: List of dicts each containing ``"path"`` (image file path)
                and ``"pose"`` (4x4 camera-to-world matrix).

        Returns:
            Textured mesh with associated texture and material files.
        """
        import logging
        import os
        import shutil
        import tempfile

        import numpy as np

        logger = logging.getLogger(__name__)

        # ---- Prepare a working directory ----
        work_dir = tempfile.mkdtemp(prefix="texture_")
        texture_file = os.path.join(work_dir, "texture_atlas.png")
        material_file = os.path.join(work_dir, "mesh.mtl")
        obj_file = os.path.join(work_dir, "mesh.obj")

        # Save mesh to OBJ for potential MVS-Texturing use
        o3d.io.write_triangle_mesh(obj_file, mesh)

        textured_mesh_result: o3d.geometry.TriangleMesh = mesh
        quality_score: float = 0.0

        # ---- Try MVS-Texturing ----
        mvs_available = shutil.which("texrecon") is not None
        mvs_success = False

        if mvs_available:
            try:
                mvs_success = self._run_mvs_texturing(
                    obj_file, images, work_dir, logger
                )
            except Exception as exc:
                logger.warning("MVS-Texturing failed: %s. Using fallback.", exc)
                mvs_success = False

        if not mvs_success:
            # ---- Fallback: vertex-colour projection ----
            logger.info("Using fallback vertex-colour texture projection.")
            self._fallback_texture_projection(
                mesh, images, texture_file, material_file, logger
            )

        # ---- Global colour correction ----
        if os.path.isfile(texture_file):
            self._apply_color_correction(texture_file, logger)

        # ---- Compute quality score ----
        quality_score = self._compute_texture_quality(texture_file, logger)

        # ---- Quality warning ----
        quality_threshold = 0.3
        if quality_score < quality_threshold:
            logger.warning(
                "Texture quality score %.2f is below threshold %.2f. "
                "The asset bundle will include a quality warning.",
                quality_score,
                quality_threshold,
            )

        return TexturedMesh(
            mesh=textured_mesh_result,
            texture_file=texture_file,
            material_file=material_file,
            quality_score=quality_score,
        )

    # ------------------------------------------------------------------
    # Private helpers for generate_texture
    # ------------------------------------------------------------------

    def _run_mvs_texturing(
        self,
        obj_file: str,
        images: List[dict],
        work_dir: str,
        logger,
    ) -> bool:
        """Attempt to run MVS-Texturing (texrecon) on the mesh.

        Returns True on success, False otherwise.
        """
        import glob
        import os
        import shutil
        import subprocess

        import numpy as np

        # Write camera files expected by texrecon
        scene_dir = os.path.join(work_dir, "scene")
        os.makedirs(scene_dir, exist_ok=True)

        for idx, img_info in enumerate(images):
            img_path = img_info["path"]
            pose = np.asarray(img_info["pose"], dtype=np.float64)

            # texrecon expects .cam files alongside images
            cam_file = os.path.join(scene_dir, f"image{idx:04d}.cam")
            R = pose[:3, :3]
            t = pose[:3, 3]
            with open(cam_file, "w") as f:
                f.write(f"{t[0]} {t[1]} {t[2]} ")
                for row in R:
                    f.write(f"{row[0]} {row[1]} {row[2]} ")
                f.write("\n")

            # Symlink or copy image
            dst = os.path.join(scene_dir, f"image{idx:04d}.jpg")
            if not os.path.exists(dst):
                try:
                    os.symlink(os.path.abspath(img_path), dst)
                except OSError:
                    shutil.copy2(img_path, dst)

        output_prefix = os.path.join(work_dir, "textured")
        result = subprocess.run(
            ["texrecon", scene_dir, obj_file, output_prefix],
            capture_output=True,
            text=True,
            timeout=300,
        )

        if result.returncode != 0:
            logger.warning(
                "texrecon exited with code %d: %s",
                result.returncode,
                result.stderr,
            )
            return False

        # texrecon outputs textured_material0000_map_Kd.png etc.
        png_files = glob.glob(
            os.path.join(work_dir, "textured*map_Kd*.png")
        )
        if png_files:
            shutil.copy2(
                png_files[0], os.path.join(work_dir, "texture_atlas.png")
            )
        mtl_out = output_prefix + ".mtl"
        if os.path.isfile(mtl_out):
            shutil.copy2(mtl_out, os.path.join(work_dir, "mesh.mtl"))

        return os.path.isfile(os.path.join(work_dir, "texture_atlas.png"))

    def _fallback_texture_projection(
        self,
        mesh: o3d.geometry.TriangleMesh,
        images: List[dict],
        texture_file: str,
        material_file: str,
        logger,
    ) -> None:
        """Project vertex colours from images using camera poses and write a
        simple texture atlas PNG + MTL file."""
        import os

        import numpy as np
        from PIL import Image

        vertices = np.asarray(mesh.vertices)
        n_verts = len(vertices)

        if n_verts == 0:
            self._write_blank_texture(texture_file, material_file)
            return

        # Accumulate colour contributions per vertex
        color_accum = np.zeros((n_verts, 3), dtype=np.float64)
        color_count = np.zeros(n_verts, dtype=np.float64)

        for img_info in images:
            img_path = img_info["path"]
            pose = np.asarray(img_info["pose"], dtype=np.float64)

            try:
                pil_img = Image.open(img_path).convert("RGB")
            except Exception:
                logger.warning("Could not read image: %s", img_path)
                continue

            img_arr = np.asarray(pil_img)  # (H, W, 3) RGB
            h, w = img_arr.shape[:2]

            # Camera-to-world → world-to-camera
            try:
                cam_from_world = np.linalg.inv(pose)
            except np.linalg.LinAlgError:
                logger.warning("Non-invertible pose for %s, skipping.", img_path)
                continue

            # Simple pinhole projection
            fx = fy = max(w, h) * 0.8
            cx, cy = w / 2.0, h / 2.0

            # Transform vertices to camera frame
            ones = np.ones((n_verts, 1))
            verts_h = np.hstack([vertices, ones])  # (N, 4)
            verts_cam = (cam_from_world @ verts_h.T).T  # (N, 4)
            z = verts_cam[:, 2]

            valid = z > 0.01
            if not np.any(valid):
                continue

            u = (fx * verts_cam[:, 0] / z + cx).astype(np.float64)
            v_coord = (fy * verts_cam[:, 1] / z + cy).astype(np.float64)

            in_bounds = (
                valid
                & (u >= 0) & (u < w - 1)
                & (v_coord >= 0) & (v_coord < h - 1)
            )
            indices = np.where(in_bounds)[0]

            if len(indices) == 0:
                continue

            ui = np.clip(u[indices].astype(int), 0, w - 1)
            vi = np.clip(v_coord[indices].astype(int), 0, h - 1)

            # Sample colours (already RGB), normalise to [0,1]
            sampled = img_arr[vi, ui].astype(np.float64) / 255.0
            color_accum[indices] += sampled
            color_count[indices] += 1.0

        # Average colours
        has_color = color_count > 0
        color_accum[has_color] /= color_count[has_color, np.newaxis]
        color_accum[~has_color] = 0.5  # default grey

        # ---- Build a simple texture atlas ----
        atlas_size = max(int(np.ceil(np.sqrt(n_verts))), 1)
        atlas = np.full((atlas_size, atlas_size, 3), 128, dtype=np.uint8)

        for i in range(n_verts):
            row = i // atlas_size
            col = i % atlas_size
            if row < atlas_size and col < atlas_size:
                atlas[row, col] = (color_accum[i] * 255).astype(np.uint8)

        Image.fromarray(atlas).save(texture_file)

        # ---- Write MTL file ----
        texture_basename = os.path.basename(texture_file)
        with open(material_file, "w") as f:
            f.write("# Material file generated by ReconstructionPipeline\n")
            f.write("newmtl material0\n")
            f.write("Ka 0.2 0.2 0.2\n")
            f.write("Kd 0.8 0.8 0.8\n")
            f.write("Ks 0.0 0.0 0.0\n")
            f.write(f"map_Kd {texture_basename}\n")

        logger.info(
            "Fallback texture atlas written: %dx%d, %d/%d vertices coloured.",
            atlas_size,
            atlas_size,
            int(has_color.sum()),
            n_verts,
        )

    @staticmethod
    def _write_blank_texture(texture_file: str, material_file: str) -> None:
        """Write a minimal blank texture and MTL when the mesh has no vertices."""
        import os

        import numpy as np
        from PIL import Image

        blank = np.full((1, 1, 3), 128, dtype=np.uint8)
        Image.fromarray(blank).save(texture_file)

        texture_basename = os.path.basename(texture_file)
        with open(material_file, "w") as f:
            f.write("# Material file generated by ReconstructionPipeline\n")
            f.write("newmtl material0\n")
            f.write(f"map_Kd {texture_basename}\n")

    @staticmethod
    def _apply_color_correction(texture_file: str, logger) -> None:
        """Apply global colour correction via mean-colour normalisation.

        Adjusts each channel so the mean brightness is ~128, which reduces
        visible seams caused by inconsistent exposure across source images.
        """
        import numpy as np
        from PIL import Image

        try:
            pil_img = Image.open(texture_file).convert("RGB")
        except Exception:
            return

        img = np.asarray(pil_img).astype(np.float64)
        if img.size == 0:
            return

        for c in range(3):
            channel = img[:, :, c]
            mean_val = channel.mean()
            if mean_val > 1e-6:
                img[:, :, c] = np.clip(channel * (128.0 / mean_val), 0, 255)

        Image.fromarray(img.astype(np.uint8)).save(texture_file)
        logger.info("Global colour correction applied to %s.", texture_file)

    @staticmethod
    def _compute_texture_quality(texture_file: str, logger) -> float:
        """Compute a texture quality score in [0, 1] based on coverage.

        The score is the fraction of non-default (non-grey-128) pixels in the
        texture atlas.  A fully covered texture scores close to 1.0.
        """
        import os

        import numpy as np
        from PIL import Image

        if not os.path.isfile(texture_file):
            logger.warning("Texture file not found: %s", texture_file)
            return 0.0

        try:
            pil_img = Image.open(texture_file).convert("RGB")
        except Exception:
            return 0.0

        img = np.asarray(pil_img)
        if img.size == 0:
            return 0.0

        # A pixel is "covered" if it differs from the default grey (128,128,128)
        grey = np.array([128, 128, 128], dtype=np.uint8)
        diff = np.abs(img.astype(np.int16) - grey.astype(np.int16))
        covered = np.any(diff > 5, axis=2)
        score = float(covered.sum()) / max(covered.size, 1)

        logger.info("Texture quality score: %.3f", score)
        return score

    def build_feature_database(
        self,
        images: List[dict],
        mesh: o3d.geometry.TriangleMesh,
    ) -> FeatureDatabase:
        """Extract ORB features from keyframes and build a visual feature database.

        For each keyframe image, extracts up to 2000 ORB features, back-projects
        2D keypoints to 3D via ray-mesh intersection, and stores valid
        correspondences. Keyframes with fewer than 20 valid features are skipped.

        After processing all keyframes, builds a visual Bag-of-Words (BoW)
        vocabulary using K-Means clustering (K=min(1000, n_descriptors)) on all
        collected descriptors, then computes L1-normalized BoW vectors for each
        keyframe.

        Args:
            images: List of dicts each containing ``"path"`` (image file path)
                and ``"pose"`` (4x4 camera-to-world matrix).
            mesh: Triangle mesh used for ray-casting.

        Returns:
            Populated feature database with keyframes, vocabulary (KMeans model),
            and global_descriptors (L1-normalized BoW vectors, shape n_keyframes x K).
        """
        import logging

        import cv2
        import numpy as np

        logger = logging.getLogger(__name__)

        # --- Step 1: Create ORB detector ---
        orb = cv2.ORB_create(nfeatures=2000)

        # --- Step 2: Prepare ray-casting scene from mesh ---
        scene = o3d.t.geometry.RaycastingScene()
        mesh_legacy = mesh
        # Convert to tensor-based mesh for raycasting
        mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh_legacy)
        scene.add_triangles(mesh_t)

        keyframes: List[KeyframeData] = []

        for idx, img_info in enumerate(images):
            img_path = img_info["path"]
            pose = np.asarray(img_info["pose"], dtype=np.float64)

            # Load image in grayscale for ORB
            img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img_gray is None:
                logger.warning("Could not read image: %s, skipping.", img_path)
                continue

            h, w = img_gray.shape[:2]

            # --- Step 2a: Extract ORB features ---
            kps, descriptors = orb.detectAndCompute(img_gray, None)

            if descriptors is None or len(kps) == 0:
                logger.info(
                    "No features detected in image %d (%s), skipping.", idx, img_path
                )
                continue

            # --- Step 2b: Estimate intrinsics from image size ---
            fx = fy = max(w, h) * 0.8
            cx, cy = w / 2.0, h / 2.0

            # --- Step 2c: Backproject 2D keypoints to 3D via ray-mesh intersection ---
            valid_keypoints: List[tuple[float, float]] = []
            valid_descriptors: List[np.ndarray] = []
            valid_points_3d: List[tuple[float, float, float]] = []

            # Build rays for all keypoints at once for efficiency
            rays_list = []
            for kp in kps:
                px, py = kp.pt
                # Convert pixel to normalised camera coordinates
                x_cam = (px - cx) / fx
                y_cam = (py - cy) / fy
                # Ray direction in camera frame (looking along +Z)
                dir_cam = np.array([x_cam, y_cam, 1.0])
                dir_cam = dir_cam / np.linalg.norm(dir_cam)

                # Transform ray to world coordinates
                R = pose[:3, :3]
                t = pose[:3, 3]
                origin_world = t
                dir_world = R @ dir_cam

                rays_list.append([
                    origin_world[0], origin_world[1], origin_world[2],
                    dir_world[0], dir_world[1], dir_world[2],
                ])

            rays_tensor = o3d.core.Tensor(
                np.array(rays_list, dtype=np.float32), dtype=o3d.core.float32
            )
            result = scene.cast_rays(rays_tensor)
            t_hit = result["t_hit"].numpy()

            for j, kp in enumerate(kps):
                if np.isinf(t_hit[j]) or t_hit[j] <= 0:
                    continue  # No intersection

                # Compute 3D hit point
                ray = np.array(rays_list[j], dtype=np.float64)
                origin = ray[:3]
                direction = ray[3:]
                hit_point = origin + t_hit[j] * direction

                valid_keypoints.append((kp.pt[0], kp.pt[1]))
                valid_descriptors.append(descriptors[j])
                valid_points_3d.append(
                    (float(hit_point[0]), float(hit_point[1]), float(hit_point[2]))
                )

            # --- Step 2d: Skip keyframes with < 20 valid features ---
            if len(valid_keypoints) < 20:
                logger.info(
                    "Image %d (%s): only %d valid features (< 20), skipping.",
                    idx, img_path, len(valid_keypoints),
                )
                continue

            # Build KeyframeData
            desc_array = np.array(valid_descriptors, dtype=np.uint8)
            keyframe = KeyframeData(
                image_id=idx,
                keypoints=valid_keypoints,
                descriptors=desc_array,
                points_3d=valid_points_3d,
                camera_pose=pose,
            )
            keyframes.append(keyframe)
            logger.info(
                "Image %d (%s): %d valid features added to database.",
                idx, img_path, len(valid_keypoints),
            )

        logger.info(
            "Feature database built: %d keyframes out of %d images.",
            len(keyframes), len(images),
        )

        # --- Step 3: Build visual Bag-of-Words (BoW) vocabulary ---
        vocabulary = None
        global_descriptors = None

        # Collect all descriptors from all keyframes
        all_descriptors = []
        for kf in keyframes:
            all_descriptors.append(kf.descriptors)

        if all_descriptors:
            all_desc_matrix = np.vstack(all_descriptors).astype(np.float64)
            n_descriptors = all_desc_matrix.shape[0]

            if n_descriptors > 0:
                from sklearn.cluster import KMeans

                k = min(1000, n_descriptors)
                logger.info(
                    "Building BoW vocabulary: K-Means with K=%d on %d descriptors.",
                    k, n_descriptors,
                )
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(all_desc_matrix)
                vocabulary = kmeans

                # --- Step 4: Compute BoW vector for each keyframe ---
                n_keyframes = len(keyframes)
                bow_vectors = np.zeros((n_keyframes, k), dtype=np.float64)

                for i, kf in enumerate(keyframes):
                    desc_float = kf.descriptors.astype(np.float64)
                    labels = kmeans.predict(desc_float)
                    # Build histogram of cluster assignments
                    for label in labels:
                        bow_vectors[i, label] += 1.0
                    # Normalize by L1 norm so the vector sums to 1.0
                    l1_norm = np.sum(bow_vectors[i])
                    if l1_norm > 0:
                        bow_vectors[i] /= l1_norm

                global_descriptors = bow_vectors
                logger.info(
                    "BoW vectors computed for %d keyframes (K=%d).",
                    n_keyframes, k,
                )

        return FeatureDatabase(
            keyframes=keyframes,
            vocabulary=vocabulary,
            global_descriptors=global_descriptors,
        )


    def export_asset_bundle(
        self,
        mesh: TexturedMesh,
        features: FeatureDatabase,
        output_dir: str,
    ) -> None:
        """Export the final Area Target asset bundle.

        Writes mesh.obj, mesh.mtl, texture_atlas.png, features.db, and
        manifest.json to *output_dir*. Validates that all file references in
        the manifest match the actual directory structure.

        Privacy: Raw RGB images are not included in the asset bundle.
        The output contains only the simplified mesh, texture atlas,
        feature descriptors database, and manifest metadata.

        Args:
            mesh: Textured mesh with material and texture files.
            features: Visual feature database.
            output_dir: Directory to write the asset bundle into.
        """
        logger = logging.getLogger(__name__)

        # 1. Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # 2. Write mesh.obj using Open3D
        obj_path = os.path.join(output_dir, "mesh.obj")
        o3d.io.write_triangle_mesh(obj_path, mesh.mesh)
        logger.info("Wrote mesh to %s", obj_path)

        # 3. Copy texture_atlas.png from mesh.texture_file
        texture_dst = os.path.join(output_dir, "texture_atlas.png")
        if mesh.texture_file and os.path.isfile(mesh.texture_file):
            shutil.copy2(mesh.texture_file, texture_dst)
            logger.info("Copied texture to %s", texture_dst)

        # 4. Copy mesh.mtl from mesh.material_file
        mtl_dst = os.path.join(output_dir, "mesh.mtl")
        if mesh.material_file and os.path.isfile(mesh.material_file):
            shutil.copy2(mesh.material_file, mtl_dst)
            logger.info("Copied material to %s", mtl_dst)

        # 5. Save features.db using save_feature_database
        from processing_pipeline.feature_db import save_feature_database

        db_path = os.path.join(output_dir, "features.db")
        save_feature_database(features, db_path)
        logger.info("Saved feature database to %s", db_path)

        # 6. Compute bounding box from mesh
        bbox = mesh.mesh.get_axis_aligned_bounding_box()
        min_bound = np.asarray(bbox.min_bound).tolist()
        max_bound = np.asarray(bbox.max_bound).tolist()

        # 7. Generate manifest.json
        manifest = {
            "version": "1.0",
            "name": os.path.basename(os.path.normpath(output_dir)),
            "meshFile": "mesh.obj",
            "textureFile": "texture_atlas.png",
            "featureDbFile": "features.db",
            "bounds": {
                "min": min_bound,
                "max": max_bound,
            },
            "keyframeCount": len(features.keyframes),
            "featureType": "ORB",
            "createdAt": datetime.now(timezone.utc).isoformat(),
        }

        manifest_path = os.path.join(output_dir, "manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        logger.info("Wrote manifest to %s", manifest_path)

        # 8. Validate all referenced files exist in output_dir
        expected_files = [
            manifest["meshFile"],
            manifest["textureFile"],
            manifest["featureDbFile"],
        ]
        for fname in expected_files:
            fpath = os.path.join(output_dir, fname)
            if not os.path.isfile(fpath):
                raise FileNotFoundError(
                    f"Asset bundle validation failed: expected file "
                    f"'{fname}' not found in {output_dir}"
                )

    def run(self, input_dir: str, output_dir: str) -> None:
        """Execute the full reconstruction pipeline end-to-end.

        Chains: point cloud preprocessing → mesh reconstruction →
        mesh simplification → texture mapping → feature extraction →
        asset bundle export.

        Args:
            input_dir: Path to the scan data directory containing
                ``pointcloud.ply``, ``poses.json``, ``intrinsics.json``,
                and an ``images/`` subdirectory.
            output_dir: Path to write the output asset bundle.

        Raises:
            FileNotFoundError: If required input files are missing.
            ValueError: If poses.json is malformed.
        """
        logger = logging.getLogger(__name__)

        # --- Validate input directory ---
        ply_path = os.path.join(input_dir, "pointcloud.ply")
        poses_path = os.path.join(input_dir, "poses.json")

        if not os.path.isfile(ply_path):
            raise FileNotFoundError(f"Point cloud file not found: {ply_path}")
        if not os.path.isfile(poses_path):
            raise FileNotFoundError(f"Poses file not found: {poses_path}")

        # --- Load poses.json ---
        with open(poses_path, "r", encoding="utf-8") as f:
            poses_data = json.load(f)

        frames = poses_data.get("frames", [])
        if not frames:
            raise ValueError("poses.json contains no frames")

        # Build images list: each entry has "path" and "pose"
        images: List[dict] = []
        for frame in frames:
            image_file = frame["imageFile"]
            image_path = os.path.join(input_dir, image_file)
            transform = np.array(frame["transform"], dtype=np.float64).reshape(4, 4, order="F")
            images.append({"path": image_path, "pose": transform})

        logger.info("Loaded %d frames from poses.json", len(images))

        # --- Step 1: Point cloud preprocessing ---
        logger.info("Step 1/6: Point cloud preprocessing")
        cloud = self.process_point_cloud(ply_path)
        logger.info("Point cloud preprocessed: %d points", cloud.point_count)

        # --- Step 2: Mesh reconstruction ---
        logger.info("Step 2/6: Mesh reconstruction")
        mesh = self.reconstruct_mesh(cloud)
        logger.info("Mesh reconstructed: %d triangles", len(mesh.triangles))

        # --- Step 3: Mesh simplification ---
        logger.info("Step 3/6: Mesh simplification")
        simplified_mesh = self.simplify_mesh(mesh)
        logger.info("Mesh simplified: %d triangles", len(simplified_mesh.triangles))

        # --- Step 4: Texture mapping ---
        logger.info("Step 4/6: Texture mapping")
        textured_mesh = self.generate_texture(simplified_mesh, images)
        logger.info("Texture mapping complete (quality=%.3f)", textured_mesh.quality_score or 0.0)

        # --- Step 5: Feature extraction ---
        logger.info("Step 5/6: Feature extraction")
        features = self.build_feature_database(images, simplified_mesh)
        logger.info("Feature database built: %d keyframes", len(features.keyframes))

        # --- Step 6: Asset bundle export ---
        logger.info("Step 6/6: Asset bundle export")
        self.export_asset_bundle(textured_mesh, features, output_dir)
        logger.info("Asset bundle exported to %s", output_dir)
