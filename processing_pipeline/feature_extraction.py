"""Feature extraction: ORB + BoW vocabulary building and Hamming word assignment."""

from __future__ import annotations

from typing import List

import numpy as np
import open3d as o3d

from .models import (
    FeatureDatabase,
    KeyframeData,
)

# Module-level popcount lookup table (256 entries)
_POPCOUNT_TABLE = np.array([bin(i).count("1") for i in range(256)], dtype=np.int32)


def _hamming_word_assignment(
    descriptors: np.ndarray,
    vocabulary_medoids: np.ndarray,
) -> np.ndarray:
    """Assign visual words to descriptors using Hamming distance.

    Vectorized implementation using broadcast XOR + popcount lookup table.
    (N, 1, D) XOR (1, K, D) → (N, K, D) → popcount lookup → sum → argmin

    Args:
        descriptors: (N, D) uint8 array of ORB descriptors.
        vocabulary_medoids: (K, D) uint8 array of vocabulary medoid descriptors.

    Returns:
        (N,) int array of word indices (0..K-1).
    """
    descriptors = np.asarray(descriptors, dtype=np.uint8)
    vocabulary_medoids = np.asarray(vocabulary_medoids, dtype=np.uint8)

    # 广播 XOR: (N, 1, D) ^ (1, K, D) → (N, K, D)
    xor = np.bitwise_xor(
        descriptors[:, np.newaxis, :],
        vocabulary_medoids[np.newaxis, :, :],
    )
    # Popcount via lookup: (N, K, D) → (N, K)
    dists = _POPCOUNT_TABLE[xor].sum(axis=2)
    return dists.argmin(axis=1).astype(np.intp)


def build_feature_database(
    images: List[dict],
    mesh: o3d.geometry.TriangleMesh,
    intrinsics: dict | None = None,
) -> FeatureDatabase:
    """Extract ORB features from keyframes and build a visual feature database.

    For each keyframe image, extracts up to 2000 ORB features, back-projects
    2D keypoints to 3D via ray-mesh intersection, and stores valid
    correspondences. Keyframes with fewer than 20 valid features are skipped.

    After processing all keyframes, builds a visual Bag-of-Words (BoW)
    vocabulary using K-Means clustering (K=min(1000, n_descriptors)) on all
    collected descriptors, then computes TF-IDF weighted, L2-normalized BoW
    vectors for each keyframe (consistent with C++ computeBoW).

    Args:
        images: List of dicts each containing ``"path"`` (image file path)
            and ``"pose"`` (4x4 camera-to-world matrix).
        mesh: Triangle mesh used for ray-casting.
        intrinsics: Optional dict with ``fx``, ``fy``, ``cx``, ``cy`` keys.
            If *None*, intrinsics are estimated from image dimensions.

    Returns:
        Populated feature database with keyframes, vocabulary (uint8 medoids),
        and global_descriptors (TF-IDF weighted, L2-normalized BoW vectors,
        shape n_keyframes x K).
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
        if intrinsics:
            fx = intrinsics["fx"]
            fy = intrinsics["fy"]
            cx = intrinsics["cx"]
            cy = intrinsics["cy"]
        else:
            fx = fy = max(w, h) * 0.8
            cx, cy = w / 2.0, h / 2.0

        # --- Step 2c: Backproject 2D keypoints to 3D via ray-mesh intersection ---
        valid_keypoints: List[tuple[float, float]] = []
        valid_descriptors: List[np.ndarray] = []
        valid_points_3d: List[tuple[float, float, float]] = []

        # Build rays for all keypoints at once for efficiency
        rays_list = []
        R = pose[:3, :3]
        t = pose[:3, 3]
        for kp in kps:
            px, py = kp.pt
            # Convert pixel to normalised camera coordinates
            x_cam = (px - cx) / fx
            y_cam = (py - cy) / fy
            # ARKit/OpenGL convention: camera looks along -Z
            dir_cam = np.array([x_cam, -y_cam, -1.0])
            dir_cam = dir_cam / np.linalg.norm(dir_cam)

            # Transform ray to world coordinates
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

    if len(keyframes) == 0 and len(images) > 0:
        raise ValueError(
            "Feature database is empty: no keyframes with sufficient features (>= 20) "
            "were found in any input image."
        )

    # --- Step 3: Build visual Bag-of-Words (BoW) vocabulary ---
    vocabulary = None
    global_descriptors = None

    # Collect all descriptors from all keyframes
    all_descriptors = []
    for kf in keyframes:
        all_descriptors.append(kf.descriptors)

    if all_descriptors:
        all_desc_uint8 = np.vstack(all_descriptors).astype(np.uint8)
        all_desc_float = all_desc_uint8.astype(np.float64)
        n_descriptors = all_desc_float.shape[0]

        if n_descriptors > 0:
            from sklearn.cluster import KMeans

            k = min(1000, n_descriptors)
            logger.info(
                "Building BoW vocabulary: K-Means with K=%d on %d descriptors.",
                k, n_descriptors,
            )
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=3)
            kmeans.fit(all_desc_float)

            # --- Step 3b: Compute medoids (real uint8 descriptors) ---
            # For each cluster, find the real ORB descriptor closest
            # to the cluster center using Hamming distance.
            labels = kmeans.labels_
            centers_uint8 = kmeans.cluster_centers_.astype(np.uint8)
            medoids = np.zeros((k, all_desc_uint8.shape[1]), dtype=np.uint8)

            for c in range(k):
                mask = labels == c
                cluster_descs = all_desc_uint8[mask]
                if len(cluster_descs) == 0:
                    # Fallback: cast center to uint8 directly
                    medoids[c] = centers_uint8[c]
                    continue
                # Hamming distance from each descriptor to the center (vectorized)
                xor = np.bitwise_xor(cluster_descs, centers_uint8[c])  # (M, 32)
                hamming_dists = _POPCOUNT_TABLE[xor].sum(axis=1)       # (M,)
                medoids[c] = cluster_descs[np.argmin(hamming_dists)]

            # Store medoids as the vocabulary (uint8, matching C++ side)
            vocabulary = medoids

            # --- Step 4: Compute BoW vector for each keyframe ---
            # Use Hamming distance-based word assignment (consistent
            # with C++ computeBoW) with TF-IDF weighting + L2 normalization
            import math

            n_keyframes = len(keyframes)

            # 4a: Assign words for all keyframes and compute document frequencies
            all_word_labels = []
            doc_freq = np.zeros(k, dtype=np.float64)
            for i, kf in enumerate(keyframes):
                word_labels = _hamming_word_assignment(
                    kf.descriptors, medoids,
                )
                all_word_labels.append(word_labels)
                unique_words = set(word_labels.tolist())
                for w in unique_words:
                    doc_freq[w] += 1.0

            # 4b: Compute IDF weights: idf(w) = log(N / (1 + df(w)))
            idf_weights = np.array(
                [math.log(n_keyframes / (1.0 + doc_freq[w])) for w in range(k)],
                dtype=np.float64,
            )

            # 4c: Build TF-IDF weighted BoW vectors with L2 normalization
            bow_vectors = np.zeros((n_keyframes, k), dtype=np.float64)
            for i, kf in enumerate(keyframes):
                word_labels = all_word_labels[i]
                for label in word_labels:
                    bow_vectors[i, label] += idf_weights[label]
                # L2 normalization (consistent with C++ computeBoW)
                l2_norm = np.linalg.norm(bow_vectors[i])
                if l2_norm > 1e-9:
                    bow_vectors[i] /= l2_norm

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
