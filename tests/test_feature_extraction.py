"""Unit tests for buildFeatureDatabase() — ORB extraction and ray-mesh intersection.

Validates Requirements 8.1, 8.2, 8.3.
"""

from __future__ import annotations

import os
import tempfile

import cv2
import numpy as np
import open3d as o3d
import pytest

from processing_pipeline.models import (
    FeatureDatabase,
    KeyframeData,
    ProcessedCloud,
)
from processing_pipeline.pipeline import ReconstructionPipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sphere_cloud(n_points: int = 5000, radius: float = 1.0) -> ProcessedCloud:
    """Create a ProcessedCloud of points on a sphere with outward normals."""
    rng = np.random.default_rng(42)
    raw = rng.standard_normal((n_points, 3))
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-10, None)
    points = (raw / norms) * radius
    normals = raw / norms
    colors = np.zeros((n_points, 3))
    return ProcessedCloud(
        points=points, normals=normals, colors=colors, point_count=n_points
    )


def _build_sphere_mesh() -> o3d.geometry.TriangleMesh:
    """Build a sphere mesh via lazy cache (avoids Poisson segfault at import time)."""
    from tests.conftest import get_sphere_mesh
    return get_sphere_mesh(n_points=5000, radius=1.0)


def _make_textured_image(
    width: int = 640, height: int = 480, seed: int = 42
) -> np.ndarray:
    """Generate a synthetic image with enough texture for ORB detection."""
    rng = np.random.RandomState(seed)
    img = rng.randint(0, 256, (height, width), dtype=np.uint8)
    # Add strong corners / edges to guarantee ORB detections
    for _ in range(80):
        cx = rng.randint(20, width - 20)
        cy = rng.randint(20, height - 20)
        r = rng.randint(5, 20)
        cv2.circle(img, (cx, cy), r, int(rng.randint(0, 256)), -1)
        cv2.rectangle(
            img,
            (cx - r, cy - r),
            (cx + r, cy + r),
            int(rng.randint(0, 256)),
            2,
        )
    return img


def _save_image(img: np.ndarray, directory: str, name: str) -> str:
    path = os.path.join(directory, name)
    cv2.imwrite(path, img)
    return path


def _make_camera_pose(
    tx: float = 0.0, ty: float = 0.0, tz: float = 3.0
) -> np.ndarray:
    """Camera-to-world pose looking at the origin from +Z."""
    pose = np.eye(4, dtype=np.float64)
    pose[0, 3] = tx
    pose[1, 3] = ty
    pose[2, 3] = tz
    return pose


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBuildFeatureDatabaseBasic:
    """Basic functionality tests for buildFeatureDatabase()."""

    def setup_method(self):
        import copy
        self.pipeline = ReconstructionPipeline()
        self.mesh = copy.deepcopy(_build_sphere_mesh())
        self.tmpdir = tempfile.mkdtemp()

    def _make_images(self, count: int = 3) -> list:
        images = []
        for i in range(count):
            img = _make_textured_image(seed=42 + i)
            path = _save_image(img, self.tmpdir, f"frame_{i:04d}.png")
            pose = _make_camera_pose(tx=0.2 * i, tz=3.0)
            images.append({"path": path, "pose": pose})
        return images

    def test_returns_feature_database(self):
        images = self._make_images(2)
        db = self.pipeline.build_feature_database(images, self.mesh)
        assert isinstance(db, FeatureDatabase)

    def test_keyframes_have_correct_structure(self):
        images = self._make_images(2)
        db = self.pipeline.build_feature_database(images, self.mesh)
        for kf in db.keyframes:
            assert isinstance(kf, KeyframeData)
            assert len(kf.keypoints) == len(kf.points_3d)
            assert kf.descriptors.shape[0] == len(kf.keypoints)
            assert kf.descriptors.shape[1] == 32  # ORB descriptor size
            assert kf.descriptors.dtype == np.uint8

    def test_keyframes_have_at_least_20_features(self):
        """Requirement 8.3: keyframes with < 20 valid features are skipped."""
        images = self._make_images(3)
        db = self.pipeline.build_feature_database(images, self.mesh)
        for kf in db.keyframes:
            assert len(kf.keypoints) >= 20

    def test_3d_points_near_mesh_surface(self):
        """Requirement 8.2: 3D points come from ray-mesh intersection."""
        images = self._make_images(2)
        db = self.pipeline.build_feature_database(images, self.mesh)
        bbox = self.mesh.get_axis_aligned_bounding_box()
        min_b = np.asarray(bbox.min_bound) - 0.5  # tolerance for Poisson mesh
        max_b = np.asarray(bbox.max_bound) + 0.5
        for kf in db.keyframes:
            for pt in kf.points_3d:
                p = np.array(pt)
                assert np.all(p >= min_b) and np.all(p <= max_b), (
                    f"3D point {p} outside mesh bounding box"
                )

    def test_camera_pose_stored(self):
        images = self._make_images(1)
        db = self.pipeline.build_feature_database(images, self.mesh)
        if db.keyframes:
            kf = db.keyframes[0]
            np.testing.assert_array_almost_equal(
                kf.camera_pose, images[0]["pose"]
            )

    def test_vocabulary_is_kmeans(self):
        """Requirement 8.4: vocabulary is a trained KMeans model."""
        from sklearn.cluster import KMeans

        images = self._make_images(2)
        db = self.pipeline.build_feature_database(images, self.mesh)
        if db.keyframes:
            assert isinstance(db.vocabulary, KMeans)
        else:
            assert db.vocabulary is None

    def test_global_descriptors_shape(self):
        """Requirement 8.5: global_descriptors has shape (n_keyframes, K)."""
        images = self._make_images(2)
        db = self.pipeline.build_feature_database(images, self.mesh)
        if db.keyframes:
            assert db.global_descriptors is not None
            assert db.global_descriptors.shape[0] == len(db.keyframes)
            k = db.vocabulary.n_clusters
            assert db.global_descriptors.shape[1] == k
        else:
            assert db.global_descriptors is None

    def test_bow_vectors_l1_normalized(self):
        """Requirement 8.5: BoW vectors are L1-normalized (sum to 1.0)."""
        images = self._make_images(2)
        db = self.pipeline.build_feature_database(images, self.mesh)
        if db.keyframes and db.global_descriptors is not None:
            for i in range(db.global_descriptors.shape[0]):
                l1 = np.sum(db.global_descriptors[i])
                np.testing.assert_almost_equal(l1, 1.0, decimal=6)

    def test_bow_vectors_non_negative(self):
        """BoW vectors should contain only non-negative values."""
        images = self._make_images(2)
        db = self.pipeline.build_feature_database(images, self.mesh)
        if db.keyframes and db.global_descriptors is not None:
            assert np.all(db.global_descriptors >= 0)


class TestBuildFeatureDatabaseEdgeCases:
    """Edge case tests for buildFeatureDatabase()."""

    def setup_method(self):
        import copy
        self.pipeline = ReconstructionPipeline()
        self.mesh = copy.deepcopy(_build_sphere_mesh())
        self.tmpdir = tempfile.mkdtemp()

    def test_empty_images_list(self):
        db = self.pipeline.build_feature_database([], self.mesh)
        assert isinstance(db, FeatureDatabase)
        assert len(db.keyframes) == 0
        assert db.vocabulary is None
        assert db.global_descriptors is None

    def test_unreadable_image_skipped(self):
        """Images that can't be read should be skipped gracefully."""
        images = [{"path": "/nonexistent/image.png", "pose": np.eye(4)}]
        db = self.pipeline.build_feature_database(images, self.mesh)
        assert len(db.keyframes) == 0

    def test_blank_image_few_features(self):
        """A blank (uniform) image produces few/no ORB features and is skipped."""
        blank = np.full((480, 640), 128, dtype=np.uint8)
        path = _save_image(blank, self.tmpdir, "blank.png")
        images = [{"path": path, "pose": _make_camera_pose()}]
        db = self.pipeline.build_feature_database(images, self.mesh)
        # Blank image should produce < 20 valid features → skipped
        assert len(db.keyframes) == 0

    def test_camera_far_away_no_hits(self):
        """Camera very far from mesh — rays miss → keyframe skipped."""
        img = _make_textured_image()
        path = _save_image(img, self.tmpdir, "far.png")
        # Camera at z=1000 looking at origin — the sphere is radius ~1
        pose = _make_camera_pose(tz=1000.0)
        images = [{"path": path, "pose": pose}]
        db = self.pipeline.build_feature_database(images, self.mesh)
        # Most rays will miss the small sphere at such distance
        # Either 0 keyframes or keyframes with >= 20 features (invariant)
        for kf in db.keyframes:
            assert len(kf.keypoints) >= 20


class TestBuildFeatureDatabaseORB:
    """Tests specifically for ORB feature extraction (Requirement 8.1)."""

    def setup_method(self):
        import copy
        self.pipeline = ReconstructionPipeline()
        self.mesh = copy.deepcopy(_build_sphere_mesh())
        self.tmpdir = tempfile.mkdtemp()

    def test_orb_max_features_2000(self):
        """Requirement 8.1: ORB uses nfeatures=2000."""
        rng = np.random.RandomState(99)
        img = rng.randint(0, 256, (480, 640), dtype=np.uint8)
        for _ in range(200):
            cx = rng.randint(10, 630)
            cy = rng.randint(10, 470)
            cv2.circle(
                img, (cx, cy), rng.randint(3, 15),
                int(rng.randint(0, 256)), -1,
            )
        path = _save_image(img, self.tmpdir, "textured.png")
        pose = _make_camera_pose(tz=3.0)
        images = [{"path": path, "pose": pose}]
        db = self.pipeline.build_feature_database(images, self.mesh)
        for kf in db.keyframes:
            assert len(kf.keypoints) <= 2000
