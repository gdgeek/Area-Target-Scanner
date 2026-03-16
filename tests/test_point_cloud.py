"""Unit tests for point cloud preprocessing.

Validates: Requirements 4.1, 4.2, 4.3, 4.7

Tests the ReconstructionPipeline.process_point_cloud() method using
synthetic point clouds (sphere, cube) and invalid inputs.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import open3d as o3d
import pytest

from processing_pipeline.pipeline import ReconstructionPipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sphere_cloud(n_points: int = 5000, radius: float = 1.0) -> np.ndarray:
    """Generate points uniformly distributed on a sphere surface."""
    rng = np.random.default_rng(42)
    raw = rng.standard_normal((n_points, 3))
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-10, None)
    return (raw / norms) * radius


def _make_cube_cloud(n_points: int = 5000, half_size: float = 1.0) -> np.ndarray:
    """Generate points uniformly distributed on the six faces of a cube."""
    rng = np.random.default_rng(42)
    points_per_face = n_points // 6
    remainder = n_points - points_per_face * 6
    faces = []

    for axis in range(3):
        for sign in (-1.0, 1.0):
            count = points_per_face + (1 if remainder > 0 else 0)
            if remainder > 0:
                remainder -= 1
            free = rng.uniform(-half_size, half_size, size=(count, 2))
            fixed = np.full((count, 1), sign * half_size)
            coords = np.empty((count, 3))
            free_idx = 0
            for i in range(3):
                if i == axis:
                    coords[:, i] = fixed[:, 0]
                else:
                    coords[:, i] = free[:, free_idx]
                    free_idx += 1
            faces.append(coords)

    return np.vstack(faces)


def _save_ply(points: np.ndarray, path: str) -> None:
    """Save an (N, 3) array as a PLY file via Open3D."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(path, pcd)


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------


class TestProcessPointCloudSphere:
    """Test process_point_cloud with a synthetic sphere (5000+ points).

    Validates: Requirements 4.1, 4.2, 4.3
    """

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.pipeline = ReconstructionPipeline()
        points = _make_sphere_cloud(n_points=5500)
        self.input_point_count = len(points)
        self.ply_path = str(tmp_path / "sphere.ply")
        _save_ply(points, self.ply_path)
        self.result = self.pipeline.process_point_cloud(self.ply_path)

    def test_output_has_normals(self):
        """Processed cloud must have normals for every point."""
        assert self.result.normals is not None
        assert self.result.normals.shape == (self.result.point_count, 3)

    def test_downsampled_fewer_points(self):
        """Downsampling must reduce the number of points."""
        assert self.result.point_count < self.input_point_count

    def test_point_count_matches(self):
        """point_count field must match the actual points array length."""
        assert self.result.point_count == len(self.result.points)


class TestProcessPointCloudCube:
    """Test process_point_cloud with a synthetic cube (5000+ points).

    Validates: Requirements 4.1, 4.2, 4.3
    """

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.pipeline = ReconstructionPipeline()
        points = _make_cube_cloud(n_points=5500)
        self.input_point_count = len(points)
        self.ply_path = str(tmp_path / "cube.ply")
        _save_ply(points, self.ply_path)
        self.result = self.pipeline.process_point_cloud(self.ply_path)

    def test_output_has_normals(self):
        """Processed cloud must have normals for every point."""
        assert self.result.normals is not None
        assert self.result.normals.shape == (self.result.point_count, 3)

    def test_downsampled_fewer_points(self):
        """Downsampling must reduce the number of points."""
        assert self.result.point_count < self.input_point_count

    def test_point_count_matches(self):
        """point_count field must match the actual points array length."""
        assert self.result.point_count == len(self.result.points)


class TestProcessPointCloudInvalidInput:
    """Test error handling for invalid inputs.

    Validates: Requirement 4.7
    """

    def setup_method(self):
        self.pipeline = ReconstructionPipeline()

    def test_nonexistent_file_raises_value_error(self):
        """Non-existent file path must raise ValueError."""
        with pytest.raises(ValueError, match="does not exist"):
            self.pipeline.process_point_cloud("/nonexistent/path/cloud.ply")

    def test_empty_ply_raises_value_error(self, tmp_path):
        """Empty PLY file must raise ValueError."""
        ply_path = str(tmp_path / "empty.ply")
        pcd = o3d.geometry.PointCloud()
        o3d.io.write_point_cloud(ply_path, pcd)
        with pytest.raises(ValueError, match="[Ii]nvalid|empty"):
            self.pipeline.process_point_cloud(ply_path)

    def test_too_few_points_raises_value_error(self, tmp_path):
        """PLY with fewer than 1000 points must raise ValueError with descriptive message."""
        rng = np.random.default_rng(0)
        points = rng.random((500, 3))
        ply_path = str(tmp_path / "small.ply")
        _save_ply(points, ply_path)
        with pytest.raises(ValueError, match="minimum 1000 required"):
            self.pipeline.process_point_cloud(ply_path)
