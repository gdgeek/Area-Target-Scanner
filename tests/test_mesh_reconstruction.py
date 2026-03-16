"""Unit tests for mesh reconstruction.

Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5, 5.6

Tests the ReconstructionPipeline.reconstruct_mesh() method using
synthetic point clouds and verifying mesh validity, density cropping,
and fallback retry logic.
"""

from __future__ import annotations

import os
import tempfile
from unittest.mock import patch

import numpy as np
import open3d as o3d
import pytest

from processing_pipeline.models import ProcessedCloud
from processing_pipeline.pipeline import ReconstructionPipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sphere_processed_cloud(n_points: int = 5000, radius: float = 1.0) -> ProcessedCloud:
    """Create a ProcessedCloud of points on a sphere with outward normals."""
    rng = np.random.default_rng(42)
    raw = rng.standard_normal((n_points, 3))
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-10, None)
    points = (raw / norms) * radius

    # Normals point outward (same direction as the point on a centered sphere)
    normals = raw / norms

    colors = np.zeros((n_points, 3))

    return ProcessedCloud(
        points=points,
        normals=normals,
        colors=colors,
        point_count=n_points,
    )


def _make_ellipsoid_processed_cloud(
    n_points: int = 5000, radii: tuple = (1.0, 0.7, 0.5)
) -> ProcessedCloud:
    """Create a ProcessedCloud of points on an ellipsoid with outward normals."""
    rng = np.random.default_rng(42)
    raw = rng.standard_normal((n_points, 3))
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-10, None)
    unit = raw / norms

    # Scale by radii to form an ellipsoid
    rx, ry, rz = radii
    points = unit * np.array([[rx, ry, rz]])

    # Outward normals for an ellipsoid: gradient of x²/rx² + y²/ry² + z²/rz² = 1
    normals_raw = points / np.array([[rx**2, ry**2, rz**2]])
    normal_norms = np.linalg.norm(normals_raw, axis=1, keepdims=True)
    normal_norms = np.clip(normal_norms, 1e-10, None)
    normals = normals_raw / normal_norms

    colors = np.zeros((n_points, 3))

    return ProcessedCloud(
        points=points,
        normals=normals,
        colors=colors,
        point_count=n_points,
    )


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------


class TestReconstructMeshSphere:
    """Test reconstruct_mesh with a synthetic sphere point cloud.

    Validates: Requirements 5.1, 5.2, 5.3, 5.4
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        from tests.conftest import get_sphere_mesh, make_sphere_cloud
        self.pipeline = ReconstructionPipeline()
        self.cloud = make_sphere_cloud(n_points=5000)
        self.mesh = get_sphere_mesh(n_points=5000)

    def test_returns_triangle_mesh(self):
        """reconstruct_mesh must return an Open3D TriangleMesh."""
        assert isinstance(self.mesh, o3d.geometry.TriangleMesh)

    def test_mesh_has_triangles(self):
        """Reconstructed mesh must contain triangles."""
        assert len(self.mesh.triangles) > 0

    def test_mesh_has_vertices(self):
        """Reconstructed mesh must contain vertices."""
        assert len(self.mesh.vertices) > 0

    def test_no_degenerate_triangles(self):
        """All triangles must have area > 0 (no degenerate triangles).

        Validates: Requirement 5.3
        """
        vertices = np.asarray(self.mesh.vertices)
        triangles = np.asarray(self.mesh.triangles)
        for tri in triangles:
            v0, v1, v2 = vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]
            area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
            assert area > 0, f"Degenerate triangle found: {tri}"

    def test_no_unreferenced_vertices(self):
        """Every vertex must belong to at least one triangle.

        Validates: Requirement 5.4
        """
        triangles = np.asarray(self.mesh.triangles)
        referenced = set(triangles.flatten())
        n_vertices = len(self.mesh.vertices)
        assert len(referenced) == n_vertices, (
            f"{n_vertices - len(referenced)} unreferenced vertices found"
        )


class TestReconstructMeshEllipsoid:
    """Test reconstruct_mesh with a synthetic ellipsoid point cloud.

    Validates: Requirements 5.1, 5.2
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        from tests.conftest import get_ellipsoid_mesh
        self.pipeline = ReconstructionPipeline()
        self.cloud = _make_ellipsoid_processed_cloud(n_points=5000)
        self.mesh = get_ellipsoid_mesh(n_points=5000)

    def test_returns_triangle_mesh(self):
        """reconstruct_mesh must return an Open3D TriangleMesh."""
        assert isinstance(self.mesh, o3d.geometry.TriangleMesh)

    def test_mesh_has_triangles(self):
        """Reconstructed mesh must contain triangles."""
        assert len(self.mesh.triangles) > 0


class TestReconstructMeshFallback:
    """Test the depth fallback retry logic.

    Validates: Requirements 5.5, 5.6
    """

    def setup_method(self):
        self.pipeline = ReconstructionPipeline()

    def test_fallback_to_depth_7_on_first_failure(self):
        """When depth=9 fails, reconstruction should retry with depth=7.

        Validates: Requirement 5.5
        """
        cloud = _make_sphere_processed_cloud(n_points=5000)
        call_count = {"n": 0}
        original_fn = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson

        def mock_poisson(pcd, **kwargs):
            call_count["n"] += 1
            if kwargs.get("depth", 9) == 9:
                raise RuntimeError("Simulated depth=9 failure")
            return original_fn(pcd, **kwargs)

        with patch.object(
            o3d.geometry.TriangleMesh,
            "create_from_point_cloud_poisson",
            side_effect=mock_poisson,
        ):
            mesh = self.pipeline.reconstruct_mesh(cloud)

        assert isinstance(mesh, o3d.geometry.TriangleMesh)
        assert len(mesh.triangles) > 0
        assert call_count["n"] == 2  # called once at depth=9, once at depth=7

    def test_raises_runtime_error_when_both_depths_fail(self):
        """When both depth=9 and depth=7 fail, a RuntimeError with error report is raised.

        Validates: Requirement 5.6
        """
        cloud = _make_sphere_processed_cloud(n_points=5000)

        def always_fail(pcd, **kwargs):
            raise RuntimeError("Simulated total failure")

        with patch.object(
            o3d.geometry.TriangleMesh,
            "create_from_point_cloud_poisson",
            side_effect=always_fail,
        ):
            with pytest.raises(RuntimeError, match="failed after retry"):
                self.pipeline.reconstruct_mesh(cloud)

    def test_error_report_suggests_rescan(self):
        """Error report should suggest rescanning.

        Validates: Requirement 5.6
        """
        cloud = _make_sphere_processed_cloud(n_points=5000)

        def always_fail(pcd, **kwargs):
            raise RuntimeError("Simulated failure")

        with patch.object(
            o3d.geometry.TriangleMesh,
            "create_from_point_cloud_poisson",
            side_effect=always_fail,
        ):
            with pytest.raises(RuntimeError, match="rescan"):
                self.pipeline.reconstruct_mesh(cloud)


# ---------------------------------------------------------------------------
# Simplify mesh tests
# ---------------------------------------------------------------------------


def _get_high_face_mesh() -> o3d.geometry.TriangleMesh:
    """Get a cached high-face-count mesh (built lazily)."""
    from tests.conftest import get_sphere_mesh
    return get_sphere_mesh(n_points=5000)


class TestSimplifyMesh:
    """Test simplify_mesh with various target face counts.

    Validates: Requirements 6.1, 6.2, 6.3
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        import copy
        self.pipeline = ReconstructionPipeline()
        self.mesh = copy.deepcopy(_get_high_face_mesh())
        self.original_face_count = len(self.mesh.triangles)

    def test_simplify_reduces_face_count(self):
        """Simplification must reduce face count to at most target_faces.

        Validates: Requirement 6.1, 6.3
        """
        target = 500
        assert self.original_face_count > target, (
            "Test precondition: original mesh must have more faces than target"
        )
        simplified = self.pipeline.simplify_mesh(self.mesh, target_faces=target)
        assert len(simplified.triangles) <= target

    def test_simplify_returns_triangle_mesh(self):
        """simplify_mesh must return an Open3D TriangleMesh."""
        simplified = self.pipeline.simplify_mesh(self.mesh, target_faces=500)
        assert isinstance(simplified, o3d.geometry.TriangleMesh)

    def test_simplify_no_unreferenced_vertices(self):
        """Simplified mesh must have no unreferenced vertices.

        Validates: Requirement 6.2
        """
        simplified = self.pipeline.simplify_mesh(self.mesh, target_faces=500)
        triangles = np.asarray(simplified.triangles)
        referenced = set(triangles.flatten())
        n_vertices = len(simplified.vertices)
        assert len(referenced) == n_vertices, (
            f"{n_vertices - len(referenced)} unreferenced vertices found"
        )

    def test_simplify_no_degenerate_triangles(self):
        """Simplified mesh must have no degenerate triangles.

        Validates: Requirement 6.2
        """
        simplified = self.pipeline.simplify_mesh(self.mesh, target_faces=500)
        vertices = np.asarray(simplified.vertices)
        triangles = np.asarray(simplified.triangles)
        for tri in triangles:
            v0, v1, v2 = vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]
            area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
            assert area > 0, f"Degenerate triangle found: {tri}"

    def test_simplify_mesh_below_target_is_noop(self):
        """When mesh already has fewer faces than target, it should pass through.

        Validates: Requirement 6.1 (conditional simplification)
        """
        target = self.original_face_count + 10000
        simplified = self.pipeline.simplify_mesh(self.mesh, target_faces=target)
        assert len(simplified.triangles) <= target

    def test_simplify_default_target(self):
        """simplify_mesh with default target_faces=50000 should work."""
        simplified = self.pipeline.simplify_mesh(self.mesh)
        assert len(simplified.triangles) <= 50000
