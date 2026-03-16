"""Property-based tests for point cloud preprocessing.

**Validates: Requirements 4.5, 4.6**

Property P1: Point cloud preprocessing preserves spatial consistency.
  - All processed points lie within the convex hull of the original point cloud.
  - All normals are unit vectors (|normal| ≈ 1.0).
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import open3d as o3d
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st
from scipy.spatial import ConvexHull

from processing_pipeline.pipeline import ReconstructionPipeline


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

@st.composite
def sphere_point_cloud(draw):
    """Generate a synthetic point cloud of points on/near a sphere.

    Produces between 1500 and 3000 points with small radial noise so the
    cloud is dense enough for Open3D preprocessing (>= 1000 after outlier
    removal and downsampling) while keeping test runtime reasonable.
    """
    n_points = draw(st.integers(min_value=1500, max_value=3000))
    radius = draw(st.floats(min_value=0.5, max_value=5.0))
    noise_scale = draw(st.floats(min_value=0.0, max_value=0.02))

    # Random points on a unit sphere via normalised Gaussian vectors
    rng = np.random.default_rng(draw(st.integers(min_value=0, max_value=2**32 - 1)))
    raw = rng.standard_normal((n_points, 3))
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-10, None)
    unit = raw / norms

    # Scale to desired radius and add small noise
    points = unit * radius + rng.normal(scale=noise_scale, size=(n_points, 3))

    return points


@st.composite
def cube_point_cloud(draw):
    """Generate a synthetic point cloud of points on/near a cube surface.

    Produces between 1500 and 3000 points distributed on the six faces of
    a cube with small noise.
    """
    n_points = draw(st.integers(min_value=1500, max_value=3000))
    half_size = draw(st.floats(min_value=0.5, max_value=5.0))
    noise_scale = draw(st.floats(min_value=0.0, max_value=0.02))

    rng = np.random.default_rng(draw(st.integers(min_value=0, max_value=2**32 - 1)))

    points_per_face = n_points // 6
    remainder = n_points - points_per_face * 6
    faces = []

    for axis in range(3):
        for sign in (-1.0, 1.0):
            count = points_per_face + (1 if remainder > 0 else 0)
            if remainder > 0:
                remainder -= 1
            # Two free coordinates in [-half_size, half_size], one fixed
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

    points = np.vstack(faces)
    points += rng.normal(scale=noise_scale, size=points.shape)
    return points


# Combine both generators
synthetic_point_cloud = st.one_of(sphere_point_cloud(), cube_point_cloud())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_ply(points: np.ndarray, path: str) -> None:
    """Save an (N, 3) array of points as a PLY file via Open3D."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(path, pcd)


def _points_inside_convex_hull(hull: ConvexHull, points: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    """Return a boolean mask indicating which *points* lie inside *hull*.

    Uses the half-plane representation: a point is inside the convex hull
    iff  A @ point + b <= tol  for every inequality.
    """
    A = hull.equations[:, :-1]
    b = hull.equations[:, -1]
    return np.all(A @ points.T + b[:, None] <= tol, axis=0)


# ---------------------------------------------------------------------------
# Property test
# ---------------------------------------------------------------------------

class TestPointCloudPreprocessingProperties:
    """Property P1: Point cloud preprocessing preserves spatial consistency.

    **Validates: Requirements 4.5, 4.6**
    """

    @given(points=synthetic_point_cloud)
    @settings(
        max_examples=5,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_p1_spatial_consistency_and_unit_normals(self, points: np.ndarray):
        """All processed points lie within the original convex hull and all
        normals are unit vectors.

        **Validates: Requirements 4.5, 4.6**
        """
        pipeline = ReconstructionPipeline()

        with tempfile.TemporaryDirectory() as tmpdir:
            ply_path = os.path.join(tmpdir, "cloud.ply")
            _save_ply(points, ply_path)

            result = pipeline.process_point_cloud(ply_path)

        # --- Convex hull check (Requirement 4.6) ---
        original_hull = ConvexHull(points)
        inside = _points_inside_convex_hull(original_hull, result.points, tol=1e-4)
        assert inside.all(), (
            f"{(~inside).sum()} of {len(result.points)} processed points "
            f"lie outside the original convex hull"
        )

        # --- Unit normal check (Requirement 4.5) ---
        norms = np.linalg.norm(result.normals, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-3), (
            f"Normal magnitudes deviate from 1.0: "
            f"min={norms.min():.6f}, max={norms.max():.6f}"
        )
