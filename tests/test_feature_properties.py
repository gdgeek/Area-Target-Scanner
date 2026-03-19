"""Property-based tests for feature extraction.

**Validates: Requirements 8.7**

Property P4: Feature database 2D-3D correspondence consistency.
  For every feature in the database, projecting its 3D point back to 2D
  using the keyframe's camera pose and intrinsics yields a point within
  5.0 pixels of the original 2D keypoint.
"""

from __future__ import annotations

import copy
import os
import tempfile

import cv2
import numpy as np
import open3d as o3d
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from processing_pipeline.models import ProcessedCloud
from processing_pipeline.feature_extraction import build_feature_database


# ---------------------------------------------------------------------------
# Module-level mesh (built once to avoid Open3D Poisson segfault issues)
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_textured_image(
    width: int = 640, height: int = 480, seed: int = 42
) -> np.ndarray:
    """Generate a synthetic image with enough texture for ORB detection."""
    rng = np.random.RandomState(seed)
    img = rng.randint(0, 256, (height, width), dtype=np.uint8)
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


# ---------------------------------------------------------------------------
# Strategy: camera positions with slight variation
# ---------------------------------------------------------------------------


@st.composite
def camera_offset(draw):
    """Generate small camera position offsets around a base position.

    The base camera is at (0, 0, 3) looking at the origin.
    We vary tx, ty slightly and tz around 3.0 to keep the sphere visible.
    """
    tx = draw(st.floats(min_value=-0.3, max_value=0.3))
    ty = draw(st.floats(min_value=-0.3, max_value=0.3))
    tz = draw(st.floats(min_value=2.5, max_value=3.5))
    return (tx, ty, tz)


def _make_camera_pose(tx: float, ty: float, tz: float) -> np.ndarray:
    """Camera-to-world pose looking at the origin from the given position."""
    pose = np.eye(4, dtype=np.float64)
    pose[0, 3] = tx
    pose[1, 3] = ty
    pose[2, 3] = tz
    return pose


# ---------------------------------------------------------------------------
# Property test
# ---------------------------------------------------------------------------


class TestFeatureDatabaseProperties:
    """Property P4: Feature database 2D-3D correspondence consistency.

    **Validates: Requirements 8.7**
    """

    @given(offsets=st.lists(camera_offset(), min_size=2, max_size=4))
    @settings(
        max_examples=5,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_p4_2d_3d_reprojection_within_5_pixels(
        self, offsets: list[tuple[float, float, float]]
    ):
        """For every feature, projecting its 3D point back to 2D must be
        within 5.0 pixels of the stored 2D keypoint.

        **Validates: Requirements 8.7**
        """
        mesh = copy.deepcopy(_build_sphere_mesh())

        with tempfile.TemporaryDirectory() as tmpdir:
            # Build images with varied camera poses
            images = []
            for i, (tx, ty, tz) in enumerate(offsets):
                img = _make_textured_image(seed=42 + i)
                path = _save_image(img, tmpdir, f"frame_{i:04d}.png")
                pose = _make_camera_pose(tx, ty, tz)
                images.append({"path": path, "pose": pose})

            db = build_feature_database(images, mesh)

        # If no keyframes were produced, the property holds vacuously
        if not db.keyframes:
            return

        # Check every feature in every keyframe
        for kf in db.keyframes:
            pose = kf.camera_pose  # 4x4 camera-to-world

            # Read image dimensions from the first image to compute intrinsics
            # (all synthetic images are 640x480)
            w, h = 640, 480
            fx = fy = max(w, h) * 0.8
            cx, cy = w / 2.0, h / 2.0

            # world-to-camera transform
            world_to_cam = np.linalg.inv(pose)

            for j, (x3d, y3d, z3d) in enumerate(kf.points_3d):
                # Project 3D point to camera frame
                p_world = np.array([x3d, y3d, z3d, 1.0])
                p_cam = world_to_cam @ p_world

                # Skip points behind the camera (shouldn't happen, but be safe)
                if p_cam[2] <= 0:
                    continue

                # Project to 2D pixel coordinates
                u = fx * p_cam[0] / p_cam[2] + cx
                v = fy * p_cam[1] / p_cam[2] + cy

                # Original 2D keypoint
                orig_x, orig_y = kf.keypoints[j]

                dist = np.sqrt((u - orig_x) ** 2 + (v - orig_y) ** 2)
                assert dist < 5.0, (
                    f"Keyframe {kf.image_id}, feature {j}: "
                    f"reprojection distance {dist:.2f} px >= 5.0 px. "
                    f"3D=({x3d:.3f}, {y3d:.3f}, {z3d:.3f}), "
                    f"projected=({u:.1f}, {v:.1f}), "
                    f"original=({orig_x:.1f}, {orig_y:.1f})"
                )
