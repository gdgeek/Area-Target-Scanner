"""Shared fixtures and lazy mesh cache for tests.

Open3D's Poisson surface reconstruction segfaults on macOS Python 3.9
when called too many times in a single process.  This module provides a
lazy, process-wide cache so that each distinct mesh is built at most once
and only when a test actually needs it (not at import/collection time).
"""

from __future__ import annotations

import numpy as np
import open3d as o3d

from processing_pipeline.models import ProcessedCloud
from processing_pipeline.pipeline import ReconstructionPipeline

_cache: dict[str, o3d.geometry.TriangleMesh] = {}


def get_sphere_mesh(n_points: int = 5000, radius: float = 1.0) -> o3d.geometry.TriangleMesh:
    """Return a cached sphere mesh (built lazily on first call)."""
    key = f"sphere_{n_points}_{radius}"
    if key not in _cache:
        rng = np.random.default_rng(42)
        raw = rng.standard_normal((n_points, 3))
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-10, None)
        unit = raw / norms
        points = unit * radius
        normals = unit.copy()
        colors = np.zeros((n_points, 3))
        cloud = ProcessedCloud(points=points, normals=normals, colors=colors, point_count=n_points)
        _cache[key] = ReconstructionPipeline().reconstruct_mesh(cloud)
    return _cache[key]


def get_ellipsoid_mesh(
    n_points: int = 5000, radii: tuple = (2.0, 1.0, 0.7),
) -> o3d.geometry.TriangleMesh:
    """Return a cached ellipsoid mesh (built lazily on first call)."""
    key = f"ellipsoid_{n_points}_{radii}"
    if key not in _cache:
        rng = np.random.default_rng(123)
        raw = rng.standard_normal((n_points, 3))
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-10, None)
        unit = raw / norms
        rx, ry, rz = radii
        points = unit * np.array([[rx, ry, rz]])
        normals_raw = points / np.array([[rx**2, ry**2, rz**2]])
        normal_norms = np.linalg.norm(normals_raw, axis=1, keepdims=True)
        normal_norms = np.clip(normal_norms, 1e-10, None)
        normals = normals_raw / normal_norms
        colors = np.zeros((n_points, 3))
        cloud = ProcessedCloud(points=points, normals=normals, colors=colors, point_count=n_points)
        _cache[key] = ReconstructionPipeline().reconstruct_mesh(cloud)
    return _cache[key]


def make_sphere_cloud(n_points: int = 5000, radius: float = 1.0) -> ProcessedCloud:
    """Create a ProcessedCloud of points on a sphere with outward normals."""
    rng = np.random.default_rng(42)
    raw = rng.standard_normal((n_points, 3))
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-10, None)
    points = (raw / norms) * radius
    normals = raw / norms
    colors = np.zeros((n_points, 3))
    return ProcessedCloud(points=points, normals=normals, colors=colors, point_count=n_points)
