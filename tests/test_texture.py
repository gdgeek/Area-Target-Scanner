"""Unit tests for generate_texture().

Validates: Requirements 7.1, 7.2, 7.4

Tests output file format correctness (PNG, MTL), quality score range,
quality warning logic, and empty-images edge case.
"""

from __future__ import annotations

import logging
import os

import numpy as np
import open3d as o3d
import pytest
from PIL import Image

from processing_pipeline.models import TexturedMesh
from processing_pipeline.pipeline import ReconstructionPipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_sphere_mesh() -> o3d.geometry.TriangleMesh:
    """Create a simple sphere mesh for testing."""
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=10)
    mesh.compute_vertex_normals()
    return mesh


def _create_synthetic_images(directory: str, count: int = 3) -> list[dict]:
    """Create synthetic JPEG images with camera poses looking at the origin.

    Each image is a solid colour, and the camera is placed along the z-axis
    at increasing distances so that the sphere is visible.
    """
    img_dir = os.path.join(directory, "synth_images")
    os.makedirs(img_dir, exist_ok=True)

    colours = [(180, 60, 60), (60, 180, 60), (60, 60, 180)]
    images: list[dict] = []

    for i in range(count):
        r, g, b = colours[i % len(colours)]
        img = Image.new("RGB", (320, 240), (r, g, b))
        path = os.path.join(img_dir, f"img_{i:04d}.jpg")
        img.save(path)

        pose = np.eye(4, dtype=np.float64)
        pose[2, 3] = 3.0 + i * 0.5  # camera along +z

        images.append({"path": path, "pose": pose})

    return images


# ---------------------------------------------------------------------------
# Tests — Output file format correctness
# ---------------------------------------------------------------------------


class TestTextureOutputFormat:
    """Verify that generate_texture produces valid PNG and MTL files.

    Validates: Requirements 7.1, 7.2
    """

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.pipeline = ReconstructionPipeline()
        self.mesh = _create_sphere_mesh()
        self.images = _create_synthetic_images(str(tmp_path))
        self.result = self.pipeline.generate_texture(self.mesh, self.images)

    def test_returns_textured_mesh_instance(self):
        """generate_texture must return a TexturedMesh."""
        assert isinstance(self.result, TexturedMesh)

    def test_texture_file_exists_and_is_png(self):
        """Texture file must exist and be a valid PNG image."""
        assert self.result.texture_file.endswith(".png")
        assert os.path.isfile(self.result.texture_file)

        img = Image.open(self.result.texture_file)
        assert img.format == "PNG"
        assert img.size[0] > 0 and img.size[1] > 0

    def test_material_file_exists_and_is_mtl(self):
        """Material file must exist, end with .mtl, and reference the texture."""
        assert self.result.material_file.endswith(".mtl")
        assert os.path.isfile(self.result.material_file)

        with open(self.result.material_file) as f:
            content = f.read()
        assert "newmtl" in content
        assert "map_Kd" in content
        assert "texture_atlas.png" in content


# ---------------------------------------------------------------------------
# Tests — Quality score
# ---------------------------------------------------------------------------


class TestTextureQualityScore:
    """Verify quality score is in [0, 1].

    Validates: Requirements 7.2, 7.4
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        self.pipeline = ReconstructionPipeline()

    def test_quality_score_in_valid_range(self, tmp_path):
        """Quality score must be a float in [0.0, 1.0]."""
        mesh = _create_sphere_mesh()
        images = _create_synthetic_images(str(tmp_path))
        result = self.pipeline.generate_texture(mesh, images)

        assert result.quality_score is not None
        assert isinstance(result.quality_score, float)
        assert 0.0 <= result.quality_score <= 1.0

    def test_quality_score_with_empty_images(self, tmp_path):
        """Quality score must still be in [0, 1] even with no images."""
        mesh = _create_sphere_mesh()
        result = self.pipeline.generate_texture(mesh, [])

        assert result.quality_score is not None
        assert 0.0 <= result.quality_score <= 1.0


# ---------------------------------------------------------------------------
# Tests — Quality warning
# ---------------------------------------------------------------------------


class TestTextureQualityWarning:
    """Verify that a quality warning is logged when score < 0.3.

    Validates: Requirement 7.4
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        self.pipeline = ReconstructionPipeline()

    def test_warning_logged_when_score_below_threshold(self, tmp_path, caplog):
        """A warning containing 'quality' must be logged when score < 0.3."""
        mesh = _create_sphere_mesh()
        # No images → fallback produces a mostly-grey atlas → low quality
        with caplog.at_level(logging.WARNING):
            result = self.pipeline.generate_texture(mesh, [])

        assert result.quality_score < 0.3
        warning_messages = [
            r.message for r in caplog.records if r.levelno >= logging.WARNING
        ]
        assert any("quality" in msg.lower() for msg in warning_messages), (
            f"Expected a quality warning in log records, got: {warning_messages}"
        )


# ---------------------------------------------------------------------------
# Tests — Empty images edge case
# ---------------------------------------------------------------------------


class TestTextureEmptyImages:
    """Verify generate_texture handles an empty images list gracefully.

    Validates: Requirements 7.1, 7.2
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        self.pipeline = ReconstructionPipeline()

    def test_empty_images_produces_output_files(self, tmp_path):
        """With an empty images list, output PNG and MTL must still be created."""
        mesh = _create_sphere_mesh()
        result = self.pipeline.generate_texture(mesh, [])

        assert isinstance(result, TexturedMesh)
        assert os.path.isfile(result.texture_file)
        assert os.path.isfile(result.material_file)

        # PNG must be openable
        img = Image.open(result.texture_file)
        assert img.size[0] > 0 and img.size[1] > 0

        # MTL must reference the texture
        with open(result.material_file) as f:
            content = f.read()
        assert "map_Kd" in content
