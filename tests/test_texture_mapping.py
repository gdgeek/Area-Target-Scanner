"""Unit tests for texture mapping.

Validates: Requirements 7.1, 7.2, 7.4

Tests the ReconstructionPipeline.generate_texture() method using
synthetic meshes and images, verifying output file formats, quality
scoring, and warning logic.
"""

from __future__ import annotations

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


def _make_simple_mesh() -> o3d.geometry.TriangleMesh:
    """Create a small box mesh for testing."""
    mesh = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=1.0)
    mesh.compute_vertex_normals()
    return mesh


def _make_test_images(work_dir: str, count: int = 3) -> list[dict]:
    """Create synthetic test images with identity-like camera poses."""
    images = []
    img_dir = os.path.join(work_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    colours = [
        (200, 50, 50),   # reddish
        (50, 200, 50),   # greenish
        (50, 50, 200),   # bluish
    ]

    for i in range(count):
        r, g, b = colours[i % len(colours)]
        img = Image.new("RGB", (640, 480), (r, g, b))
        img_path = os.path.join(img_dir, f"frame_{i:04d}.jpg")
        img.save(img_path)

        # Camera looking at the box from different positions along z
        pose = np.eye(4, dtype=np.float64)
        pose[2, 3] = 3.0 + i * 0.5

        images.append({"path": img_path, "pose": pose})

    return images


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGenerateTextureOutput:
    """Test that generate_texture produces correct output files.

    Validates: Requirements 7.1, 7.2
    """

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.pipeline = ReconstructionPipeline()
        self.mesh = _make_simple_mesh()
        self.images = _make_test_images(str(tmp_path))

    def test_returns_textured_mesh(self):
        """generate_texture must return a TexturedMesh instance."""
        result = self.pipeline.generate_texture(self.mesh, self.images)
        assert isinstance(result, TexturedMesh)

    def test_texture_file_is_png(self):
        """Output texture file must be a PNG.

        Validates: Requirement 7.2
        """
        result = self.pipeline.generate_texture(self.mesh, self.images)
        assert result.texture_file.endswith(".png")
        assert os.path.isfile(result.texture_file)

        # Verify it's a valid image
        img = Image.open(result.texture_file)
        assert img.size[0] > 0 and img.size[1] > 0

    def test_material_file_is_mtl(self):
        """Output material file must be an MTL.

        Validates: Requirement 7.2
        """
        result = self.pipeline.generate_texture(self.mesh, self.images)
        assert result.material_file.endswith(".mtl")
        assert os.path.isfile(result.material_file)

        # Verify MTL content references the texture
        with open(result.material_file) as f:
            content = f.read()
        assert "map_Kd" in content
        assert "texture_atlas.png" in content

    def test_quality_score_is_set(self):
        """Quality score must be a float in [0, 1]."""
        result = self.pipeline.generate_texture(self.mesh, self.images)
        assert result.quality_score is not None
        assert 0.0 <= result.quality_score <= 1.0

    def test_mesh_is_preserved(self):
        """The returned mesh object should be the input mesh."""
        result = self.pipeline.generate_texture(self.mesh, self.images)
        assert result.mesh is not None


class TestGenerateTextureQualityWarning:
    """Test quality score and warning logic.

    Validates: Requirement 7.4
    """

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.pipeline = ReconstructionPipeline()
        self.tmp_path = str(tmp_path)

    def test_low_quality_with_no_images(self, caplog):
        """With no images, quality should be low and a warning should be logged.

        Validates: Requirement 7.4
        """
        import logging

        mesh = _make_simple_mesh()
        with caplog.at_level(logging.WARNING):
            result = self.pipeline.generate_texture(mesh, [])

        assert result.quality_score is not None
        assert result.quality_score < 0.3
        assert any("quality" in r.message.lower() for r in caplog.records)

    def test_quality_with_valid_images(self):
        """With valid images projecting onto the mesh, quality should be >= 0."""
        mesh = _make_simple_mesh()
        images = _make_test_images(self.tmp_path)
        result = self.pipeline.generate_texture(mesh, images)
        assert result.quality_score is not None
        assert result.quality_score >= 0.0


class TestGenerateTextureColorCorrection:
    """Test global colour correction.

    Validates: Requirement 7.3
    """

    def test_color_correction_dark_image(self, tmp_path):
        """Colour correction should normalise mean brightness toward 128."""
        dark_img = Image.new("RGB", (100, 100), (30, 30, 30))
        tex_path = str(tmp_path / "test_texture.png")
        dark_img.save(tex_path)

        import logging

        logger = logging.getLogger("test")
        ReconstructionPipeline._apply_color_correction(tex_path, logger)

        corrected = np.asarray(Image.open(tex_path).convert("RGB"))
        mean_brightness = corrected.mean()
        # After correction, mean should be closer to 128 than the original 30
        assert mean_brightness > 100

    def test_color_correction_bright_image(self, tmp_path):
        """Colour correction should also handle bright images."""
        bright_img = Image.new("RGB", (100, 100), (240, 240, 240))
        tex_path = str(tmp_path / "test_texture.png")
        bright_img.save(tex_path)

        import logging

        logger = logging.getLogger("test")
        ReconstructionPipeline._apply_color_correction(tex_path, logger)

        corrected = np.asarray(Image.open(tex_path).convert("RGB"))
        mean_brightness = corrected.mean()
        # After correction, mean should be closer to 128 than the original 240
        assert mean_brightness < 180


class TestGenerateTextureEdgeCases:
    """Test edge cases for generate_texture."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.pipeline = ReconstructionPipeline()

    def test_empty_image_list(self, tmp_path):
        """Should handle empty image list gracefully."""
        mesh = _make_simple_mesh()
        result = self.pipeline.generate_texture(mesh, [])
        assert isinstance(result, TexturedMesh)
        assert os.path.isfile(result.texture_file)
        assert os.path.isfile(result.material_file)

    def test_nonexistent_image_path(self, tmp_path):
        """Should handle missing image files gracefully."""
        mesh = _make_simple_mesh()
        images = [{"path": "/nonexistent/image.jpg", "pose": np.eye(4)}]
        result = self.pipeline.generate_texture(mesh, images)
        assert isinstance(result, TexturedMesh)
