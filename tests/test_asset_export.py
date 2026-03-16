"""Unit tests for asset bundle export.

Validates: Requirements 9.1, 9.2, 9.3

Tests the ReconstructionPipeline.export_asset_bundle() method, verifying
output directory structure, manifest.json field completeness and correctness,
and file reference consistency.
"""

from __future__ import annotations

import json
import os
from datetime import datetime

import numpy as np
import open3d as o3d
import pytest
from PIL import Image

from processing_pipeline.models import (
    FeatureDatabase,
    KeyframeData,
    TexturedMesh,
)
from processing_pipeline.pipeline import ReconstructionPipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

EXPECTED_FILES = [
    "mesh.obj",
    "mesh.mtl",
    "texture_atlas.png",
    "features.db",
    "manifest.json",
]


def _make_sphere_mesh() -> o3d.geometry.TriangleMesh:
    """Create a simple sphere mesh (avoids Poisson segfaults)."""
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=10)
    mesh.compute_vertex_normals()
    return mesh


def _make_textured_mesh(tmp_path: str) -> TexturedMesh:
    """Create a TexturedMesh with temporary texture and material files."""
    mesh = _make_sphere_mesh()

    # Write a small texture PNG
    tex_path = os.path.join(tmp_path, "src_texture.png")
    img = Image.new("RGB", (64, 64), (128, 100, 80))
    img.save(tex_path)

    # Write a minimal MTL file
    mtl_path = os.path.join(tmp_path, "src_mesh.mtl")
    with open(mtl_path, "w") as f:
        f.write("# Material\n")
        f.write("newmtl material0\n")
        f.write("map_Kd texture_atlas.png\n")

    return TexturedMesh(
        mesh=mesh,
        texture_file=tex_path,
        material_file=mtl_path,
        quality_score=0.75,
    )


def _make_feature_database(n_keyframes: int = 3) -> FeatureDatabase:
    """Create a simple FeatureDatabase with synthetic keyframes."""
    keyframes = []
    for i in range(n_keyframes):
        n_features = 25
        kf = KeyframeData(
            image_id=i,
            keypoints=[(float(x), float(x + 1)) for x in range(n_features)],
            descriptors=np.random.randint(
                0, 256, size=(n_features, 32), dtype=np.uint8
            ),
            points_3d=[
                (float(x) * 0.1, float(x) * 0.2, float(x) * 0.3)
                for x in range(n_features)
            ],
            camera_pose=np.eye(4, dtype=np.float64),
        )
        keyframes.append(kf)

    return FeatureDatabase(
        keyframes=keyframes,
        global_descriptors=None,
        vocabulary=None,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAssetBundleDirectoryStructure:
    """Test that export_asset_bundle produces all expected files.

    Validates: Requirement 9.1, 9.3
    """

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.pipeline = ReconstructionPipeline()
        self.src_dir = str(tmp_path / "src")
        os.makedirs(self.src_dir, exist_ok=True)
        self.output_dir = str(tmp_path / "asset_bundle")
        self.textured_mesh = _make_textured_mesh(self.src_dir)
        self.feature_db = _make_feature_database(n_keyframes=3)

    def test_all_expected_files_exist(self):
        """All five asset files must exist in the output directory.

        Validates: Requirement 9.1
        """
        self.pipeline.export_asset_bundle(
            self.textured_mesh, self.feature_db, self.output_dir
        )
        for fname in EXPECTED_FILES:
            fpath = os.path.join(self.output_dir, fname)
            assert os.path.isfile(fpath), f"Expected file missing: {fname}"

    def test_mesh_obj_is_nonempty(self):
        """mesh.obj must be a non-empty file."""
        self.pipeline.export_asset_bundle(
            self.textured_mesh, self.feature_db, self.output_dir
        )
        obj_path = os.path.join(self.output_dir, "mesh.obj")
        assert os.path.getsize(obj_path) > 0

    def test_texture_atlas_is_valid_png(self):
        """texture_atlas.png must be a valid PNG image."""
        self.pipeline.export_asset_bundle(
            self.textured_mesh, self.feature_db, self.output_dir
        )
        tex_path = os.path.join(self.output_dir, "texture_atlas.png")
        img = Image.open(tex_path)
        assert img.format == "PNG"
        assert img.size[0] > 0 and img.size[1] > 0

    def test_features_db_is_nonempty(self):
        """features.db must be a non-empty SQLite file."""
        self.pipeline.export_asset_bundle(
            self.textured_mesh, self.feature_db, self.output_dir
        )
        db_path = os.path.join(self.output_dir, "features.db")
        assert os.path.getsize(db_path) > 0


class TestManifestFieldCompleteness:
    """Test that manifest.json contains all required fields.

    Validates: Requirement 9.2
    """

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.pipeline = ReconstructionPipeline()
        self.src_dir = str(tmp_path / "src")
        os.makedirs(self.src_dir, exist_ok=True)
        self.output_dir = str(tmp_path / "asset_bundle")
        self.textured_mesh = _make_textured_mesh(self.src_dir)
        self.feature_db = _make_feature_database(n_keyframes=4)
        self.pipeline.export_asset_bundle(
            self.textured_mesh, self.feature_db, self.output_dir
        )
        manifest_path = os.path.join(self.output_dir, "manifest.json")
        with open(manifest_path, encoding="utf-8") as f:
            self.manifest = json.load(f)

    def test_has_all_required_fields(self):
        """manifest.json must contain every required field.

        Validates: Requirement 9.2
        """
        required_fields = [
            "version",
            "name",
            "meshFile",
            "textureFile",
            "featureDbFile",
            "bounds",
            "keyframeCount",
            "featureType",
            "createdAt",
        ]
        for field in required_fields:
            assert field in self.manifest, f"Missing manifest field: {field}"

    def test_version_is_1_0(self):
        """version must be '1.0'."""
        assert self.manifest["version"] == "1.0"

    def test_feature_type_is_orb(self):
        """featureType must be 'ORB'."""
        assert self.manifest["featureType"] == "ORB"

    def test_mesh_file_value(self):
        """meshFile must be 'mesh.obj'."""
        assert self.manifest["meshFile"] == "mesh.obj"

    def test_texture_file_value(self):
        """textureFile must be 'texture_atlas.png'."""
        assert self.manifest["textureFile"] == "texture_atlas.png"

    def test_feature_db_file_value(self):
        """featureDbFile must be 'features.db'."""
        assert self.manifest["featureDbFile"] == "features.db"

    def test_bounds_has_min_and_max(self):
        """bounds must have 'min' and 'max' arrays with 3 elements each."""
        bounds = self.manifest["bounds"]
        assert "min" in bounds
        assert "max" in bounds
        assert len(bounds["min"]) == 3
        assert len(bounds["max"]) == 3

    def test_bounds_values_are_numeric(self):
        """bounds min and max values must be numeric."""
        for val in self.manifest["bounds"]["min"]:
            assert isinstance(val, (int, float))
        for val in self.manifest["bounds"]["max"]:
            assert isinstance(val, (int, float))

    def test_keyframe_count_matches(self):
        """keyframeCount must match the actual number of keyframes."""
        assert self.manifest["keyframeCount"] == 4

    def test_created_at_is_valid_iso8601(self):
        """createdAt must be a valid ISO 8601 timestamp."""
        created_at = self.manifest["createdAt"]
        # datetime.fromisoformat handles ISO 8601 strings
        parsed = datetime.fromisoformat(created_at)
        assert parsed is not None


class TestManifestKeyframeCount:
    """Test keyframeCount with different keyframe counts.

    Validates: Requirement 9.2
    """

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.pipeline = ReconstructionPipeline()
        self.src_dir = str(tmp_path / "src")
        os.makedirs(self.src_dir, exist_ok=True)
        self.output_dir = str(tmp_path / "asset_bundle")
        self.textured_mesh = _make_textured_mesh(self.src_dir)

    def test_zero_keyframes(self):
        """keyframeCount should be 0 when database has no keyframes."""
        db = _make_feature_database(n_keyframes=0)
        self.pipeline.export_asset_bundle(
            self.textured_mesh, db, self.output_dir
        )
        manifest_path = os.path.join(self.output_dir, "manifest.json")
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
        assert manifest["keyframeCount"] == 0

    def test_many_keyframes(self):
        """keyframeCount should match when database has many keyframes."""
        db = _make_feature_database(n_keyframes=10)
        self.pipeline.export_asset_bundle(
            self.textured_mesh, db, self.output_dir
        )
        manifest_path = os.path.join(self.output_dir, "manifest.json")
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
        assert manifest["keyframeCount"] == 10
