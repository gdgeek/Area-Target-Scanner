"""Unit tests for ReconstructionPipeline.run() and CLI entry point.

Validates: Requirements 4.1, 5.1, 6.1, 7.1, 8.1, 9.1
"""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from processing_pipeline.models import (
    FeatureDatabase,
    KeyframeData,
    ProcessedCloud,
    TexturedMesh,
)
from processing_pipeline.pipeline import ReconstructionPipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_poses_json(scan_dir: str, n_frames: int = 2) -> None:
    """Write a minimal poses.json with n_frames entries."""
    os.makedirs(os.path.join(scan_dir, "images"), exist_ok=True)
    frames = []
    for i in range(n_frames):
        img_name = f"images/frame_{i:04d}.jpg"
        # Create a dummy image file
        img_path = os.path.join(scan_dir, img_name)
        with open(img_path, "wb") as f:
            f.write(b"\xff\xd8\xff\xe0")  # minimal JPEG header
        # Identity transform in column-major order (16 floats)
        transform = np.eye(4, dtype=np.float64).flatten(order="F").tolist()
        frames.append({
            "index": i,
            "timestamp": float(i) * 0.5,
            "imageFile": img_name,
            "transform": transform,
        })
    poses = {"frames": frames}
    with open(os.path.join(scan_dir, "poses.json"), "w") as f:
        json.dump(poses, f)


def _write_dummy_ply(scan_dir: str) -> None:
    """Write a minimal PLY file."""
    ply_path = os.path.join(scan_dir, "pointcloud.ply")
    with open(ply_path, "w") as f:
        f.write("ply\nformat ascii 1.0\nelement vertex 3\n")
        f.write("property float x\nproperty float y\nproperty float z\nend_header\n")
        f.write("0 0 0\n1 0 0\n0 1 0\n")


def _make_mock_cloud() -> ProcessedCloud:
    return ProcessedCloud(
        points=np.random.rand(100, 3),
        normals=np.random.rand(100, 3),
        colors=np.random.rand(100, 3),
        point_count=100,
    )


def _make_mock_feature_db() -> FeatureDatabase:
    kf = KeyframeData(
        image_id=0,
        keypoints=[(float(x), float(x)) for x in range(25)],
        descriptors=np.random.randint(0, 256, (25, 32), dtype=np.uint8),
        points_3d=[(0.0, 0.0, 0.0)] * 25,
        camera_pose=np.eye(4),
    )
    return FeatureDatabase(keyframes=[kf])


# ---------------------------------------------------------------------------
# Tests for run()
# ---------------------------------------------------------------------------


class TestRunMissingFiles:
    """Test run() raises errors for missing input files."""

    def test_missing_ply_raises(self, tmp_path):
        scan_dir = str(tmp_path / "scan")
        os.makedirs(scan_dir)
        _write_poses_json(scan_dir)
        # No pointcloud.ply
        pipeline = ReconstructionPipeline()
        with pytest.raises(FileNotFoundError, match="pointcloud.ply"):
            pipeline.run(scan_dir, str(tmp_path / "out"))

    def test_missing_poses_raises(self, tmp_path):
        scan_dir = str(tmp_path / "scan")
        os.makedirs(scan_dir)
        _write_dummy_ply(scan_dir)
        # No poses.json
        pipeline = ReconstructionPipeline()
        with pytest.raises(FileNotFoundError, match="poses.json"):
            pipeline.run(scan_dir, str(tmp_path / "out"))

    def test_empty_frames_raises(self, tmp_path):
        scan_dir = str(tmp_path / "scan")
        os.makedirs(scan_dir)
        _write_dummy_ply(scan_dir)
        with open(os.path.join(scan_dir, "poses.json"), "w") as f:
            json.dump({"frames": []}, f)
        pipeline = ReconstructionPipeline()
        with pytest.raises(ValueError, match="no frames"):
            pipeline.run(scan_dir, str(tmp_path / "out"))


class TestRunPipelineOrchestration:
    """Test that run() calls all pipeline steps in order with correct args.

    Validates: Requirements 4.1, 5.1, 6.1, 7.1, 8.1, 9.1
    """

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.scan_dir = str(tmp_path / "scan")
        self.output_dir = str(tmp_path / "output")
        os.makedirs(self.scan_dir)
        _write_poses_json(self.scan_dir, n_frames=2)
        _write_dummy_ply(self.scan_dir)

    @patch.object(ReconstructionPipeline, "export_asset_bundle")
    @patch.object(ReconstructionPipeline, "build_feature_database")
    @patch.object(ReconstructionPipeline, "generate_texture")
    @patch.object(ReconstructionPipeline, "simplify_mesh")
    @patch.object(ReconstructionPipeline, "reconstruct_mesh")
    @patch.object(ReconstructionPipeline, "process_point_cloud")
    def test_all_steps_called_in_order(
        self,
        mock_ppc,
        mock_rm,
        mock_sm,
        mock_gt,
        mock_bfd,
        mock_eab,
    ):
        """run() must call all 6 pipeline steps in sequence."""
        import open3d as o3d

        mock_cloud = _make_mock_cloud()
        mock_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.5)
        mock_simplified = o3d.geometry.TriangleMesh.create_sphere(radius=0.5)
        mock_textured = TexturedMesh(
            mesh=mock_simplified,
            texture_file="/tmp/tex.png",
            material_file="/tmp/mesh.mtl",
            quality_score=0.8,
        )
        mock_features = _make_mock_feature_db()

        mock_ppc.return_value = mock_cloud
        mock_rm.return_value = mock_mesh
        mock_sm.return_value = mock_simplified
        mock_gt.return_value = mock_textured
        mock_bfd.return_value = mock_features

        pipeline = ReconstructionPipeline()
        pipeline.run(self.scan_dir, self.output_dir)

        # Verify each step was called exactly once
        mock_ppc.assert_called_once()
        mock_rm.assert_called_once_with(mock_cloud)
        mock_sm.assert_called_once_with(mock_mesh)
        mock_gt.assert_called_once()
        mock_bfd.assert_called_once()
        mock_eab.assert_called_once()

        # Verify export_asset_bundle received the correct output_dir
        _, kwargs = mock_eab.call_args
        if not kwargs:
            args = mock_eab.call_args[0]
            assert args[2] == self.output_dir
        else:
            assert kwargs.get("output_dir") == self.output_dir

    @patch.object(ReconstructionPipeline, "export_asset_bundle")
    @patch.object(ReconstructionPipeline, "build_feature_database")
    @patch.object(ReconstructionPipeline, "generate_texture")
    @patch.object(ReconstructionPipeline, "simplify_mesh")
    @patch.object(ReconstructionPipeline, "reconstruct_mesh")
    @patch.object(ReconstructionPipeline, "process_point_cloud")
    def test_images_list_built_from_poses(
        self,
        mock_ppc,
        mock_rm,
        mock_sm,
        mock_gt,
        mock_bfd,
        mock_eab,
    ):
        """run() must build images list with correct paths and poses from poses.json."""
        import open3d as o3d

        mock_ppc.return_value = _make_mock_cloud()
        mock_rm.return_value = o3d.geometry.TriangleMesh.create_sphere()
        mock_sm.return_value = o3d.geometry.TriangleMesh.create_sphere()
        mock_gt.return_value = TexturedMesh(
            mesh=o3d.geometry.TriangleMesh.create_sphere(),
            texture_file="/tmp/t.png",
            material_file="/tmp/m.mtl",
        )
        mock_bfd.return_value = _make_mock_feature_db()

        pipeline = ReconstructionPipeline()
        pipeline.run(self.scan_dir, self.output_dir)

        # Check generate_texture received images with correct structure
        gt_args = mock_gt.call_args[0]
        images = gt_args[1]
        assert len(images) == 2
        for img in images:
            assert "path" in img
            assert "pose" in img
            assert img["pose"].shape == (4, 4)

    @patch.object(ReconstructionPipeline, "export_asset_bundle")
    @patch.object(ReconstructionPipeline, "build_feature_database")
    @patch.object(ReconstructionPipeline, "generate_texture")
    @patch.object(ReconstructionPipeline, "simplify_mesh")
    @patch.object(ReconstructionPipeline, "reconstruct_mesh")
    @patch.object(ReconstructionPipeline, "process_point_cloud")
    def test_transform_column_major_reshape(
        self,
        mock_ppc,
        mock_rm,
        mock_sm,
        mock_gt,
        mock_bfd,
        mock_eab,
    ):
        """Transforms from poses.json (column-major) must be reshaped correctly."""
        import open3d as o3d

        mock_ppc.return_value = _make_mock_cloud()
        mock_rm.return_value = o3d.geometry.TriangleMesh.create_sphere()
        mock_sm.return_value = o3d.geometry.TriangleMesh.create_sphere()
        mock_gt.return_value = TexturedMesh(
            mesh=o3d.geometry.TriangleMesh.create_sphere(),
            texture_file="/tmp/t.png",
            material_file="/tmp/m.mtl",
        )
        mock_bfd.return_value = _make_mock_feature_db()

        pipeline = ReconstructionPipeline()
        pipeline.run(self.scan_dir, self.output_dir)

        # Identity matrix in column-major is still identity
        gt_args = mock_gt.call_args[0]
        images = gt_args[1]
        np.testing.assert_array_almost_equal(images[0]["pose"], np.eye(4))


# ---------------------------------------------------------------------------
# Tests for CLI
# ---------------------------------------------------------------------------


class TestCLI:
    """Test the CLI entry point."""

    def test_cli_requires_input(self):
        """CLI must require --input argument."""
        from processing_pipeline.cli import main

        with pytest.raises(SystemExit) as exc_info:
            with patch("sys.argv", ["cli", "--output", "/tmp/out"]):
                main()
        assert exc_info.value.code != 0

    def test_cli_requires_output(self):
        """CLI must require --output argument."""
        from processing_pipeline.cli import main

        with pytest.raises(SystemExit) as exc_info:
            with patch("sys.argv", ["cli", "--input", "/tmp/in"]):
                main()
        assert exc_info.value.code != 0

    @patch("processing_pipeline.pipeline.ReconstructionPipeline.run")
    def test_cli_calls_pipeline_run(self, mock_run, tmp_path):
        """CLI must instantiate pipeline and call run() with correct args."""
        from processing_pipeline.cli import main

        input_dir = str(tmp_path / "scan")
        output_dir = str(tmp_path / "out")

        with patch("sys.argv", ["cli", "--input", input_dir, "--output", output_dir]):
            main()

        mock_run.assert_called_once_with(input_dir, output_dir)
