"""Unit tests for CLI entry point.

Validates: Requirements for CLI usage with OptimizedPipeline.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest


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

    @patch("processing_pipeline.optimized_pipeline.OptimizedPipeline.run")
    def test_cli_calls_pipeline_run(self, mock_run, tmp_path):
        """CLI must instantiate OptimizedPipeline and call run() with correct args."""
        from processing_pipeline.cli import main

        input_dir = str(tmp_path / "scan")
        output_dir = str(tmp_path / "out")

        with patch("sys.argv", ["cli", "--input", input_dir, "--output", output_dir]):
            main()

        mock_run.assert_called_once_with(input_dir, output_dir)
