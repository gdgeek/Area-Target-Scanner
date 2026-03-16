"""Command-line interface for the Area Target reconstruction pipeline."""

from __future__ import annotations

import argparse
import logging
import sys


def main() -> None:
    """Parse arguments and run the reconstruction pipeline."""
    parser = argparse.ArgumentParser(
        description="Area Target Scanner - Post-Processing Pipeline",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the scan data directory (contains pointcloud.ply, poses.json, images/).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to the output asset bundle directory.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    from processing_pipeline.pipeline import ReconstructionPipeline

    pipeline = ReconstructionPipeline()
    pipeline.run(args.input, args.output)


if __name__ == "__main__":
    main()
