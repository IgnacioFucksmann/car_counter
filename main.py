#!/usr/bin/env python3
"""Main script for vehicle counting and tracking."""

from pathlib import Path
from supervision.geometry.dataclasses import Point
from model import VehicleDetector
from processor import VideoProcessor


def main():
    """Process video with vehicle detection and counting."""
    # Configuration
    input_video = "data/vehicle-counting.mp4"
    output_video = "output/vehicle-counting-result.mp4"
    model_name = "yolov8x.pt"
    line_start = Point(50, 1500)
    line_end = Point(3840 - 50, 1500)

    # Create output directory
    Path(output_video).parent.mkdir(parents=True, exist_ok=True)

    # Create detector and processor
    detector = VehicleDetector(model_name=model_name)
    processor = VideoProcessor(
        detector=detector, line_start=line_start, line_end=line_end
    )

    # Process and save video
    processor.process_and_save(
        input_video_path=input_video,
        output_video_path=output_video,
        annotate=True,
    )

    # Print results
    print(f"\nVehicles counted (in):  {processor.line_counter.in_count}")
    print(f"Vehicles counted (out): {processor.line_counter.out_count}")
    print(
        f"Total:                  {processor.line_counter.in_count + processor.line_counter.out_count}"
    )


if __name__ == "__main__":
    main()
