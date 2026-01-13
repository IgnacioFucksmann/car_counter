from typing import Iterator, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict, deque
import numpy as np
from supervision.tools.detections import Detections, BoxAnnotator
from supervision.tools.line_counter import LineCounter, LineCounterAnnotator
from supervision.draw.color import ColorPalette
from supervision.geometry.dataclasses import Point
from supervision.video.source import get_video_frames_generator
from supervision.video.sink import VideoSink
from supervision.video.dataclasses import VideoInfo
from tqdm import tqdm
from yolox.tracker.byte_tracker import BYTETracker
from model import VehicleDetector
from utils.tracking_utils import detections2boxes, match_detections_with_tracks
from utils.view_transformer import ViewTransformer
from utils.speed_smoother import AdaptiveSpeedSmoother


@dataclass(frozen=True)
class BYTETrackerArgs:
    """Configuration arguments for ByteTrack."""

    track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.8
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1.0
    mot20: bool = False


class VideoProcessor:
    """
    Processes video with detection, tracking, counting, and annotation.

    This class orchestrates VehicleDetector, ByteTrack, LineCounter,
    and visualization to process videos frame by frame.
    """

    def __init__(
        self,
        detector: VehicleDetector,
        line_start: Optional[Point] = None,
        line_end: Optional[Point] = None,
        source_roi: Optional[np.ndarray] = None,
        target_roi: Optional[np.ndarray] = None,
        enable_speed_estimation: bool = True,
    ):
        """
        Initialize video processor.

        Args:
            detector: VehicleDetector instance for vehicle detection
            line_start: Start point of counting line (default: Point(50, 1500))
            line_end: End point of counting line (default: Point(3840-50, 1500))
            source_roi: Source polygon for perspective transformation (4 points)
            target_roi: Target polygon for perspective transformation (4 points)
            enable_speed_estimation: If True, enables speed estimation
        """
        self.detector = detector
        self.byte_tracker = BYTETracker(BYTETrackerArgs())
        self.box_annotator = BoxAnnotator(
            color=ColorPalette(), thickness=4, text_thickness=4, text_scale=2
        )

        # Line counter for vehicle counting
        if line_start is None:
            line_start = Point(50, 1500)
        if line_end is None:
            line_end = Point(3840 - 50, 1500)

        self.line_counter = LineCounter(start=line_start, end=line_end)
        self.line_annotator = LineCounterAnnotator(
            thickness=4, text_thickness=4, text_scale=2
        )

        # Speed estimation setup
        self.enable_speed_estimation = enable_speed_estimation
        if enable_speed_estimation:
            if source_roi is None:
                # Default ROI from the notebook (same video)
                source_roi = np.array(
                    [[1252, 787], [2298, 803], [5039, 2159], [-550, 2159]]
                )
            if target_roi is None:
                # Default target ROI (25x250)
                target_width = 25
                target_height = 250
                target_roi = np.array(
                    [
                        [0, 0],
                        [target_width - 1, 0],
                        [target_width - 1, target_height - 1],
                        [0, target_height - 1],
                    ]
                )

            self.view_transformer = ViewTransformer(
                source=source_roi, target=target_roi
            )
            # Initialize adaptive speed smoother
            self.speed_smoother = AdaptiveSpeedSmoother(
                min_alpha=0.1,  # Low alpha for small changes (more smoothing)
                max_alpha=0.7,  # Higher alpha for large changes (faster response)
                threshold=3.0,  # Threshold in km/h for considering change as "small"
                sensitivity=2.0,  # Exponential sensitivity factor
            )

    def process_video(
        self, video_path: str, annotate: bool = True
    ) -> Iterator[Tuple[np.ndarray, Detections]]:
        """
        Process video frame by frame with detection and tracking.

        Uses get_video_frames_generator to efficiently read frames
        without loading entire video into memory.

        Args:
            video_path: Path to the video file
            annotate: If True, returns annotated frames

        Yields:
            tuple: (frame, detections) for each frame
                - frame: annotated or original frame (numpy array)
                - detections: Detections object with tracking IDs
        """
        # Create frame generator - this efficiently reads frames one by one
        generator = get_video_frames_generator(video_path)

        # Get video info (for progress bar)
        video_info = VideoInfo.from_video_path(video_path)

        # Initialize coordinates storage for speed estimation if enabled
        if self.enable_speed_estimation:
            coordinates = defaultdict(lambda: deque(maxlen=video_info.fps))
            # Reset speed smoother for new video processing
            self.speed_smoother.reset()

        # Process each frame
        for frame in tqdm(generator, total=video_info.total_frames):
            # Detect vehicles in the frame
            detections = self.detector.detect(frame)

            # Update tracker with detections
            tracks = self.byte_tracker.update(
                output_results=detections2boxes(detections=detections),
                img_info=frame.shape,
                img_size=frame.shape,
            )

            # Match detections with tracks
            tracker_ids = match_detections_with_tracks(
                detections=detections, tracks=tracks
            )
            detections.tracker_id = np.array(tracker_ids)

            # Filter out detections without trackers
            mask = np.array(
                [tracker_id is not None for tracker_id in detections.tracker_id],
                dtype=bool,
            )
            detections.filter(mask=mask, inplace=True)

            # Update line counter (counts vehicles crossing the line)
            self.line_counter.update(detections=detections)

            # Speed estimation
            if self.enable_speed_estimation and len(detections) > 0:
                # Get bottom center points of detections
                points = np.array(
                    [
                        [(xyxy[0] + xyxy[2]) / 2, xyxy[3]]  # [x_center, y_bottom]
                        for xyxy in detections.xyxy
                    ]
                )

                # Transform points to target ROI perspective
                transformed_points = self.view_transformer.transform_points(
                    points
                ).astype(int)

                # Store Y coordinates (vertical position in transformed space)
                for tracker_id, [_, y] in zip(
                    detections.tracker_id, transformed_points
                ):
                    coordinates[tracker_id].append(y)

            # Annotate if requested
            if annotate:
                # Format labels with tracker ID and speed
                labels = []
                for _, confidence, class_id, tracker_id in detections:
                    if self.enable_speed_estimation and tracker_id in coordinates:
                        if len(coordinates[tracker_id]) < video_info.fps / 2:
                            # Not enough data yet, show only ID
                            labels.append(
                                f"#{tracker_id} {self.detector.get_class_name(class_id)}"
                            )
                        else:
                            # Calculate raw speed
                            coordinate_start = coordinates[tracker_id][
                                -1
                            ]  # Most recent
                            coordinate_end = coordinates[tracker_id][0]  # Oldest
                            distance = abs(coordinate_start - coordinate_end)
                            time = len(coordinates[tracker_id]) / video_info.fps
                            raw_speed = distance / time * 3.6  # Convert to km/h

                            # Apply adaptive smoothing
                            smoothed_speed = self.speed_smoother.smooth(
                                tracker_id=tracker_id, current_speed=raw_speed
                            )

                            labels.append(
                                f"#{tracker_id} {self.detector.get_class_name(class_id)} {int(smoothed_speed)} km/h"
                            )
                    else:
                        # No speed estimation or no coordinates
                        labels.append(
                            f"#{tracker_id} {self.detector.get_class_name(class_id)} {confidence:0.2f}"
                        )

                frame = self.box_annotator.annotate(
                    frame=frame.copy(), detections=detections, labels=labels
                )
                # Annotate counting line
                self.line_annotator.annotate(
                    frame=frame, line_counter=self.line_counter
                )

            yield frame, detections

    def process_and_save(
        self,
        input_video_path: str,
        output_video_path: str,
        annotate: bool = True,
    ) -> None:
        """
        Process video and save the result to a file.

        Args:
            input_video_path: Path to input video file
            output_video_path: Path where processed video will be saved
            annotate: If True, annotates frames with bounding boxes and labels
        """
        # Get video info
        video_info = VideoInfo.from_video_path(input_video_path)

        # Process video and save frames
        with VideoSink(output_video_path, video_info) as sink:
            for frame, detections in self.process_video(
                input_video_path, annotate=annotate
            ):
                sink.write_frame(frame)
