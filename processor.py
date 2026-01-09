from typing import Iterator, Tuple, Optional
from dataclasses import dataclass
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
    ):
        """
        Initialize video processor.

        Args:
            detector: VehicleDetector instance for vehicle detection
            line_start: Start point of counting line (default: Point(50, 1500))
            line_end: End point of counting line (default: Point(3840-50, 1500))
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

            # Annotate if requested
            if annotate:
                # Format labels with tracker ID
                labels = [
                    f"#{tracker_id} {self.detector.get_class_name(class_id)} {confidence:0.2f}"
                    for _, confidence, class_id, tracker_id in detections
                ]
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
