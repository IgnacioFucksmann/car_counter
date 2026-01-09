from typing import List, Optional
import numpy as np
from ultralytics import YOLO
from supervision.tools.detections import Detections


class VehicleDetector:
    """
    Class for loading and using YOLOv8 models for vehicle detection.

    This class handles only detection, not tracking or annotation.
    For complete video processing with tracking, use VideoProcessor.

    Args:
        model_name: Name of the YOLOv8 model (e.g., "yolov8x.pt", "yolov8n.pt")
        class_ids: List of class IDs to detect. Default: [2, 3, 5, 7]
                   (car, motorcycle, bus, truck)
        fuse: If True, fuses model layers for better performance
    """

    def __init__(
        self,
        model_name: str = "yolov8x.pt",
        class_ids: Optional[List[int]] = None,
        fuse: bool = True,
    ):
        self.model_name = model_name
        self.model = YOLO(model_name)

        if fuse:
            self.model.fuse()

        # Dictionary of class names
        self.class_names = self.model.model.names

        # Class IDs of interest (vehicles)
        self.class_ids = class_ids if class_ids is not None else [2, 3, 5, 7]

    def detect(self, frame: np.ndarray) -> Detections:
        """
        Detect vehicles in a single frame.

        Args:
            frame: The input frame to detect vehicles in (numpy array, BGR format)

        Returns:
            Detections: A Detections object containing the detected vehicles
                with bounding boxes, confidences, and class IDs
        """
        # Model prediction (verbose=False to silence prints)
        results = self.model(frame, verbose=False)

        # Convert to supervision Detections format
        detections = Detections(
            xyxy=results[0].boxes.xyxy.cpu().numpy(),
            confidence=results[0].boxes.conf.cpu().numpy(),
            class_id=results[0].boxes.cls.cpu().numpy().astype(int),
        )

        # Filter only classes of interest (vehicles)
        mask = np.array(
            [class_id in self.class_ids for class_id in detections.class_id], dtype=bool
        )
        detections.filter(mask=mask, inplace=True)

        return detections

    def get_class_name(self, class_id: int) -> str:
        """
        Gets the name of a class given its ID.

        Args:
            class_id: Class ID

        Returns:
            str: Class name (e.g., "car", "truck")
        """
        return self.class_names.get(class_id, f"Unknown_{class_id}")

    def get_class_names_dict(self) -> dict:
        """
        Returns the complete dictionary of class names.

        Returns:
            dict: Dictionary mapping class_id to class_name
                Example: {0: 'person', 2: 'car', 3: 'motorcycle', ...}
        """
        return self.class_names
