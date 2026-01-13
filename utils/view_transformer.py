"""View transformer for perspective transformation."""

import numpy as np
import cv2


class ViewTransformer:
    """
    Transforms points from source perspective to target perspective.

    This class uses OpenCV's perspective transformation to convert
    coordinates from the original video frame to a bird's-eye view
    or normalized coordinate system for accurate distance measurement.
    """

    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        """
        Initialize the view transformer.

        Args:
            source: Source polygon coordinates (4 points) in original frame
            target: Target polygon coordinates (4 points) in transformed space
        """
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """
        Transform points from source to target perspective.

        Args:
            points: Array of points to transform (N, 2) or (N,)

        Returns:
            Transformed points in target perspective
        """
        if points.size == 0:
            return points

        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)
