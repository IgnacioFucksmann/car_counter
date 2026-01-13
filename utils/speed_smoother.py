"""Adaptive speed smoothing filter for vehicle tracking."""

from typing import Dict, Optional
import numpy as np


class AdaptiveSpeedSmoother:
    """
    Adaptive exponential moving average filter for speed smoothing.

    This filter uses an adaptive alpha factor that:
    - Uses low alpha (more smoothing) for small speed changes
    - Uses high alpha (faster response) for large speed changes

    The alpha factor is calculated exponentially based on the speed difference.
    """

    def __init__(
        self,
        min_alpha: float = 0.1,
        max_alpha: float = 0.9,
        threshold: float = 5.0,
        sensitivity: float = 2.0,
    ):
        """
        Initialize the adaptive speed smoother.

        Args:
            min_alpha: Minimum smoothing factor (for small changes)
            max_alpha: Maximum smoothing factor (for large changes)
            threshold: Speed difference threshold in km/h (below this, use min_alpha)
            sensitivity: Exponential sensitivity factor (higher = steeper curve)
        """
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        self.threshold = threshold
        self.sensitivity = sensitivity
        self.smoothed_speeds: Dict[int, float] = {}

    def smooth(self, tracker_id: int, current_speed: float) -> float:
        """
        Apply adaptive smoothing to the current speed measurement.

        Args:
            tracker_id: Unique tracker ID for the vehicle
            current_speed: Current raw speed measurement in km/h

        Returns:
            Smoothed speed value in km/h
        """
        # Get previous smoothed speed (or use current if first time)
        previous_speed = self.smoothed_speeds.get(tracker_id, current_speed)

        # Calculate speed difference
        speed_diff = abs(current_speed - previous_speed)

        # Calculate adaptive alpha factor using exponential function
        if speed_diff <= self.threshold:
            # Small change: use minimum alpha (more smoothing)
            alpha = self.min_alpha
        else:
            # Large change: calculate alpha based on exponential curve
            # Normalize speed difference relative to threshold
            normalized_diff = (speed_diff - self.threshold) / self.threshold

            # Exponential function: alpha increases with speed difference
            # Using sigmoid-like curve for smooth transition
            exp_factor = 1 - np.exp(-self.sensitivity * normalized_diff)
            alpha = self.min_alpha + (self.max_alpha - self.min_alpha) * exp_factor

        # Apply exponential moving average
        smoothed_speed = alpha * current_speed + (1 - alpha) * previous_speed

        # Store smoothed speed for next iteration
        self.smoothed_speeds[tracker_id] = smoothed_speed

        return smoothed_speed

    def reset(self, tracker_id: Optional[int] = None):
        """
        Reset smoothed speeds for a specific tracker or all trackers.

        Args:
            tracker_id: If provided, reset only this tracker. Otherwise reset all.
        """
        if tracker_id is not None:
            self.smoothed_speeds.pop(tracker_id, None)
        else:
            self.smoothed_speeds.clear()
