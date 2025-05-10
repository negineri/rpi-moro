"""Frame processor implementations for camera streaming.

This module provides implementations of the FrameProcessor protocol
for processing video frames in the camera streaming system.
"""

import logging
from typing import Any, Optional

import cv2
import numpy as np

# ロガーの設定
logger = logging.getLogger(__name__)


class DefaultFrameProcessor:
    """Default implementation of the FrameProcessor protocol.

    This class handles frame processing logic including resizing, grayscale conversion,
    and motion detection.
    """

    def __init__(
        self,
        resolution: Optional[tuple[int, int]] = None,
        grayscale: bool = False,
        motion_detection: bool = False,
        motion_threshold: float = 0.005,
        motion_max_skip_frames: int = 30,
    ) -> None:
        """Initialize a DefaultFrameProcessor.

        Args:
            resolution: Optional tuple (width, height) to resize frames. Defaults to None.
            grayscale: Whether to convert frames to grayscale. Defaults to False.
            motion_detection: Whether to enable motion-based frame skipping. Defaults to False.
            motion_threshold: Threshold for motion detection (0.0-1.0). Defaults to 0.005.
            motion_max_skip_frames: Maximum number of consecutive frames to skip. Defaults to 30.
        """
        self.resolution = resolution
        self.grayscale = grayscale
        self.motion_detection = motion_detection
        self.motion_threshold = motion_threshold
        self.motion_max_skip_frames = motion_max_skip_frames

        # State for motion detection
        self._previous_frame: Optional[np.ndarray[Any, np.dtype[Any]]] = None
        self._skipped_frames_count: int = 0
        self._last_change_ratio: float = 0.0

    def process_frame(
        self, frame: np.ndarray[Any, np.dtype[Any]]
    ) -> Optional[np.ndarray[Any, np.dtype[Any]]]:
        """Process a frame according to the configured options.

        This method applies resolution changes, grayscale conversion, and motion detection.

        Args:
            frame: The source frame to process.

        Returns:
            The processed frame or None if the frame should be skipped.
        """
        processed_frame = frame.copy()

        # Resize the frame first if resolution is specified
        if self.resolution:
            processed_frame = cv2.resize(processed_frame, self.resolution)

        # Apply motion detection if enabled
        if self.motion_detection and self._previous_frame is not None:
            # Force sending a frame if we've skipped too many
            if self._skipped_frames_count >= self.motion_max_skip_frames:
                logger.debug(f"Forced frame send after {self._skipped_frames_count} skipped frames")
                self._skipped_frames_count = 0
            # Check for motion only if frame shapes match
            elif processed_frame.shape == self._previous_frame.shape:
                # Convert to grayscale for motion detection (reduces noise and color changes)
                gray_current = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
                gray_previous = cv2.cvtColor(self._previous_frame, cv2.COLOR_BGR2GRAY)

                # Apply Gaussian blur to reduce noise
                gray_current = cv2.GaussianBlur(gray_current, (5, 5), 0)
                gray_previous = cv2.GaussianBlur(gray_previous, (5, 5), 0)

                # Compute absolute difference between current and previous frame
                diff = cv2.absdiff(gray_current, gray_previous)

                # Apply threshold to highlight significant changes
                _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

                # Count non-zero pixels (changed pixels)
                non_zero = np.count_nonzero(thresh)
                total_pixels = thresh.shape[0] * thresh.shape[1]
                change_ratio = non_zero / total_pixels

                # Store for logging
                self._last_change_ratio = change_ratio

                # Skip frame if change is below threshold
                if change_ratio < self.motion_threshold:
                    self._skipped_frames_count += 1
                    # Log every 10 frames to avoid excessive logging
                    if self._skipped_frames_count % 10 == 1:
                        logger.debug(
                            f"Motion below threshold: {change_ratio:.6f} < {self.motion_threshold}, "  # noqa: E501
                            f"skipped {self._skipped_frames_count} frames"
                        )
                    return None
                if self._skipped_frames_count > 0:
                    logger.debug(
                        f"Motion detected: {change_ratio:.6f} > {self.motion_threshold}, "
                        f"sent frame after {self._skipped_frames_count} skipped frames"
                    )
                self._skipped_frames_count = 0

        # Apply grayscale conversion if requested (after motion detection)
        if self.grayscale:
            processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
            # Convert back to 3 channels for consistent processing
            processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2BGR)

        # Store frame for next comparison (always store original/color frame for motion detection)
        self._previous_frame = (
            frame.copy() if self.resolution is None else cv2.resize(frame, self.resolution)
        )

        return processed_frame

    @property
    def last_change_ratio(self) -> float:
        """Get the last calculated motion change ratio.

        Returns:
            The last motion change ratio between frames.
        """
        return self._last_change_ratio

    @property
    def skipped_frames_count(self) -> int:
        """Get the current count of consecutive skipped frames.

        Returns:
            Number of consecutive frames skipped due to low motion.
        """
        return self._skipped_frames_count
