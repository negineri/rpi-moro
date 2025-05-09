"""Camera module for handling USB camera operations.

This module provides the Camera class for interacting with USB cameras
connected to a Raspberry Pi.
"""

import logging
from typing import Any, Optional, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class CameraError(Exception):
    """Exception raised for camera-related errors."""

    pass


class Camera:
    """Camera class for USB camera operations.

    This class handles the connection to a USB camera and provides methods
    for retrieving frames and controlling camera settings.

    Attributes:
        device_id: The camera device identifier (integer or path).
        width: The width of the captured frames.
        height: The height of the captured frames.
        is_running: Boolean indicating if the camera is currently active.
    """

    def __init__(
        self,
        device_id: Union[int, str] = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
    ) -> None:
        """Initialize a Camera instance.

        Args:
            device_id: Camera device identifier (integer or path string).
                       Defaults to 0 (first available camera).
            width: Frame width in pixels. Defaults to 640.
            height: Frame height in pixels. Defaults to 480.
            fps: Frames per second. Defaults to 30.

        Raises:
            CameraError: If camera cannot be initialized.
        """
        self.device_id = device_id
        self.width = width
        self.height = height
        self.fps = fps
        self._camera: Optional[cv2.VideoCapture] = None
        self.is_running = False

        logger.debug(
            f"Initializing camera with device_id={device_id}, "
            f"resolution={width}x{height}, fps={fps}"
        )

    def start(self) -> None:
        """Start the camera.

        Opens the camera device and prepares it for capturing frames.

        Raises:
            CameraError: If camera cannot be opened or configured.
        """
        if self.is_running:
            logger.warning("Camera is already running")
            return

        logger.info(f"Starting camera with device_id={self.device_id}")

        try:
            self._camera = cv2.VideoCapture(self.device_id)

            if not self._camera.isOpened():
                raise CameraError(f"Failed to open camera with device_id={self.device_id}")

            # Set camera properties
            self._camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self._camera.set(cv2.CAP_PROP_FPS, self.fps)

            # Check if settings were applied correctly
            actual_width = self._camera.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_height = self._camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
            actual_fps = self._camera.get(cv2.CAP_PROP_FPS)

            logger.debug(
                f"Camera started with actual settings: "
                f"resolution={actual_width}x{actual_height}, fps={actual_fps}"
            )

            self.is_running = True

        except Exception as e:
            if self._camera is not None:
                self._camera.release()
            self._camera = None
            raise CameraError(f"Error starting camera: {e!s}") from e

    def stop(self) -> None:
        """Stop the camera.

        Releases the camera resources.
        """
        if not self.is_running or self._camera is None:
            logger.warning("Camera is not running")
            return

        logger.info("Stopping camera")
        self._camera.release()
        self._camera = None
        self.is_running = False

    def read_frame(self) -> Optional[np.ndarray[Any, Any]]:
        """Read a single frame from the camera.

        Returns:
            np.ndarray: Frame as a numpy array if successful, None otherwise.

        Raises:
            CameraError: If camera is not running.
        """
        if not self.is_running or self._camera is None:
            raise CameraError("Camera is not running")

        ret, frame = self._camera.read()
        if not ret:
            logger.warning("Failed to read frame from camera")
            return None

        return frame

    def get_camera_info(self) -> dict[str, Union[int, float, str]]:
        """Get information about the camera.

        Returns:
            dict containing camera properties.

        Raises:
            CameraError: If camera is not running.
        """
        if not self.is_running or self._camera is None:
            raise CameraError("Camera is not running")

        info: dict[str, Any] = {
            "width": self._camera.get(cv2.CAP_PROP_FRAME_WIDTH),
            "height": self._camera.get(cv2.CAP_PROP_FRAME_HEIGHT),
            "fps": self._camera.get(cv2.CAP_PROP_FPS),
            "device_id": self.device_id,
        }

        return info

    def __enter__(self) -> "Camera":
        """Enter context manager."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # type: ignore
        """Exit context manager."""
        self.stop()
