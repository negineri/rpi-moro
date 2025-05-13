"""Protocol definitions for camera streaming components.

This module provides protocol definitions for frame processing
and emission interfaces used in the camera streaming system.
"""

from typing import Any, Optional, Protocol

import numpy as np


class FrameProcessor(Protocol):
    """Protocol defining the interface for frame processing.

    This interface allows for dependency injection and easier testing of frame
    processing logic independent of the streaming system.
    """

    def process_frame(
        self, frame: np.ndarray[Any, np.dtype[Any]]
    ) -> Optional[np.ndarray[Any, np.dtype[Any]]]:
        """Process a single video frame.

        Args:
            frame: The source frame to process.

        Returns:
            The processed frame or None if the frame should be skipped.
        """
        ...

    @property
    def last_change_ratio(self) -> float:
        """Get the last calculated motion change ratio."""
        ...

    @property
    def skipped_frames_count(self) -> int:
        """Get the current count of consecutive skipped frames."""
        ...


class FrameEmitter(Protocol):
    """Protocol defining the interface for frame emission.

    This interface allows for dependency injection and easier testing of
    frame emission logic independent of the streaming system.
    """

    def emit_frame(self, frame: np.ndarray[Any, np.dtype[Any]]) -> None:
        """Emit a processed frame to clients.

        Args:
            frame: The processed frame to emit.
        """
        ...

    def emit_camera_info(self, camera_info: dict[str, Any]) -> None:
        """Emit camera information to clients.

        Args:
            camera_info: Dictionary containing camera information.
        """
        ...

    @property
    def cached_frame(self) -> Optional[str]:
        """Get the currently cached encoded frame."""
        ...

    @property
    def camera_info(self) -> Optional[dict[str, Any]]:
        """Get the currently cached camera information."""
        ...
