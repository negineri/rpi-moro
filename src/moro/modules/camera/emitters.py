"""Frame emitter implementations for camera streaming.

This module provides implementations of the FrameEmitter protocol
for emitting processed video frames to clients in the camera streaming system.
"""

import base64
from typing import Any, Optional

import cv2
import numpy as np
from flask_socketio import SocketIO


class SocketIOFrameEmitter:
    """SocketIO implementation of the FrameEmitter protocol.

    This class handles encoding and emitting frames via SocketIO.
    """

    def __init__(self, socketio: SocketIO, quality: int = 70) -> None:
        """Initialize a SocketIOFrameEmitter.

        Args:
            socketio: SocketIO instance to use for emitting frames.
            quality: JPEG compression quality (0-100). Defaults to 70.
        """
        self.socketio = socketio
        self.quality = quality
        self._cached_frame: Optional[str] = None
        self._camera_info: Optional[dict[str, Any]] = None

    def emit_frame(self, frame: np.ndarray[Any, np.dtype[Any]]) -> None:
        """Encode and emit a frame to connected clients.

        Args:
            frame: The processed frame to emit.
        """
        # Encode frame to JPEG
        _, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.quality])
        # Convert to base64 for sending over WebSocket
        b64_frame = base64.b64encode(buffer).decode("utf-8")
        # Store the latest frame for new clients
        self._cached_frame = b64_frame
        self.socketio.emit("frame", {"frame": b64_frame})

    def emit_camera_info(self, camera_info: dict[str, Any]) -> None:
        """Emit camera information to clients.

        Args:
            camera_info: Dictionary containing camera information.
        """
        self._camera_info = camera_info
        self.socketio.emit("camera_info", camera_info)

    @property
    def cached_frame(self) -> Optional[str]:
        """Get the currently cached encoded frame.

        Returns:
            Base64 encoded JPEG frame, or None if no frame has been cached.
        """
        return self._cached_frame

    @property
    def camera_info(self) -> Optional[dict[str, Any]]:
        """Get the currently cached camera information.

        Returns:
            Camera information dictionary, or None if not yet set.
        """
        return self._camera_info
