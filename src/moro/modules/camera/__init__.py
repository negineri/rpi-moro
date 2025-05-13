"""Camera module for USB camera streaming functionality.

This module provides functionality for accessing and streaming USB cameras
connected to a Raspberry Pi.
"""

from moro.modules.camera.camera import Camera, CameraError
from moro.modules.camera.streamer import CameraStreamer

__all__ = ["Camera", "CameraError", "CameraStreamer"]
