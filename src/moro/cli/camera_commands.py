"""CLI commands for camera control and streaming."""

import logging
import sys
from typing import Optional, Tuple, Union

import click

from moro.modules.camera import Camera, CameraError, CameraStreamer

logger = logging.getLogger(__name__)


@click.group(name="camera")
def camera_group() -> None:
    """Camera control and streaming commands."""
    pass


@camera_group.command(name="stream")
@click.option("--device", "-d", default=0, help="Camera device ID or path (default: 0)")
@click.option("--width", "-w", default=640, help="Frame width in pixels (default: 640)")
@click.option("--height", "-h", default=480, help="Frame height in pixels (default: 480)")
@click.option("--fps", "-f", default=15, help="Target FPS for streaming (default: 15)")
@click.option("--quality", "-q", default=70, help="JPEG compression quality 0-100 (default: 70)")
@click.option(
    "--host",
    default="0.0.0.0",  # noqa: S104
    help="Host address to bind the server to (default: 0.0.0.0)",
)
@click.option("--port", "-p", default=5000, help="Port to run the server on (default: 5000)")
@click.option(
    "--stream-resolution",
    type=(int, int),
    default=None,
    help="Resize frames to this resolution (width, height). Example: --stream-resolution 320 240",
)
@click.option("--grayscale", is_flag=True, help="Convert frames to grayscale to reduce bandwidth")
@click.option(
    "--motion-detection",
    is_flag=True,
    help="Enable motion-based frame skipping to reduce bandwidth when scene is static",
)
@click.option(
    "--motion-threshold",
    type=float,
    default=0.005,
    help="Threshold for motion detection (0.0-1.0, higher means less sensitive, default: 0.005)",
)
@click.option(
    "--motion-max-skip",
    type=int,
    default=30,
    help="Maximum number of consecutive frames to skip in motion detection (default: 30)",
)
def stream_camera(
    device: str,
    width: int,
    height: int,
    fps: int,
    quality: int,
    host: str,
    port: int,
    stream_resolution: Optional[Tuple[int, int]],
    grayscale: bool,
    motion_detection: bool,
    motion_threshold: float,
    motion_max_skip: int,
) -> None:
    """Start camera streaming server.

    Streams USB camera feed to a web interface that can be accessed via browser.

    The command supports several bandwidth reduction options:

    - Reduce resolution via --stream-resolution
    - Use grayscale conversion with --grayscale
    - Skip frames when scene is static with --motion-detection
    """
    try:
        # Convert device to int if it's a number, otherwise keep as string (for paths)
        device_id: Union[int, str] = ""
        try:
            device_id = int(device)
        except ValueError:
            device_id = device

        logger.info(f"Initializing camera with device_id={device_id}")
        camera = Camera(device_id=device_id, width=width, height=height, fps=fps)

        logger.info(f"Starting camera streamer on {host}:{port}")
        click.echo(f"Starting camera streaming server at http://{host}:{port}")
        click.echo("Press Ctrl+C to stop the server")

        # Log bandwidth reduction settings
        if stream_resolution:
            click.echo(f"Streaming resolution: {stream_resolution[0]}x{stream_resolution[1]}")
        if grayscale:
            click.echo("Grayscale mode enabled")
        if motion_detection:
            click.echo(
                f"Motion detection enabled (threshold: {motion_threshold}, "
                f"max consecutive skips: {motion_max_skip})"
            )

        streamer = CameraStreamer(
            camera=camera,
            fps=fps,
            quality=quality,
            host=host,
            port=port,
            resolution=stream_resolution,
            grayscale=grayscale,
            motion_detection=motion_detection,
            motion_threshold=motion_threshold,
            motion_max_skip_frames=motion_max_skip,
        )

        try:
            streamer.start()
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received, stopping server")
            click.echo("Stopping camera streaming server...")
            streamer.stop()
            camera.stop()
            click.echo("Server stopped")

    except CameraError as e:
        logger.error(f"Camera error: {e!s}")
        click.echo(f"Error: {e!s}", err=True)
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Unexpected error: {e!s}")
        click.echo(f"Unexpected error: {e!s}", err=True)
        sys.exit(1)


@camera_group.command(name="info")
@click.option("--device", "-d", default=0, help="Camera device ID or path (default: 0)")
def camera_info(device: str) -> None:
    """Show information about connected camera."""
    try:
        # Convert device to int if it's a number, otherwise keep as string (for paths)
        device_id: Union[int, str] = ""
        try:
            device_id = int(device)
        except ValueError:
            device_id = device

        logger.info(f"Checking camera info for device_id={device_id}")
        camera = Camera(device_id=device_id)

        try:
            camera.start()
            info = camera.get_camera_info()

            click.echo("===== Camera Information =====")
            for key, value in info.items():
                click.echo(f"{key}: {value}")

        finally:
            camera.stop()

    except CameraError as e:
        logger.error(f"Camera error: {e!s}")
        click.echo(f"Error: {e!s}", err=True)
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Unexpected error: {e!s}")
        click.echo(f"Unexpected error: {e!s}", err=True)
        sys.exit(1)
