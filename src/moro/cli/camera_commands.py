"""CLI commands for camera control and streaming."""

import logging
import sys
from typing import Union

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
def stream_camera(
    device: str,
    width: int,
    height: int,
    fps: int,
    quality: int,
    host: str,
    port: int,
) -> None:
    """Start camera streaming server.

    Streams USB camera feed to a web interface that can be accessed via browser.
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

        streamer = CameraStreamer(camera=camera, fps=fps, quality=quality, host=host, port=port)

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
