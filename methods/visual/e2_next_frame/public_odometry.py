"""Public RGB/action visual odometry for the axis-aligned WCM-Grid camera.

The formal camera is a 9x9 world-aligned crop with a fixed marker at its
centre.  A successful ``forward`` therefore appears as an exact one-tile
translation of the surrounding public scene.  This module recovers those
translations solely by registering adjacent *public* RGB frames.  It never
reads a simulator pose, world coordinates, target RGB, or evaluator records.

The resulting transform is used by the WCM visual compiler to reproject a
Visual Binding Store frame into the query's predicted next-camera view.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from e0.public_protocol import PublicBenchmark, PublicStream


TILE_PIXELS = 32
_GRID_SIDE = 9
_REGISTRATION_TILE_PIXELS = 8
_REGISTRATION_SIDE = _GRID_SIDE * _REGISTRATION_TILE_PIXELS
_CANDIDATE_SHIFTS = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))


class PublicOdometryError(ValueError):
    """A public trajectory cannot support a causal binding projection."""


@dataclass(frozen=True, slots=True)
class BindingProjection:
    """Binding-frame to next-view scene translation, expressed in RGB pixels."""

    delta_y: int
    delta_x: int
    available: bool


@dataclass(frozen=True, slots=True)
class _StreamOdometry:
    scene_offsets: tuple[tuple[int, int], ...]
    headings: tuple[tuple[int, int] | None, ...]


def _small_public_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(
            image.convert("RGB").resize((_REGISTRATION_SIDE, _REGISTRATION_SIDE), Image.Resampling.NEAREST),
            dtype=np.float32,
        ).copy()


def _alignment_error(*, before: np.ndarray, after: np.ndarray, delta_y: int, delta_x: int) -> float:
    """Error after translating ``before`` into ``after`` by grid-cell units."""

    if before.shape != after.shape or before.shape != (_REGISTRATION_SIDE, _REGISTRATION_SIDE, 3):
        raise PublicOdometryError("public RGB registration inputs are malformed")
    pixel_y, pixel_x = delta_y * _REGISTRATION_TILE_PIXELS, delta_x * _REGISTRATION_TILE_PIXELS
    height, width = before.shape[:2]
    y0, y1 = max(0, pixel_y), min(height, height + pixel_y)
    x0, x1 = max(0, pixel_x), min(width, width + pixel_x)
    source = before[y0 - pixel_y:y1 - pixel_y, x0 - pixel_x:x1 - pixel_x]
    target = after[y0:y1, x0:x1]


    valid = np.ones((y1 - y0, x1 - x0), dtype=bool)
    marker_y0, marker_y1 = max(0, 4 * _REGISTRATION_TILE_PIXELS - y0), min(y1 - y0, 5 * _REGISTRATION_TILE_PIXELS - y0)
    marker_x0, marker_x1 = max(0, 4 * _REGISTRATION_TILE_PIXELS - x0), min(x1 - x0, 5 * _REGISTRATION_TILE_PIXELS - x0)
    valid[marker_y0:marker_y1, marker_x0:marker_x1] = False
    if not bool(valid.any()):
        raise PublicOdometryError("registration overlap contains only the agent marker")
    return float(np.abs(source - target)[valid].mean())


def _forward_scene_delta(*, before: np.ndarray, after: np.ndarray) -> tuple[int, int]:
    """Recover the public-scene shift induced by one recorded forward action."""

    ranked = sorted(
        (
            _alignment_error(before=before, after=after, delta_y=delta_y, delta_x=delta_x),
            (delta_y, delta_x),
        )
        for delta_y, delta_x in _CANDIDATE_SHIFTS
    )


    return ranked[0][1]


def _turn_left(heading: tuple[int, int]) -> tuple[int, int]:
    dy, dx = heading
    return -dx, dy


def _turn_right(heading: tuple[int, int]) -> tuple[int, int]:
    dy, dx = heading
    return dx, -dy


def _build_stream_odometry(*, stream: PublicStream, public_root: Path) -> _StreamOdometry:
    if not stream.frames or stream.frames[0].frame_index != 0 or stream.frames[0].action is not None:
        raise PublicOdometryError("public stream does not begin with a valid initial frame")


    image_cache: dict[int, np.ndarray] = {}

    def image_at(index: int) -> np.ndarray:
        if index not in image_cache:
            image_cache[index] = _small_public_rgb(public_root / stream.frames[index].rgb_path)
        return image_cache[index]
    offsets: list[tuple[int, int]] = [(0, 0)]
    headings: list[tuple[int, int] | None] = [None]
    heading: tuple[int, int] | None = None
    for index, frame in enumerate(stream.frames[1:], start=1):
        if frame.frame_index != index:
            raise PublicOdometryError("public stream frame indices must be contiguous")
        action = frame.action
        if action is None:
            raise PublicOdometryError("non-initial public frame is missing its action")
        delta_y = delta_x = 0
        if action == "forward":
            grid_dy, grid_dx = _forward_scene_delta(before=image_at(index - 1), after=image_at(index))
            delta_y, delta_x = grid_dy * TILE_PIXELS, grid_dx * TILE_PIXELS
            if grid_dy or grid_dx:

                heading = (-grid_dy, -grid_dx)
        elif action == "left" and heading is not None:
            heading = _turn_left(heading)
        elif action == "right" and heading is not None:
            heading = _turn_right(heading)
        previous_y, previous_x = offsets[-1]
        offsets.append((previous_y + delta_y, previous_x + delta_x))
        headings.append(heading)
    return _StreamOdometry(tuple(offsets), tuple(headings))


class PublicVisualOdometry:
    """Lazily caches causal RGB/action camera transforms per public stream."""

    def __init__(self, *, benchmark: PublicBenchmark, dataset_root: str | Path) -> None:
        self._benchmark = benchmark
        self._public_root = Path(dataset_root).resolve() / "public"
        self._cache: dict[str, _StreamOdometry] = {}

    def _stream(self, stream_id: str) -> _StreamOdometry:
        if stream_id not in self._cache:
            stream = self._benchmark.streams.get(stream_id)
            if stream is None:
                raise PublicOdometryError(f"unknown public stream: {stream_id}")
            self._cache[stream_id] = _build_stream_odometry(stream=stream, public_root=self._public_root)
        return self._cache[stream_id]

    def binding_to_next_view(
        self, *, stream_id: str, binding_after_frame_index: int, prefix_end_frame: int, next_action: str
    ) -> BindingProjection:
        """Causally project a binding frame into the immediate next camera view."""

        if next_action != "forward":
            return BindingProjection(0, 0, False)
        trace = self._stream(stream_id)
        if not 0 <= binding_after_frame_index <= prefix_end_frame < len(trace.scene_offsets):
            raise PublicOdometryError("binding/query frame indices are outside the public stream")
        heading = trace.headings[prefix_end_frame]
        if heading is None:
            return BindingProjection(0, 0, False)
        binding_y, binding_x = trace.scene_offsets[binding_after_frame_index]
        current_y, current_x = trace.scene_offsets[prefix_end_frame]


        next_scene_y, next_scene_x = current_y - heading[0] * TILE_PIXELS, current_x - heading[1] * TILE_PIXELS
        return BindingProjection(next_scene_y - binding_y, next_scene_x - binding_x, True)


def translate_public_binding(image: torch.Tensor, *, delta_y: int, delta_x: int) -> torch.Tensor:
    """Translate a public binding frame without wrapped pixels or interpolation."""

    if image.ndim != 3 or image.shape[0] != 3 or image.shape[1] != image.shape[2]:
        raise PublicOdometryError("public binding projection expects square RGB tensors")
    side = image.shape[1]
    scaled_y = int(round(delta_y * side / 288))
    scaled_x = int(round(delta_x * side / 288))
    result = torch.zeros_like(image)
    destination_y0, destination_y1 = max(0, scaled_y), min(side, side + scaled_y)
    destination_x0, destination_x1 = max(0, scaled_x), min(side, side + scaled_x)
    if destination_y0 >= destination_y1 or destination_x0 >= destination_x1:
        return result
    source_y0, source_y1 = destination_y0 - scaled_y, destination_y1 - scaled_y
    source_x0, source_x1 = destination_x0 - scaled_x, destination_x1 - scaled_x
    result[:, destination_y0:destination_y1, destination_x0:destination_x1] = image[:, source_y0:source_y1, source_x0:source_x1]
    return result
