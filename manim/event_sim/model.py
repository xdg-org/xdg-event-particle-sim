"""Deterministic particle paths and volume lookup for the demonstration."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TransportModel:
    grid_columns: int
    grid_rows: int
    volume_spacing: float
    volume_centers: tuple[np.ndarray, ...]
    particle_count: int
    shared_start: np.ndarray
    initial_positions: np.ndarray
    first_event_positions: np.ndarray
    post_event_positions: np.ndarray
    sampled_path_points: tuple[tuple[np.ndarray, ...], ...]
    collision_indices: tuple[int, ...]
    surface_indices: tuple[int, ...]
    split_step: int = 5
    final_advance_step: int = 6

    @property
    def volume_side_length(self):
        return self.volume_spacing

    @property
    def grid_width(self):
        return self.grid_columns * self.volume_spacing

    @property
    def grid_height(self):
        return self.grid_rows * self.volume_spacing

    def volume_index(self, position):
        column = int((position[0] + self.grid_width / 2) // self.volume_spacing)
        row_from_bottom = int(
            (position[1] + self.grid_height / 2) // self.volume_spacing
        )
        row_from_top = self.grid_rows - 1 - row_from_bottom
        return row_from_top * self.grid_columns + column

    def unique_volumes(self, step):
        seen = set()
        result = []
        for particle_index, points in enumerate(self.sampled_path_points):
            volume_index = self.volume_index(points[step])
            if volume_index in seen:
                continue
            seen.add(volume_index)
            result.append((particle_index, volume_index))
        return tuple(result)


def build_transport_model():
    columns = 8
    rows = 6
    spacing = 1.1
    centers = tuple(
        np.array((
            (column - (columns - 1) / 2) * spacing,
            ((rows - 1) / 2 - row) * spacing,
            0,
        ))
        for row in range(rows)
        for column in range(columns)
    )

    particle_count = 10
    shared_start = centers[2 * columns + 3] + np.array((0, 0, 0.65))
    initial_positions = np.tile(shared_start, (particle_count, 1))
    source_angles = np.deg2rad((15, 50, 90, 125, 160, 200, 235, 270, 305, 340))
    source_directions = np.array(tuple(
        (np.cos(angle), np.sin(angle), 0)
        for angle in source_angles
    ))
    surface_indices = (2, 7)
    collision_indices = tuple(
        index for index in range(particle_count)
        if index not in surface_indices
    )
    first_event_distances = np.array((
        0.30, 0.34, 0.55, 0.28, 0.34,
        0.31, 0.27, 0.55, 0.34, 0.30,
    ))
    first_event_positions = (
        initial_positions
        + first_event_distances[:, None] * source_directions
    )
    post_event_positions = first_event_positions.copy()

    grid_width = columns * spacing
    grid_height = rows * spacing
    movement_limits = np.array((
        grid_width / 2 - 0.12,
        grid_height / 2 - 0.12,
    ))
    rng = np.random.default_rng(917)
    sampled_positions = post_event_positions.copy()
    layers = [
        initial_positions.copy(),
        first_event_positions.copy(),
        post_event_positions.copy(),
    ]
    for advance_index in range(6):
        directions = rng.normal(size=(particle_count, 2))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        if advance_index == 0:
            directions[surface_indices[0], 1] = abs(
                directions[surface_indices[0], 1]
            )
            directions[surface_indices[1], 1] = -abs(
                directions[surface_indices[1], 1]
            )
            directions /= np.linalg.norm(directions, axis=1, keepdims=True)

        candidates = sampled_positions.copy()
        candidates[:, :2] += 0.75 * directions
        for axis, limit in enumerate(movement_limits):
            outside = np.abs(candidates[:, axis]) > limit
            candidates[outside, axis] = (
                sampled_positions[outside, axis]
                - 0.75 * directions[outside, axis]
            )
        sampled_positions = candidates
        layers.append(sampled_positions.copy())

    paths = tuple(
        tuple(layer[index].copy() for layer in layers)
        for index in range(particle_count)
    )
    return TransportModel(
        grid_columns=columns,
        grid_rows=rows,
        volume_spacing=spacing,
        volume_centers=centers,
        particle_count=particle_count,
        shared_start=shared_start,
        initial_positions=initial_positions,
        first_event_positions=first_event_positions,
        post_event_positions=post_event_positions,
        sampled_path_points=paths,
        collision_indices=collision_indices,
        surface_indices=surface_indices,
    )
