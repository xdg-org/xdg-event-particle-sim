"""Event based pseudo particle simulation animation, reusable in a slide or standalone scene."""

from manimlib import *
import numpy as np


def event_sim_scene(scene):
    """Play the event simulation demonstration on an existing Scene or Slide."""
    scene.camera.frame.reorient(0, 0)
    scene.camera.frame.set_field_of_view(15 * DEGREES)
    scene.camera.light_source.move_to(4 * LEFT + 3 * UP + 6 * OUT)
    scene.camera.frame.save_state()

    grid_columns = 8
    grid_rows = 6
    volume_spacing = 1.1
    volume_side_length = volume_spacing
    grid_width = grid_columns * volume_spacing
    grid_height = grid_rows * volume_spacing
    volume_centers = tuple(
        np.array((
            (column - (grid_columns - 1) / 2) * volume_spacing,
            ((grid_rows - 1) / 2 - row) * volume_spacing,
            0,
        ))
        for row in range(grid_rows)
        for column in range(grid_columns)
    )
    volume_faces = VGroup(
        *(
            Square(
                side_length=volume_side_length,
                fill_color=BLUE_D,
                fill_opacity=0.025,
                stroke_width=0,
            ).move_to(center)
            for center in volume_centers
        )
    ).deactivate_depth_test()
    volume_edges = VGroup(
        *(
            Square(
                side_length=volume_side_length,
                fill_opacity=0,
                stroke_color=BLUE_B,
                stroke_width=1.1,
            ).move_to(center)
            for center in volume_centers
        )
    ).deactivate_depth_test()

    particle_count = 6
    shared_start = volume_centers[2 * grid_columns + 3] + 0.65 * OUT
    initial_positions = np.tile(shared_start, (particle_count, 1))
    particle_colors = (
        RED_A,
        RED_B,
        RED_C,
        RED_D,
        RED_E,
        MAROON_A,
    )
    particles = Group(
        *(
            Sphere(
                radius=0.07,
                resolution=(51, 26),
                color=color,
                shading=(0.25, 0.65, 0.2),
            ).move_to(position)
            for position, color in zip(initial_positions, particle_colors)
        )
    )
    scene.add(
        volume_faces,
        particles,
        volume_edges,
    )
    scene.play(
        FadeIn(volume_faces),
        LaggedStart(
            *(ShowCreation(edges) for edges in volume_edges),
            lag_ratio=0.01,
        ),
        run_time=1.5,
    )

    def get_volume_index(position):
        column = int((position[0] + grid_width / 2) // volume_spacing)
        row_from_bottom = int((position[1] + grid_height / 2) // volume_spacing)
        row_from_top = grid_rows - 1 - row_from_bottom
        return row_from_top * grid_columns + column

    rng = np.random.default_rng(261)
    sampled_positions = initial_positions.copy()
    movement_limits = np.array((
        grid_width / 2 - 0.12,
        grid_height / 2 - 0.12,
    ))
    step_size = 0.25
    sampled_path_points = [
        [position.copy()]
        for position in initial_positions
    ]
    for _ in range(20):
        directions = rng.normal(size=(len(particles), 2))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        candidate_positions = sampled_positions.copy()
        candidate_positions[:, :2] += step_size * directions
        for axis, limit in enumerate(movement_limits):
            outside = np.abs(candidate_positions[:, axis]) > limit
            candidate_positions[outside, axis] = (
                sampled_positions[outside, axis] - step_size * directions[outside, axis]
            )
        sampled_positions = candidate_positions
        for path_points, position in zip(sampled_path_points, sampled_positions):
            path_points.append(position.copy())

    split_step = 5
    collision_particle_index = 2
    collision_step = 7
    surface_particle_index = 3
    crossing_step = 16
    assert (
        get_volume_index(sampled_path_points[collision_particle_index][collision_step - 1])
        == get_volume_index(sampled_path_points[collision_particle_index][collision_step])
        == get_volume_index(sampled_path_points[collision_particle_index][collision_step + 1])
    )
    assert (
        get_volume_index(sampled_path_points[surface_particle_index][crossing_step - 1])
        != get_volume_index(sampled_path_points[surface_particle_index][crossing_step])
    )
    left_scale = 0.5
    left_shift = 3.5 * LEFT
    left_panel_center = 3.5 * LEFT
    event_zoom = 2.15
    event_depth = 0.18 * OUT
    overview_frame_width = scene.camera.frame.get_width()
    overview_frame_center = scene.camera.frame.get_center().copy()

    def left_panel_point(point):
        return left_scale * point + left_shift

    def focus_camera_on(display_point):
        focus_center = overview_frame_center.copy()
        focus_center[:2] = (
            display_point[:2] - left_panel_center[:2] / event_zoom
        )
        return (
            scene.camera.frame.animate
            .set_width(overview_frame_width / event_zoom)
            .move_to(focus_center)
        )

    def make_movement_paths(start_step, end_step, split_screen=False):
        movement_paths = []
        for path_points in sampled_path_points:
            segment_points = path_points[start_step:end_step + 1]
            if split_screen:
                segment_points = [
                    left_panel_point(point)
                    for point in segment_points
                ]
            if len(segment_points) == 2:
                movement_paths.append(Line(segment_points[0], segment_points[1]))
            else:
                movement_paths.append(VMobject().set_points_smoothly(segment_points))
        return tuple(movement_paths)

    def play_particle_steps(start_step, end_step, run_time, split_screen=False):
        scene.play(
            *(
                MoveAlongPath(particle, path)
                for particle, path in zip(
                    particles,
                    make_movement_paths(start_step, end_step, split_screen),
                )
            ),
            run_time=run_time,
            rate_func=linear,
        )

    def get_unique_volume_infos(step):
        seen_indices = set()
        unique_infos = []
        for particle_index, path_points in enumerate(sampled_path_points):
            volume_index = get_volume_index(path_points[step])
            if volume_index in seen_indices:
                continue
            seen_indices.add(volume_index)
            unique_infos.append((
                particle_index,
                volume_index,
                particle_colors[particle_index],
            ))
        return tuple(unique_infos)

    def make_scalene_triangle(center, color, vertices, opacity=0.22):
        return Polygon(
            *(center + x * RIGHT + y * UP for x, y in vertices),
            fill_color=color,
            fill_opacity=opacity,
            stroke_color=color,
            stroke_width=1.3,
        )

    def make_bvh_tree(center, color, label, scale=1.0, event=None):
        def point(x, y):
            return center + scale * (x * RIGHT + y * UP) + event_depth

        root = Circle(
            radius=0.28 * scale,
            fill_color=color,
            fill_opacity=0.16,
            stroke_color=color,
            stroke_width=2,
        ).move_to(point(0, 0.65))
        root_label = Text(
            label,
            font_size=max(8, int(12 * scale)),
            color=color,
        ).move_to(root)
        left_node = Circle(
            radius=0.17 * scale,
            fill_color=BLUE_B,
            fill_opacity=0.20,
            stroke_color=BLUE_B,
            stroke_width=2,
        ).move_to(point(-0.43, 0.08))
        right_node = Circle(
            radius=0.17 * scale,
            fill_color=TEAL_A,
            fill_opacity=0.20,
            stroke_color=TEAL_A,
            stroke_width=2,
        ).move_to(point(0.43, 0.08))
        primitives = VGroup(
            make_scalene_triangle(
                point(-0.62, -0.55),
                BLUE_B,
                ((-0.15, -0.16), (0.02, 0.18), (0.16, -0.12)),
                opacity=0.32,
            ).scale(scale, about_point=point(-0.62, -0.55)),
            make_scalene_triangle(
                point(-0.25, -0.55),
                BLUE_B,
                ((-0.17, 0.15), (0.15, 0.10), (0.05, -0.18)),
                opacity=0.32,
            ).scale(scale, about_point=point(-0.25, -0.55)),
            make_scalene_triangle(
                point(0.43, -0.55),
                TEAL_A,
                ((-0.14, -0.17), (0.03, 0.19), (0.17, -0.13)),
                opacity=0.32,
            ).scale(scale, about_point=point(0.43, -0.55)),
        )
        branches = VGroup(
            Line(root.get_bottom(), left_node.get_top(), color=BLUE_B),
            Line(root.get_bottom(), right_node.get_top(), color=TEAL_A),
            Line(
                left_node.get_bottom(),
                primitives[0].get_top(),
                color=BLUE_B,
            ),
            Line(
                left_node.get_bottom(),
                primitives[1].get_top(),
                color=BLUE_B,
            ),
            Line(
                right_node.get_bottom(),
                primitives[2].get_top(),
                color=TEAL_A,
            ),
        )
        branches.set_stroke(width=max(1, 1.6 * scale))
        tree = VGroup(
            branches,
            root,
            root_label,
            left_node,
            right_node,
            primitives,
        ).deactivate_depth_test()
        if event is None:
            return tree

        route_points = [
            root.get_center(),
            left_node.get_center(),
        ]
        highlighted_objects = [
            root.copy(),
            left_node.copy(),
        ]
        if event == "surface":
            route_points.append(primitives[1].get_center())
            highlighted_objects.append(primitives[1].copy())

        traversal_path = VMobject()
        traversal_path.set_points_as_corners(route_points)
        traversal_path.set_stroke(YELLOW, width=3)
        traversal_pulse = GlowDot(
            route_points[0],
            color=YELLOW_A,
            radius=0.045,
        ).make_3d()
        traversal_highlights = VGroup(*highlighted_objects)
        traversal_highlights.set_fill(opacity=0)
        traversal_highlights.set_stroke(YELLOW, width=3)
        traversal = Group(
            traversal_path,
            traversal_pulse,
            traversal_highlights,
        ).deactivate_depth_test()
        return tree, traversal

    def make_bvh_geometry(center, color, label, scale=1.0, event=None):
        def point(x, y):
            return center + scale * (x * RIGHT + y * UP) + event_depth

        root_box = Rectangle(
            width=2.35 * scale,
            height=1.45 * scale,
            fill_color=color,
            fill_opacity=0.07,
            stroke_color=GREY_A,
            stroke_width=2.0,
        ).move_to(point(0, 0))
        root_label = Text(
            label,
            font_size=max(8, int(15 * scale)),
            color=color,
        ).move_to(point(-0.70, 0.56))
        left_box = Rectangle(
            width=1.05 * scale,
            height=0.92 * scale,
            fill_color=BLUE_B,
            fill_opacity=0.10,
            stroke_color=BLUE_B,
            stroke_width=1.6,
        ).move_to(point(-0.43, -0.10))
        right_box = Rectangle(
            width=0.72 * scale,
            height=0.86 * scale,
            fill_color=TEAL_A,
            fill_opacity=0.08,
            stroke_color=TEAL_A,
            stroke_width=1.4,
        ).move_to(point(0.70, -0.12))
        triangles = VGroup(
            make_scalene_triangle(
                point(-0.55, -0.13),
                BLUE_B,
                ((-0.24, 0.34), (-0.36, -0.34), (0.23, -0.25)),
                opacity=0.30,
            ).scale(scale, about_point=point(-0.55, -0.13)),
            make_scalene_triangle(
                point(-0.07, -0.08),
                BLUE_B,
                ((-0.42, 0.31), (0.27, -0.25), (-0.46, -0.29)),
                opacity=0.30,
            ).scale(scale, about_point=point(-0.07, -0.08)),
            make_scalene_triangle(
                point(0.68, -0.16),
                TEAL_A,
                ((-0.24, -0.34), (0.30, 0.34), (0.40, -0.32)),
                opacity=0.30,
            ).scale(scale, about_point=point(0.68, -0.16)),
        )
        bvh = VGroup(
            root_box,
            left_box,
            right_box,
            triangles,
            root_label,
        ).deactivate_depth_test()

        if event is None:
            return bvh

        ray_start = point(-1.42, 0.08)
        lambda_end = point(1.22, 0.08)
        if event == "collision":
            ray_end = point(-0.93, 0.08)
            lambda_end = ray_end
            event_label = Text(
                "no primitive hit",
                font_size=17,
                color=GREY_A,
            ).move_to(point(0, -1.20))
            tmax_label = Text(
                "ray limit: tMax = lambda (MFP)",
                font_size=18,
                color=YELLOW_A,
            ).move_to(point(0, -0.92))
            dashed_tail = VGroup()
            hit_marker = VGroup()
        else:
            ray_end = point(-0.08, 0.08)
            triangles[1].set_fill(YELLOW, opacity=0.42)
            triangles[1].set_stroke(YELLOW, width=2.5)
            event_label = Text(
                "primitive hit before lambda",
                font_size=17,
                color=YELLOW_A,
            ).move_to(point(0, -1.20))
            tmax_label = Text(
                "intersection: tHit < lambda (MFP)",
                font_size=18,
                color=YELLOW_A,
            ).move_to(point(0, -0.92))
            dashed_tail = DashedLine(
                ray_end,
                lambda_end,
                color=YELLOW_A,
                stroke_width=2,
                dash_length=0.07,
            )
            hit_marker = Dot(
                ray_end,
                radius=0.055,
                fill_color=YELLOW,
            )

        ray = StrokeArrow(
            ray_start,
            ray_end,
            stroke_color=YELLOW,
            stroke_width=4,
            buff=0.02,
        ).deactivate_depth_test()
        ray_path = Line(ray_start, ray_end)
        ray_pulse = GlowDot(
            ray_start,
            color=YELLOW_A,
            radius=0.06,
        ).make_3d()
        tmax_marker = Line(
            lambda_end + 0.14 * DOWN,
            lambda_end + 0.14 * UP,
            color=YELLOW_A,
            stroke_width=2.5,
        )
        ray_annotations = VGroup(
            tmax_marker,
            dashed_tail,
            hit_marker,
            tmax_label,
            event_label,
        ).deactivate_depth_test()
        return bvh, ray, ray_path, ray_pulse, ray_annotations

    def make_event_panel(title_text, color, bvh_label, event):
        title = Text(
            title_text,
            font_size=27,
            color=color,
        ).move_to(4.05 * RIGHT + 2.72 * UP + event_depth)
        bvh, ray, ray_path, ray_pulse, ray_annotations = make_bvh_geometry(
            3.25 * RIGHT + 0.65 * UP,
            color,
            bvh_label,
            scale=1.28,
            event=event,
        )
        bvh_tree, tree_traversal = make_bvh_tree(
            5.95 * RIGHT + 0.55 * UP,
            color,
            bvh_label,
            scale=0.95,
            event=event,
        )
        for mobject in (
            title,
            bvh,
            bvh_tree,
            tree_traversal,
            ray,
            ray_path,
            ray_pulse,
            ray_annotations,
        ):
            mobject.deactivate_depth_test()
            mobject.fix_in_frame()
        return (
            title,
            bvh,
            bvh_tree,
            tree_traversal,
            ray,
            ray_path,
            ray_pulse,
            ray_annotations,
        )

    def get_unit_vector(start, end):
        vector = end - start
        norm = np.linalg.norm(vector)
        if norm == 0:
            return RIGHT
        return vector / norm

    scene.wait(0.6)
    play_particle_steps(0, split_step, 3.0)
    scene.wait(0.8)

    split_unique_infos = get_unique_volume_infos(split_step)
    scene.play(
        *(
            volume_faces[index].animate.set_fill(color, opacity=0.22)
            for _, index, color in split_unique_infos
        ),
        *(
            volume_edges[index].animate.set_stroke(color, width=3)
            for _, index, color in split_unique_infos
        ),
        run_time=1.0,
    )

    cell_numbers = VGroup(
        *(
            Text(
                str(index + 1),
                font_size=28,
                color=color,
            ).move_to(volume_centers[index] + 0.66 * OUT)
            for _, index, color in split_unique_infos
        )
    ).deactivate_depth_test()
    scene.play(
        LaggedStart(
            *(FadeIn(number, scale=0.4, shift=0.35 * OUT) for number in cell_numbers),
            lag_ratio=0.12,
        ),
        run_time=1.0,
    )
    scene.wait(0.7)

    simulation = Group(
        volume_faces,
        particles,
        volume_edges,
        cell_numbers,
    )
    divider = Line(
        3.5 * UP,
        3.5 * DOWN,
        color=GREY_B,
        stroke_width=2,
    ).fix_in_frame()
    right_panel_mask = Rectangle(
        width=FRAME_WIDTH / 2,
        height=FRAME_HEIGHT,
        fill_color="#333333",
        fill_opacity=1,
        stroke_width=0,
    ).move_to((FRAME_WIDTH / 4) * RIGHT).fix_in_frame()

    scene.play(
        simulation.animate.scale(left_scale, about_point=ORIGIN).shift(left_shift),
        run_time=1.5,
    )
    scene.play(
        FadeIn(right_panel_mask),
        ShowCreation(divider),
        run_time=0.7,
    )
    scene.wait(0.6)

    shown_infos = split_unique_infos[:2] + split_unique_infos[-1:]
    representative_rows = (
        (
            0,
            shown_infos[0][1],
            shown_infos[0][2],
            f"VOLUME {shown_infos[0][1] + 1}",
            f"BVH {shown_infos[0][1] + 1}",
        ),
        (
            1,
            shown_infos[1][1],
            shown_infos[1][2],
            f"VOLUME {shown_infos[1][1] + 1}",
            f"BVH {shown_infos[1][1] + 1}",
        ),
        (
            len(split_unique_infos) - 1,
            shown_infos[-1][1],
            shown_infos[-1][2],
            "VOLUME n",
            "BVH n",
        ),
    )
    row_y_positions = (2.15, 0.80, -2.15)
    volume_face_targets = Group()
    volume_edge_targets = Group()
    volume_labels = VGroup()
    volume_centers_right = tuple(
        0.70 * RIGHT + y * UP + event_depth
        for y in row_y_positions
    )
    for (_, _, color, volume_label, _), center in zip(
        representative_rows,
        volume_centers_right,
    ):
        face = Square(
            side_length=0.52,
            fill_color=color,
            fill_opacity=0.22,
            stroke_width=0,
        ).move_to(center)
        edge = Square(
            side_length=0.52,
            fill_opacity=0,
            stroke_color=color,
            stroke_width=3,
        ).move_to(center)
        label = Text(
            volume_label,
            font_size=17,
            color=color,
        ).move_to(center + 0.58 * DOWN)
        volume_face_targets.add(face)
        volume_edge_targets.add(edge)
        volume_labels.add(label)
    volume_face_targets.deactivate_depth_test()
    volume_edge_targets.deactivate_depth_test()
    volume_labels.deactivate_depth_test()
    volume_ellipsis = VGroup(
        *(
            Dot(
                radius=0.035,
                fill_color=GREY_B,
            ).move_to(0.70 * RIGHT + (-0.42 + offset) * UP + event_depth)
            for offset in (0.18, 0.0, -0.18)
        )
    ).deactivate_depth_test()
    bvh_ellipsis = VGroup(
        *(
            Dot(
                radius=0.035,
                fill_color=GREY_B,
            ).move_to(5.05 * RIGHT + (-0.42 + offset) * UP + event_depth)
            for offset in (0.18, 0.0, -0.18)
        )
    ).deactivate_depth_test()
    compact_bvh_trees = VGroup(
        *(
            make_bvh_tree(
                5.05 * RIGHT + y * UP,
                color,
                bvh_label,
                scale=0.68,
            )
            for (_, _, color, _, bvh_label), y in zip(
                representative_rows,
                row_y_positions,
            )
        )
    )
    breakdown_rays = VGroup(
        *(
            StrokeArrow(
                1.35 * RIGHT + y * UP + event_depth,
                3.70 * RIGHT + y * UP + event_depth,
                stroke_color=color,
                stroke_width=3,
                buff=0.03,
            )
            for (_, _, color, _, _), y in zip(representative_rows, row_y_positions)
        )
    ).deactivate_depth_test()
    right_breakdown = Group(
        volume_face_targets,
        volume_edge_targets,
        volume_labels,
        volume_ellipsis,
        bvh_ellipsis,
        compact_bvh_trees,
        breakdown_rays,
    ).fix_in_frame()

    scene.play(
        *(
            TransformFromCopy(volume_faces[index], target)
            for (_, index, _, _, _), target in zip(
                representative_rows,
                volume_face_targets,
            )
        ),
        *(
            TransformFromCopy(volume_edges[index], target)
            for (_, index, _, _, _), target in zip(
                representative_rows,
                volume_edge_targets,
            )
        ),
        *(
            TransformFromCopy(cell_numbers[label_index], target)
            for (label_index, _, _, _, _), target in zip(
                representative_rows,
                volume_labels,
            )
        ),
        FadeIn(volume_ellipsis),
        run_time=1.5,
    )
    scene.play(
        LaggedStart(
            *(ShowCreation(ray) for ray in breakdown_rays),
            *(FadeIn(tree) for tree in compact_bvh_trees),
            FadeIn(bvh_ellipsis),
            lag_ratio=0.08,
        ),
        run_time=2.0,
    )
    scene.wait(1.8)
    scene.play(
        *(
            volume_faces[index].animate.set_fill(BLUE_D, opacity=0.025)
            for _, index, _ in split_unique_infos
        ),
        *(
            volume_edges[index].animate.set_stroke(BLUE_B, width=1.1)
            for _, index, _ in split_unique_infos
        ),
        FadeOut(cell_numbers),
        run_time=1.0,
    )
    scene.wait(0.6)

    play_particle_steps(split_step, collision_step, 1.8, split_screen=True)
    scene.wait(0.6)
    collision_color = particle_colors[collision_particle_index]
    collision_point = sampled_path_points[collision_particle_index][collision_step]
    collision_previous = sampled_path_points[collision_particle_index][collision_step - 1]
    collision_next = sampled_path_points[collision_particle_index][collision_step + 1]
    incoming_direction = get_unit_vector(collision_previous, collision_point)
    sampled_direction = get_unit_vector(collision_point, collision_next)
    collision_display_point = left_panel_point(collision_point) + event_depth
    collision_volume_index = get_volume_index(collision_point)
    collision_cell_highlight = volume_edges[collision_volume_index].copy()
    collision_cell_highlight.set_stroke(collision_color, width=4)
    collision_cell_highlight.deactivate_depth_test()
    collision_other_particles = Group(
        *(
            particle
            for index, particle in enumerate(particles)
            if index != collision_particle_index
        )
    )
    collision_marker = Circle(
        radius=0.11,
        stroke_color=collision_color,
        stroke_width=3,
        fill_opacity=0,
    ).move_to(collision_display_point).deactivate_depth_test()
    incoming_arrow = StrokeArrow(
        collision_display_point - 0.34 * incoming_direction,
        collision_display_point,
        stroke_color=GREY_A,
        stroke_width=2,
        buff=0.02,
    ).deactivate_depth_test()
    sampled_direction_arrow = StrokeArrow(
        collision_display_point,
        collision_display_point + 0.42 * sampled_direction,
        stroke_color=YELLOW,
        stroke_width=4,
        buff=0.02,
    ).deactivate_depth_test()
    collision_context = VGroup(
        Text(
            "TRACKED PARTICLE",
            font_size=14,
            color=GREY_B,
        ),
        Text(
            "sampled isotropic direction",
            font_size=19,
            color=YELLOW_A,
        ),
    ).arrange(DOWN, buff=0.10)
    collision_context.move_to(
        left_panel_center + 2.68 * UP + event_depth
    )
    collision_context.deactivate_depth_test()
    collision_context.fix_in_frame()
    (
        collision_title,
        collision_bvh,
        collision_bvh_tree,
        collision_tree_traversal,
        collision_ray,
        collision_ray_path,
        collision_pulse,
        collision_annotations,
    ) = make_event_panel(
        "COLLISION EVENT",
        collision_color,
        f"BVH {collision_volume_index + 1}",
        "collision",
    )
    (
        collision_tree_path,
        collision_tree_pulse,
        collision_tree_highlights,
    ) = collision_tree_traversal
    scene.play(
        FadeOut(right_breakdown),
        FadeIn(collision_title),
        FadeIn(collision_bvh),
        FadeIn(collision_bvh_tree),
        focus_camera_on(collision_display_point),
        collision_other_particles.animate.set_opacity(0.15),
        ShowCreation(collision_cell_highlight),
        ShowCreation(collision_marker),
        ShowCreation(incoming_arrow),
        FadeIn(collision_context),
        run_time=1.5,
    )
    scene.wait(0.8)
    scene.add(collision_pulse, collision_tree_pulse)
    scene.play(
        ShowCreation(collision_ray),
        MoveAlongPath(collision_pulse, collision_ray_path),
        ShowCreation(collision_tree_path),
        MoveAlongPath(collision_tree_pulse, collision_tree_path),
        FadeIn(collision_tree_highlights),
        run_time=1.5,
        rate_func=linear,
    )
    scene.wait(0.5)
    scene.play(
        FadeIn(collision_annotations),
        run_time=0.8,
    )
    scene.wait(0.6)
    scene.play(
        ShowCreation(sampled_direction_arrow),
        run_time=1.1,
    )
    scene.wait(2.2)
    scene.play(
        Restore(scene.camera.frame),
        collision_other_particles.animate.set_opacity(1),
        FadeOut(collision_cell_highlight),
        FadeOut(collision_marker),
        FadeOut(incoming_arrow),
        FadeOut(sampled_direction_arrow),
        FadeOut(collision_context),
        FadeOut(collision_title),
        FadeOut(collision_bvh),
        FadeOut(collision_bvh_tree),
        FadeOut(collision_tree_traversal),
        FadeOut(collision_ray),
        FadeOut(collision_pulse),
        FadeOut(collision_annotations),
        run_time=1.3,
    )
    scene.wait(0.5)

    play_particle_steps(collision_step, crossing_step - 1, 4.0, split_screen=True)
    scene.wait(0.7)
    surface_start = sampled_path_points[surface_particle_index][crossing_step - 1]
    surface_end = sampled_path_points[surface_particle_index][crossing_step]
    surface_next = sampled_path_points[surface_particle_index][crossing_step + 1]
    surface_start_volume_index = get_volume_index(surface_start)
    surface_end_volume_index = get_volume_index(surface_end)
    surface_start_center = volume_centers[surface_start_volume_index]
    surface_end_center = volume_centers[surface_end_volume_index]
    if abs(surface_start_center[0] - surface_end_center[0]) > 1e-6:
        boundary_x = 0.5 * (surface_start_center[0] + surface_end_center[0])
        boundary_start = np.array((
            boundary_x,
            surface_start_center[1] - 0.5 * volume_side_length,
            0,
        ))
        boundary_end = np.array((
            boundary_x,
            surface_start_center[1] + 0.5 * volume_side_length,
            0,
        ))
        alpha = (boundary_x - surface_start[0]) / (surface_end[0] - surface_start[0])
    else:
        boundary_y = 0.5 * (surface_start_center[1] + surface_end_center[1])
        boundary_start = np.array((
            surface_start_center[0] - 0.5 * volume_side_length,
            boundary_y,
            0,
        ))
        boundary_end = np.array((
            surface_start_center[0] + 0.5 * volume_side_length,
            boundary_y,
            0,
        ))
        alpha = (boundary_y - surface_start[1]) / (surface_end[1] - surface_start[1])
    boundary_point = surface_start + alpha * (surface_end - surface_start)
    surface_boundary = Line(
        left_panel_point(boundary_start) + event_depth,
        left_panel_point(boundary_end) + event_depth,
        color=YELLOW,
        stroke_width=5,
    ).deactivate_depth_test()
    surface_cell_highlights = VGroup(
        volume_edges[surface_start_volume_index].copy(),
        volume_edges[surface_end_volume_index].copy(),
    )
    surface_cell_highlights.set_stroke(YELLOW, width=3)
    surface_cell_highlights.deactivate_depth_test()
    surface_other_particles = Group(
        *(
            particle
            for index, particle in enumerate(particles)
            if index != surface_particle_index
        )
    )
    surface_context = VGroup(
        Text(
            "TRACKED PARTICLE",
            font_size=14,
            color=GREY_B,
        ),
        Text(
            "nearest boundary intersection",
            font_size=19,
            color=YELLOW_A,
        ),
    ).arrange(DOWN, buff=0.10)
    surface_context.move_to(
        left_panel_center + 2.68 * UP + event_depth
    )
    surface_context.deactivate_depth_test()
    surface_context.fix_in_frame()
    next_volume_context = VGroup(
        Text(
            f"VOLUME {surface_end_volume_index + 1}",
            font_size=14,
            color=particle_colors[surface_particle_index],
        ),
        Text(
            "new isotropic direction sampled",
            font_size=19,
            color=YELLOW_A,
        ),
    ).arrange(DOWN, buff=0.10)
    next_volume_context.move_to(
        left_panel_center + 2.68 * UP + event_depth
    )
    next_volume_context.deactivate_depth_test()
    next_volume_context.fix_in_frame()
    surface_end_display = left_panel_point(surface_end) + event_depth
    surface_next_direction = get_unit_vector(surface_end, surface_next)
    surface_next_direction_arrow = StrokeArrow(
        surface_end_display,
        surface_end_display + 0.45 * surface_next_direction,
        stroke_color=YELLOW,
        stroke_width=4,
        buff=0.02,
    ).deactivate_depth_test()
    (
        surface_title,
        surface_bvh,
        surface_bvh_tree,
        surface_tree_traversal,
        surface_ray,
        surface_ray_path,
        surface_pulse,
        surface_annotations,
    ) = make_event_panel(
        "SURFACE CROSSING EVENT",
        YELLOW,
        f"BVH {surface_start_volume_index + 1}",
        "surface",
    )
    (
        surface_tree_path,
        surface_tree_pulse,
        surface_tree_highlights,
    ) = surface_tree_traversal
    scene.play(
        FadeIn(surface_title),
        FadeIn(surface_bvh),
        FadeIn(surface_bvh_tree),
        focus_camera_on(left_panel_point(surface_start)),
        surface_other_particles.animate.set_opacity(0.15),
        ShowCreation(surface_cell_highlights),
        ShowCreation(surface_boundary),
        FadeIn(surface_context),
        run_time=1.5,
    )
    scene.wait(0.8)
    scene.add(surface_pulse, surface_tree_pulse)
    scene.play(
        ShowCreation(surface_ray),
        MoveAlongPath(surface_pulse, surface_ray_path),
        ShowCreation(surface_tree_path),
        MoveAlongPath(surface_tree_pulse, surface_tree_path),
        FadeIn(surface_tree_highlights),
        particles[surface_particle_index].animate.move_to(
            left_panel_point(boundary_point)
        ),
        run_time=1.8,
        rate_func=linear,
    )
    scene.play(
        FadeIn(surface_annotations),
        run_time=0.8,
    )
    scene.wait(1.5)
    scene.play(
        particles[surface_particle_index].animate.move_to(
            left_panel_point(surface_end)
        ),
        FadeOut(surface_cell_highlights[0]),
        surface_cell_highlights[1].animate.set_stroke(
            particle_colors[surface_particle_index],
            width=4,
        ),
        Transform(surface_context, next_volume_context),
        run_time=1.2,
        rate_func=linear,
    )
    scene.wait(0.5)
    scene.play(
        FadeOut(surface_boundary),
        ShowCreation(surface_next_direction_arrow),
        run_time=1.1,
    )
    scene.wait(2.0)


class EventSimDemo(ThreeDScene):
    """Standalone wrapper used when rendering this demonstration to MP4."""

    always_depth_test = False

    def construct(self):
        event_sim_scene(self)
