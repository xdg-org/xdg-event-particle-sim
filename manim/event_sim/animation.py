"""Phase orchestration for the event-based particle transport animation."""

from dataclasses import dataclass

import numpy as np
from manimlib import *

from .components import DeathPanel, EventPanel, ProcessDiagram, VisualFactory
from .model import TransportModel, build_transport_model


@dataclass
class EventShowcase:
    kind: str
    indices: tuple[int, ...]
    context: VGroup
    updated_context: VGroup
    replay_particles: VGroup
    replay_paths: tuple[VMobject, ...]
    panel: EventPanel
    process: ProcessDiagram
    boundaries: VGroup

    @property
    def moving_group(self):
        return Group(
            self.replay_particles,
            self.process.particles,
            self.process.paths,
            self.panel.ray,
            self.panel.pulse,
            self.panel.traversal.path,
            self.panel.traversal.pulse,
        )


class EventSimulation:
    """Own shared scene state and play the five narrative phases."""

    particle_color = RED_A
    particle_radius = 0.055
    left_scale = 0.5
    left_shift = 3.5 * LEFT
    left_panel_center = 3.5 * LEFT
    event_zoom = 2.15
    event_depth = 0.18 * OUT

    def __init__(self, scene):
        self.scene = scene
        self.model = build_transport_model()
        self.factory = VisualFactory(self.event_depth, self.particle_color)
        self.volume_faces = VGroup()
        self.volume_edges = VGroup()
        self.particles = Group()
        self.overview_frame_width = 0
        self.overview_frame_center = ORIGIN

    def play(self):
        self.setup_scene()
        divider, right_mask = self.play_initial_phase()
        collision = self.build_showcase("collision")
        collision_group = self.play_collision_phase(collision)
        surface = self.build_showcase("surface")
        self.play_surface_phase(surface, collision, collision_group)
        final_advance = self.play_transport_and_incoherency_phase()
        self.play_death_phase(final_advance, right_mask, divider)

    def setup_scene(self):
        scene = self.scene
        model = self.model
        scene.camera.frame.reorient(0, 0)
        scene.camera.frame.set_field_of_view(15 * DEGREES)
        scene.camera.light_source.move_to(4 * LEFT + 3 * UP + 6 * OUT)
        scene.camera.frame.save_state()

        self.volume_faces = VGroup(*(
            Square(
                side_length=model.volume_side_length,
                fill_color=BLUE_D,
                fill_opacity=0.025,
                stroke_width=0,
            ).move_to(center)
            for center in model.volume_centers
        )).deactivate_depth_test()
        self.volume_edges = VGroup(*(
            Square(
                side_length=model.volume_side_length,
                fill_opacity=0,
                stroke_color=BLUE_B,
                stroke_width=1.1,
            ).move_to(center)
            for center in model.volume_centers
        )).deactivate_depth_test()
        self.particles = Group(*(
            Sphere(
                radius=self.particle_radius,
                resolution=(51, 26),
                color=self.particle_color,
                shading=(0.25, 0.65, 0.2),
            ).move_to(position)
            for position in model.initial_positions
        ))
        scene.add(self.volume_faces, self.particles, self.volume_edges)
        scene.play(
            FadeIn(self.volume_faces),
            LaggedStart(
                *(ShowCreation(edges) for edges in self.volume_edges),
                lag_ratio=0.01,
            ),
            run_time=1.5,
        )
        self.overview_frame_width = scene.camera.frame.get_width()
        self.overview_frame_center = scene.camera.frame.get_center().copy()

    def left_point(self, point):
        return self.left_scale * point + self.left_shift

    def focus_camera_on(self, display_point):
        center = self.overview_frame_center.copy()
        center[:2] = (
            display_point[:2]
            - self.left_panel_center[:2] / self.event_zoom
        )
        return (
            self.scene.camera.frame.animate
            .set_width(self.overview_frame_width / self.event_zoom)
            .move_to(center)
        )

    def text_stack(self, lines, center, buff=0.10, fixed=True):
        """Build consistently positioned labels from text/style tuples."""
        labels = []
        for text, size, color, bold in lines:
            style = dict(font_size=size, color=color)
            if bold:
                style["weight"] = BOLD
            labels.append(Text(text, **style))
        group = VGroup(*labels).arrange(DOWN, buff=buff)
        group.move_to(center + self.event_depth)
        group.deactivate_depth_test()
        if fixed:
            group.fix_in_frame()
        return group

    def advance_context(self, number, title_size, body_size, center):
        body = VGroup(
            Text(
                "all active histories advance together",
                font_size=body_size, weight=BOLD, color=GREY_A,
            ),
            Text(
                "processing collisions and surface crossings together",
                font_size=body_size, weight=BOLD, color=GREY_A,
            ),
        ).arrange(DOWN, buff=0.05)
        context = VGroup(
            Text(
                f"PARTICLE ADVANCE {number}",
                font_size=title_size, weight=BOLD, color=YELLOW_A,
            ),
            body,
        ).arrange(DOWN, buff=0.10)
        context.move_to(center + self.event_depth)
        context.deactivate_depth_test().fix_in_frame()
        return context

    def play_initial_phase(self):
        scene = self.scene
        model = self.model
        initial_context = self.text_stack((
            ("INITIAL PARTICLE EVENT", 30, self.particle_color, True),
            ("isotropic directions sampled at one source", 21, GREY_A, True),
        ), 2.85 * UP, buff=0.12)
        scene.wait(0.6)
        scene.play(
            scene.camera.frame.animate.set_width(4.5).move_to(model.shared_start),
            FadeIn(initial_context),
            run_time=1.5,
        )
        scene.wait(2.0)

        advance_context = self.advance_context(1, 30, 18, 2.82 * UP)
        summary = Text(
            "8 COLLISIONS  |  2 SURFACE CROSSINGS",
            font_size=21, weight=BOLD, color=GREY_A,
        ).move_to(2.38 * DOWN + self.event_depth)
        summary.deactivate_depth_test().fix_in_frame()
        scene.play(Transform(initial_context, advance_context), run_time=0.6)
        scene.wait(0.5)
        scene.play(*(
            MoveAlongPath(
                self.particles[index],
                Line(
                    model.initial_positions[index],
                    model.first_event_positions[index],
                ),
            )
            for index in range(model.particle_count)
        ), run_time=2.4, rate_func=linear)
        scene.play(FadeIn(summary), run_time=0.8)
        scene.wait(1.8)

        simulation = Group(self.volume_faces, self.particles, self.volume_edges)
        divider = Line(
            3.5 * UP,
            3.5 * DOWN,
            color=GREY_B,
            stroke_width=2,
        ).fix_in_frame()
        right_mask = Rectangle(
            width=FRAME_WIDTH / 2,
            height=FRAME_HEIGHT,
            fill_color="#333333",
            fill_opacity=1,
            stroke_width=0,
        ).move_to((FRAME_WIDTH / 4) * RIGHT).fix_in_frame()
        scene.play(
            Restore(scene.camera.frame),
            FadeOut(initial_context),
            FadeOut(summary),
            run_time=1.2,
        )
        scene.play(
            simulation.animate.scale(self.left_scale, about_point=ORIGIN)
            .shift(self.left_shift),
            run_time=1.5,
        )
        scene.play(FadeIn(right_mask), ShowCreation(divider), run_time=0.7)
        scene.wait(0.7)
        return divider, right_mask

    def build_showcase(self, kind):
        model = self.model
        collision = kind == "collision"
        indices = model.collision_indices if collision else model.surface_indices
        heading = "COLLISION EVENTS" if collision else "SURFACE CROSSING EVENTS"
        panel_color = self.particle_color if collision else YELLOW
        context_color = self.particle_color if collision else YELLOW_A
        count_text = (
            "8 collision histories selected from the queue"
            if collision
            else "2 boundary histories selected from the queue"
        )
        context = self.text_stack((
            (heading, 21, context_color, True),
            (count_text, 19, GREY_A, False),
        ), self.left_panel_center + 3.16 * UP, buff=0.09)
        updated_context = self.text_stack(
            (
                ("ISOTROPIC DIRECTIONS SAMPLED", 20, YELLOW_A, True),
                (
                    "particle histories updated before next advance",
                    18,
                    GREY_A,
                    False,
                ),
            ) if collision else (
                ("TRANSMISSION", 21, TEAL_A, True),
                (
                    "new directions sampled; particle histories updated",
                    16,
                    YELLOW_A,
                    True,
                ),
            ),
            self.left_panel_center + 3.16 * UP,
            buff=0.09,
        )

        replay_particles = VGroup(*(
            Dot(
                self.left_point(model.shared_start) + self.event_depth,
                radius=self.particle_radius * self.left_scale,
                fill_color=self.particle_color,
            )
            for _ in indices
        )).deactivate_depth_test()
        path_type = Line if collision else VMobject
        replay_paths = []
        for index in indices:
            points = (
                self.left_point(model.shared_start) + self.event_depth,
                self.left_point(model.first_event_positions[index]) + self.event_depth,
            )
            path = path_type(*points) if collision else path_type().set_points_as_corners(points)
            replay_paths.append(path)

        source_volume = model.volume_index(model.shared_start)
        panel = self.factory.event_panel(
            heading,
            panel_color,
            f"BVH {source_volume + 1}",
            kind,
        )
        process = self.factory.process_diagram(kind, panel_color)
        boundaries = VGroup()
        if not collision:
            center = model.volume_centers[source_volume]
            boundaries = VGroup(*(
                Line(
                    self.left_point(
                        center
                        + 0.5 * model.volume_side_length * (LEFT + vertical)
                    ) + self.event_depth,
                    self.left_point(
                        center
                        + 0.5 * model.volume_side_length * (RIGHT + vertical)
                    ) + self.event_depth,
                    color=YELLOW,
                    stroke_width=5,
                )
                for vertical in (UP, DOWN)
            )).deactivate_depth_test()
        return EventShowcase(
            kind,
            indices,
            context,
            updated_context,
            replay_particles,
            tuple(replay_paths),
            panel,
            process,
            boundaries,
        )

    def reset_showcase_particles(self, showcase):
        for dot, path in zip(showcase.replay_particles, showcase.replay_paths):
            dot.move_to(path.get_start())
        for dot, path in zip(showcase.process.particles, showcase.process.paths):
            dot.move_to(path.get_start())

    def showcase_motion(self, showcase, include_bvh=False):
        animations = [
            *(
                MoveAlongPath(dot, path)
                for dot, path in zip(
                    showcase.replay_particles,
                    showcase.replay_paths,
                )
            ),
            *(ShowCreation(path) for path in showcase.process.paths),
            *(
                MoveAlongPath(dot, path)
                for dot, path in zip(
                    showcase.process.particles,
                    showcase.process.paths,
                )
            ),
        ]
        if include_bvh:
            animations.extend((
                ShowCreation(showcase.panel.ray),
                MoveAlongPath(showcase.panel.pulse, showcase.panel.ray_path),
                ShowCreation(showcase.panel.traversal.path),
                MoveAlongPath(
                    showcase.panel.traversal.pulse,
                    showcase.panel.traversal.path,
                ),
            ))
        return animations

    def replay_completed_showcase(self, showcase):
        scene = self.scene
        panel = showcase.panel
        process = showcase.process
        scene.play(
            FadeOut(showcase.replay_particles),
            FadeOut(process.particles),
            FadeOut(process.paths),
            FadeOut(panel.ray),
            FadeOut(panel.pulse),
            FadeOut(panel.traversal.path),
            FadeOut(panel.traversal.pulse),
            run_time=0.3,
        )
        self.reset_showcase_particles(showcase)
        panel.pulse.move_to(panel.ray_path.get_start())
        panel.traversal.pulse.move_to(panel.traversal.path.get_start())
        scene.play(
            FadeIn(showcase.replay_particles),
            FadeIn(process.particles),
            FadeIn(panel.pulse),
            FadeIn(panel.traversal.pulse),
            run_time=0.3,
        )
        scene.play(
            *self.showcase_motion(showcase, include_bvh=True),
            run_time=2.4,
            rate_func=linear,
        )

    def play_showcase(self, showcase, intro_animations):
        scene = self.scene
        panel = showcase.panel
        process = showcase.process
        scene.play(
            *intro_animations,
            FadeIn(showcase.context),
            run_time=1.4,
        )
        scene.wait(0.8)

        scene.play(FadeIn(showcase.replay_particles), run_time=0.35)
        scene.play(*(
            MoveAlongPath(dot, path)
            for dot, path in zip(showcase.replay_particles, showcase.replay_paths)
        ), run_time=2.4, rate_func=linear)
        scene.wait(1.0)
        scene.play(FadeOut(showcase.replay_particles), run_time=0.35)
        self.reset_showcase_particles(showcase)

        scene.play(
            FadeIn(panel.title),
            FadeIn(process.domain),
            FadeIn(process.label),
            FadeIn(process.particles),
            FadeIn(showcase.replay_particles),
            run_time=0.9,
        )
        scene.play(
            *self.showcase_motion(showcase),
            run_time=2.4,
            rate_func=linear,
        )
        scene.play(
            FadeIn(process.hits),
            FadeIn(process.outcome[0]),
            run_time=0.6,
        )
        scene.wait(1.0)
        scene.play(
            FadeOut(showcase.replay_particles),
            FadeOut(process.particles),
            FadeOut(process.paths),
            run_time=0.35,
        )
        self.reset_showcase_particles(showcase)

        scene.play(
            FadeIn(panel.bvh),
            FadeIn(panel.tree),
            FadeIn(showcase.replay_particles),
            FadeIn(process.particles),
            run_time=0.9,
        )
        scene.add(panel.pulse, panel.traversal.pulse)
        scene.play(
            *self.showcase_motion(showcase, include_bvh=True),
            FadeIn(panel.traversal.highlights),
            run_time=2.4,
            rate_func=linear,
        )
        if showcase.kind == "surface":
            for index in showcase.indices:
                self.particles[index].move_to(
                    self.left_point(self.model.post_event_positions[index])
                )
        scene.play(
            FadeIn(process.outcome[1]),
            FadeIn(panel.annotations),
            Transform(showcase.context, showcase.updated_context),
            run_time=0.9,
        )
        scene.wait(1.2)
        for _ in range(2):
            self.replay_completed_showcase(showcase)
            scene.wait(0.6)
        return panel.group_with(process, showcase.replay_particles)

    def play_collision_phase(self, showcase):
        model = self.model
        collision_particles = Group(*(
            self.particles[index] for index in model.collision_indices
        ))
        surface_particles = Group(*(
            self.particles[index] for index in model.surface_indices
        ))
        source_center = model.volume_centers[model.volume_index(model.shared_start)]
        return self.play_showcase(showcase, (
            self.focus_camera_on(self.left_point(source_center)),
            collision_particles.animate.set_opacity(0.08),
            surface_particles.animate.set_opacity(0.08),
        ))

    def play_surface_phase(self, showcase, collision, collision_group):
        scene = self.scene
        surface_particles = Group(*(
            self.particles[index] for index in self.model.surface_indices
        ))
        group = self.play_showcase(showcase, (
            FadeOut(collision_group),
            FadeOut(collision.context),
            surface_particles.animate.set_opacity(0.08),
            FadeIn(showcase.boundaries),
        ))
        scene.play(
            Restore(scene.camera.frame),
            self.particles.animate.set_opacity(1),
            FadeOut(group),
            FadeOut(showcase.boundaries),
            FadeOut(showcase.context),
            run_time=1.3,
        )

    def play_transport_and_incoherency_phase(self):
        scene = self.scene
        model = self.model
        advance_context = self.advance_context(
            2, 24, 16, self.left_panel_center + 2.72 * UP
        )
        for number, start_step in enumerate((2, 3, 4), start=2):
            next_context = self.advance_context(
                number, 24, 16, self.left_panel_center + 2.72 * UP
            )
            context_animation = (
                FadeIn(advance_context)
                if number == 2
                else Transform(advance_context, next_context)
            )
            scene.play(
                context_animation,
                *(
                    MoveAlongPath(
                        self.particles[index],
                        Line(
                            self.left_point(model.sampled_path_points[index][start_step]),
                            self.left_point(
                                model.sampled_path_points[index][start_step + 1]
                            ),
                        ),
                    )
                    for index in range(model.particle_count)
                ),
                run_time=1.5,
                rate_func=linear,
            )
            scene.wait(0.7)

        self.play_incoherency_breakdown(advance_context)
        final_context = self.advance_context(
            5, 24, 16, self.left_panel_center + 2.72 * UP
        )
        scene.play(
            FadeIn(final_context),
            *(
                MoveAlongPath(
                    self.particles[index],
                    Line(
                        self.left_point(
                            model.sampled_path_points[index][model.split_step]
                        ),
                        self.left_point(
                            model.sampled_path_points[index][model.final_advance_step]
                        ),
                    ),
                )
                for index in range(model.particle_count)
            ),
            run_time=1.5,
            rate_func=linear,
        )
        scene.wait(0.9)
        return final_context

    def play_incoherency_breakdown(self, advance_context):
        scene = self.scene
        model = self.model
        unique = model.unique_volumes(model.split_step)
        volume_context = self.text_stack((
            ("VOLUME INCOHERENCY", 26, self.particle_color, True),
            ("particles spread across multiple model volumes", 17, GREY_A, False),
        ), self.left_panel_center + 2.70 * UP)
        bvh_context = self.text_stack((
            ("BVH INCOHERENCY", 26, self.particle_color, True),
            ("incoherent ray work split across multiple BVHs", 17, GREY_A, False),
        ), 4.05 * RIGHT + 3.08 * UP)
        scene.play(
            FadeOut(advance_context),
            FadeIn(volume_context),
            FadeIn(bvh_context),
            *(
                self.volume_faces[index].animate.set_fill(
                    self.particle_color, opacity=0.22
                )
                for _, index in unique
            ),
            *(
                self.volume_edges[index].animate.set_stroke(
                    self.particle_color, width=3
                )
                for _, index in unique
            ),
            run_time=1.0,
        )
        cell_numbers = VGroup(*(
            Text(
                str(index + 1),
                font_size=30, weight=BOLD, color=self.particle_color,
            ).move_to(self.volume_edges[index].get_center())
            for _, index in unique
        )).deactivate_depth_test()
        scene.play(LaggedStart(*(
            FadeIn(number, scale=0.4, shift=0.20 * OUT)
            for number in cell_numbers
        ), lag_ratio=0.08), run_time=1.0)
        scene.wait(0.8)

        shown = unique[:2] + unique[-1:]
        rows = (
            (shown[0][1], f"VOLUME {shown[0][1] + 1}", f"BVH {shown[0][1] + 1}"),
            (shown[1][1], f"VOLUME {shown[1][1] + 1}", f"BVH {shown[1][1] + 1}"),
            (shown[-1][1], "VOLUME n", "BVH n"),
        )
        y_positions = (2.15, 0.80, -2.15)
        faces, edges, labels = Group(), Group(), VGroup()
        for (_, volume_label, _), y in zip(rows, y_positions):
            center = 0.70 * RIGHT + y * UP + self.event_depth
            faces.add(Square(
                side_length=0.52,
                fill_color=self.particle_color,
                fill_opacity=0.22,
                stroke_width=0,
            ).move_to(center))
            edges.add(Square(
                side_length=0.52,
                fill_opacity=0,
                stroke_color=self.particle_color,
                stroke_width=3,
            ).move_to(center))
            labels.add(Text(
                volume_label,
                font_size=18, weight=BOLD, color=self.particle_color,
            ).move_to(center + 0.58 * DOWN))
        faces.deactivate_depth_test()
        edges.deactivate_depth_test()
        labels.deactivate_depth_test()
        volume_dots = self.vertical_ellipsis(0.70 * RIGHT - 0.42 * UP)
        bvh_dots = self.vertical_ellipsis(5.05 * RIGHT - 0.42 * UP)
        trees = VGroup(*(
            self.factory.bvh_tree(
                5.05 * RIGHT + y * UP,
                self.particle_color,
                bvh_label,
                scale=0.68,
            )
            for (_, _, bvh_label), y in zip(rows, y_positions)
        ))
        rays = VGroup(*(
            StrokeArrow(
                1.35 * RIGHT + y * UP + self.event_depth,
                3.70 * RIGHT + y * UP + self.event_depth,
                stroke_color=self.particle_color,
                stroke_width=3,
                buff=0.03,
            )
            for y in y_positions
        )).deactivate_depth_test()
        right_breakdown = Group(
            faces, edges, labels, volume_dots, bvh_dots, trees, rays
        ).fix_in_frame()
        scene.play(
            *(
                TransformFromCopy(self.volume_faces[index], target)
                for (index, _, _), target in zip(rows, faces)
            ),
            *(
                TransformFromCopy(self.volume_edges[index], target)
                for (index, _, _), target in zip(rows, edges)
            ),
            FadeIn(labels),
            FadeIn(volume_dots),
            run_time=1.5,
        )
        scene.play(LaggedStart(
            *(ShowCreation(ray) for ray in rays),
            *(FadeIn(tree) for tree in trees),
            FadeIn(bvh_dots),
            lag_ratio=0.08,
        ), run_time=2.0)
        scene.wait(1.8)
        scene.play(
            *(
                self.volume_faces[index].animate.set_fill(BLUE_D, opacity=0.025)
                for _, index in unique
            ),
            *(
                self.volume_edges[index].animate.set_stroke(BLUE_B, width=1.1)
                for _, index in unique
            ),
            FadeOut(cell_numbers),
            FadeOut(right_breakdown),
            FadeOut(volume_context),
            FadeOut(bvh_context),
            run_time=1.0,
        )

    def vertical_ellipsis(self, center):
        return VGroup(*(
            Dot(radius=0.035, fill_color=GREY_B).move_to(
                center + offset * UP + self.event_depth
            )
            for offset in (0.18, 0.0, -0.18)
        )).deactivate_depth_test()

    def play_death_phase(self, final_context, right_mask, divider):
        scene = self.scene
        model = self.model
        index = 0
        start = model.sampled_path_points[index][model.final_advance_step]
        approach = np.array((start[0], model.grid_height / 2 - 0.45, start[2]))
        boundary_point = np.array((start[0], model.grid_height / 2, start[2]))
        other_particles = Group(*(
            particle
            for particle_index, particle in enumerate(self.particles)
            if particle_index != index
        ))
        perimeter = SurroundingRectangle(
            self.volume_edges,
            buff=0,
            fill_opacity=0,
            stroke_color=RED_A,
            stroke_width=4,
        ).shift(self.event_depth).deactivate_depth_test()
        context = self.text_stack((
            ("TRACKED PARTICLE", 18, GREY_B, True),
            ("approaching model boundary", 21, self.particle_color, False),
        ), self.left_panel_center + 2.68 * UP)

        scene.play(
            Restore(scene.camera.frame),
            FadeOut(final_context),
            run_time=0.8,
        )
        scene.add(perimeter)
        scene.bring_to_front(right_mask, divider)
        scene.play(
            ShowCreation(perimeter),
            FadeIn(context),
            other_particles.animate.set_opacity(0.12),
            MoveAlongPath(
                self.particles[index],
                Line(self.left_point(start), self.left_point(approach)),
            ),
            run_time=2.8,
            rate_func=linear,
        )
        scene.wait(0.8)

        death_context = self.text_stack((
            ("VACUUM BOUNDARY", 18, RED_A, True),
            ("particle history terminates", 21, RED_A, False),
        ), self.left_panel_center + 2.68 * UP)
        panel = self.build_death_panel()
        scene.play(
            self.focus_camera_on(self.left_point(approach) + self.event_depth),
            Transform(context, death_context),
            FadeIn(panel.title),
            FadeIn(panel.subtitle),
            FadeIn(panel.domain),
            ShowCreation(panel.boundary),
            FadeIn(panel.vacuum_label),
            FadeIn(panel.model_label),
            FadeIn(panel.particle),
            run_time=1.5,
        )
        scene.wait(0.8)
        scene.add(panel.pulse)
        self.play_death_motion(index, boundary_point, panel, first=True)
        scene.wait(1.2)
        for _ in range(2):
            scene.play(
                FadeOut(panel.ray),
                FadeOut(panel.pulse),
                run_time=0.3,
            )
            self.particles[index].move_to(self.left_point(approach))
            panel.particle.move_to(panel.path.get_start())
            panel.pulse.move_to(panel.path.get_start())
            scene.play(
                FadeIn(self.particles[index]),
                FadeIn(panel.particle),
                FadeIn(panel.pulse),
                run_time=0.3,
            )
            self.play_death_motion(index, boundary_point, panel)
            scene.wait(0.6)

    def build_death_panel(self):
        depth = self.event_depth
        title = Text("DEATH EVENT", font_size=30, weight=BOLD, color=RED_A)
        title.move_to(4.05 * RIGHT + 2.72 * UP + depth)
        subtitle = Text(
            "Surface Boundary Condition: VACUUM", font_size=18, color=GREY_A,
        ).move_to(4.05 * RIGHT + 2.25 * UP + depth)
        domain = Rectangle(
            width=2.8,
            height=1.25,
            fill_color=BLUE_D,
            fill_opacity=0.10,
            stroke_color=BLUE_B,
            stroke_width=2,
        ).move_to(4.05 * RIGHT - 0.05 * UP + depth)
        boundary_point = 4.05 * RIGHT + 0.575 * UP + depth
        boundary = Line(
            2.45 * RIGHT + 0.575 * UP + depth,
            5.65 * RIGHT + 0.575 * UP + depth,
            color=RED_A,
            stroke_width=5,
        )
        vacuum_label = Text("VACUUM", font_size=20, weight=BOLD, color=RED_A)
        vacuum_label.move_to(4.05 * RIGHT + 1.05 * UP + depth)
        model_label = Text("MODEL", font_size=18, weight=BOLD, color=BLUE_A)
        model_label.move_to(4.05 * RIGHT - 0.43 * UP + depth)
        particle = Dot(
            4.05 * RIGHT - 0.27 * UP + depth,
            radius=0.05,
            fill_color=self.particle_color,
        )
        path = Line(particle.get_center(), boundary_point)
        ray = StrokeArrow(
            path.get_start(),
            path.get_end(),
            stroke_color=YELLOW,
            stroke_width=4,
            buff=0.04,
        )
        pulse = GlowDot(
            path.get_start(), color=YELLOW_A, radius=0.055
        ).make_3d()
        hit = VGroup(
            Line(
                boundary_point + 0.10 * (LEFT + DOWN),
                boundary_point + 0.10 * (RIGHT + UP),
                color=RED_A,
                stroke_width=3,
            ),
            Line(
                boundary_point + 0.10 * (LEFT + UP),
                boundary_point + 0.10 * (RIGHT + DOWN),
                color=RED_A,
                stroke_width=3,
            ),
        )
        outcome = VGroup(
            Text("particle leaves the model", font_size=19, color=GREY_A),
            Text("history terminated", font_size=21, weight=BOLD, color=RED_A),
        ).arrange(DOWN, buff=0.16)
        outcome.move_to(4.05 * RIGHT - 1.25 * UP + depth)
        panel = DeathPanel(
            title,
            subtitle,
            domain,
            boundary,
            vacuum_label,
            model_label,
            particle,
            path,
            ray,
            pulse,
            hit,
            outcome,
        )
        panel.group.deactivate_depth_test().fix_in_frame()
        return panel

    def play_death_motion(self, index, boundary_point, panel, first=False):
        scene = self.scene
        scene.play(
            ShowCreation(panel.ray),
            MoveAlongPath(panel.pulse, panel.path),
            MoveAlongPath(panel.particle, panel.path),
            self.particles[index].animate.move_to(self.left_point(boundary_point)),
            run_time=1.5,
            rate_func=linear,
        )
        animations = [
            FadeOut(self.particles[index], scale=0.25),
            FadeOut(panel.particle, scale=0.25),
        ]
        if first:
            animations.extend((FadeIn(panel.hit), FadeIn(panel.outcome)))
        scene.play(*animations, run_time=1.0 if first else 0.6)


def event_sim_scene(scene):
    """Play the demonstration on an existing Scene or Slide."""
    EventSimulation(scene).play()
