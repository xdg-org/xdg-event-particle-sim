"""Reusable visual components for event and BVH explanations."""

from dataclasses import dataclass

import numpy as np
from manimlib import *


@dataclass
class TreeTraversal:
    path: VMobject
    pulse: GlowDot
    highlights: VGroup

    @property
    def group(self):
        return Group(self.path, self.pulse, self.highlights)


@dataclass
class EventPanel:
    title: Text
    bvh: VGroup
    tree: VGroup
    traversal: TreeTraversal
    ray: StrokeArrow
    ray_path: Line
    pulse: GlowDot
    annotations: VGroup

    def group_with(self, process, replay_visuals):
        return Group(
            self.title,
            self.bvh,
            self.tree,
            self.traversal.group,
            self.ray,
            self.pulse,
            self.annotations,
            process.group,
            replay_visuals,
        )


@dataclass
class ProcessDiagram:
    domain: Rectangle
    label: Text
    particles: VGroup
    arrows: VGroup
    paths: VGroup
    continuations: VGroup
    hits: VGroup
    outcome: VGroup
    group: Group


@dataclass
class DeathPanel:
    title: Text
    subtitle: Text
    domain: Rectangle
    boundary: Line
    vacuum_label: Text
    model_label: Text
    particle: Dot
    path: Line
    ray: StrokeArrow
    pulse: GlowDot
    hit: VGroup
    outcome: VGroup

    @property
    def group(self):
        return Group(
            self.title,
            self.subtitle,
            self.domain,
            self.boundary,
            self.vacuum_label,
            self.model_label,
            self.particle,
            self.path,
            self.ray,
            self.pulse,
            self.hit,
            self.outcome,
        )


class VisualFactory:
    def __init__(self, event_depth, particle_color):
        self.event_depth = event_depth
        self.particle_color = particle_color

    def triangle(self, center, color, vertices, opacity=0.22):
        return Polygon(
            *(center + x * RIGHT + y * UP for x, y in vertices),
            fill_color=color,
            fill_opacity=opacity,
            stroke_color=color,
            stroke_width=1.3,
        )

    def attached_direction_arrow(
        self,
        particle,
        direction,
        length=0.16,
        gap=0.035,
        stroke_width=2.2,
    ):
        direction = direction / np.linalg.norm(direction)
        arrow = StrokeArrow(
            ORIGIN,
            RIGHT,
            stroke_color=YELLOW,
            stroke_width=stroke_width,
            buff=0,
            tip_width_ratio=3.5,
        )

        def position(direction_arrow):
            center = particle.get_center()
            direction_arrow.put_start_and_end_on(
                center + gap * direction,
                center + length * direction,
            )

        position(arrow)
        arrow.add_updater(position)
        return arrow

    def bvh_tree(self, center, color, label, scale=1.0, event=None):
        def point(x, y):
            return center + scale * (x * RIGHT + y * UP) + self.event_depth

        root = Circle(
            radius=0.28 * scale,
            fill_color=color,
            fill_opacity=0.16,
            stroke_color=color,
            stroke_width=2,
        ).move_to(point(0, 0.65))
        root_label = Text(label, font_size=max(8, int(12 * scale)), color=color).move_to(root)
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
            self.triangle(
                point(-0.62, -0.55),
                BLUE_B,
                ((-0.15, -0.16), (0.02, 0.18), (0.16, -0.12)),
                opacity=0.32,
            ).scale(scale, about_point=point(-0.62, -0.55)),
            self.triangle(
                point(-0.25, -0.55),
                BLUE_B,
                ((-0.17, 0.15), (0.15, 0.10), (0.05, -0.18)),
                opacity=0.32,
            ).scale(scale, about_point=point(-0.25, -0.55)),
            self.triangle(
                point(0.43, -0.55),
                TEAL_A,
                ((-0.14, -0.17), (0.03, 0.19), (0.17, -0.13)),
                opacity=0.32,
            ).scale(scale, about_point=point(0.43, -0.55)),
        )
        branches = VGroup(
            Line(root.get_bottom(), left_node.get_top(), color=BLUE_B),
            Line(root.get_bottom(), right_node.get_top(), color=TEAL_A),
            Line(left_node.get_bottom(), primitives[0].get_top(), color=BLUE_B),
            Line(left_node.get_bottom(), primitives[1].get_top(), color=BLUE_B),
            Line(right_node.get_bottom(), primitives[2].get_top(), color=TEAL_A),
        ).set_stroke(width=max(1, 1.6 * scale))
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

        route = [root.get_center(), left_node.get_center()]
        highlighted = [root.copy(), left_node.copy()]
        if event == "surface":
            route.append(primitives[1].get_center())
            highlighted.append(primitives[1].copy())
        path = VMobject().set_points_as_corners(route).set_stroke(YELLOW, width=3)
        pulse = GlowDot(route[0], color=YELLOW_A, radius=0.045).make_3d()
        highlights = VGroup(*highlighted)
        highlights.set_fill(opacity=0).set_stroke(YELLOW, width=3)
        return tree, TreeTraversal(path, pulse, highlights)

    def bvh_geometry(self, center, color, label, scale=1.0, event=None):
        def point(x, y):
            return center + scale * (x * RIGHT + y * UP) + self.event_depth

        root_box = Rectangle(
            width=2.35 * scale,
            height=1.45 * scale,
            fill_color=color,
            fill_opacity=0.07,
            stroke_color=GREY_A,
            stroke_width=2.0,
        ).move_to(point(0, 0))
        root_label = Text(label, font_size=max(8, int(15 * scale)), color=color)
        root_label.move_to(point(-0.70, 0.56))
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
            self.triangle(
                point(-0.55, -0.13),
                BLUE_B,
                ((-0.24, 0.34), (-0.36, -0.34), (0.23, -0.25)),
                opacity=0.30,
            ).scale(scale, about_point=point(-0.55, -0.13)),
            self.triangle(
                point(-0.07, -0.08),
                BLUE_B,
                ((-0.42, 0.31), (0.27, -0.25), (-0.46, -0.29)),
                opacity=0.30,
            ).scale(scale, about_point=point(-0.07, -0.08)),
            self.triangle(
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
            ray_buff = 0.02
            event_label = Text("no primitive hit", font_size=17, color=GREY_A)
            event_label.move_to(point(0, -1.20))
            limit_label = Text("ray limit: tMax = lambda (MFP)", font_size=18, color=YELLOW_A)
            limit_label.move_to(point(0, -0.92))
            dashed_tail = VGroup()
            hit_marker = VGroup()
        else:
            hit_triangle = triangles[1]
            vertices = hit_triangle.get_vertices()
            ray_y = ray_start[1]
            intersections = []
            for start, end in zip(vertices, np.roll(vertices, -1, axis=0)):
                if abs(end[1] - start[1]) < 1e-8:
                    continue
                alpha = (ray_y - start[1]) / (end[1] - start[1])
                if 0 <= alpha <= 1:
                    intersections.append(start[0] + alpha * (end[0] - start[0]))
            hit_xs = sorted(
                x for x in intersections
                if ray_start[0] <= x <= lambda_end[0]
            )
            ray_end = np.array((
                0.5 * (hit_xs[0] + hit_xs[-1]),
                ray_y,
                ray_start[2],
            ))
            ray_buff = 0
            hit_triangle.set_fill(YELLOW, opacity=0.42)
            hit_triangle.set_stroke(YELLOW, width=2.5)
            event_label = Text("primitive hit before lambda", font_size=17, color=YELLOW_A)
            event_label.move_to(point(0, -1.20))
            limit_label = Text("intersection: tHit < lambda (MFP)", font_size=18, color=YELLOW_A)
            limit_label.move_to(point(0, -0.92))
            dashed_tail = DashedLine(
                ray_end,
                lambda_end,
                color=YELLOW_A,
                stroke_width=2,
                dash_length=0.07,
            )
            hit_marker = Dot(ray_end, radius=0.055, fill_color=YELLOW)

        ray = StrokeArrow(
            ray_start,
            ray_end,
            stroke_color=YELLOW,
            stroke_width=4,
            buff=ray_buff,
        ).deactivate_depth_test()
        ray_path = Line(ray_start, ray_end)
        pulse = GlowDot(ray_start, color=YELLOW_A, radius=0.06).make_3d()
        marker = Line(
            lambda_end + 0.14 * DOWN,
            lambda_end + 0.14 * UP,
            color=YELLOW_A,
            stroke_width=2.5,
        )
        annotations = VGroup(
            marker,
            dashed_tail,
            hit_marker,
            limit_label,
            event_label,
        ).deactivate_depth_test()
        return bvh, ray, ray_path, pulse, annotations

    def event_panel(self, title_text, color, bvh_label, event):
        title = Text(title_text, font_size=30, weight=BOLD, color=color)
        title.move_to(4.05 * RIGHT + 2.92 * UP + self.event_depth)
        bvh, ray, ray_path, pulse, annotations = self.bvh_geometry(
            3.25 * RIGHT - 0.88 * UP,
            color,
            bvh_label,
            scale=0.92,
            event=event,
        )
        tree, traversal = self.bvh_tree(
            5.92 * RIGHT - 0.92 * UP,
            color,
            bvh_label,
            scale=0.72,
            event=event,
        )
        for mobject in (
            title, bvh, tree, traversal.group, ray, ray_path, pulse, annotations,
        ):
            mobject.deactivate_depth_test()
            mobject.fix_in_frame()
        return EventPanel(
            title, bvh, tree, traversal, ray, ray_path, pulse, annotations
        )

    def process_diagram(self, event, color):
        center = 4.05 * RIGHT + 1.18 * UP + self.event_depth
        domain = Rectangle(
            width=3.2,
            height=1.18,
            fill_color=BLUE_D,
            fill_opacity=0.10,
            stroke_color=BLUE_B,
            stroke_width=2,
        ).move_to(center)
        label = Text("EVENT OUTCOME", font_size=15, weight=BOLD, color=GREY_B)
        label.move_to(center + 0.43 * UP)

        if event == "collision":
            starts = (
                center + 1.15 * LEFT + 0.22 * UP,
                center + 1.02 * LEFT + 0.08 * DOWN,
                center + 0.55 * LEFT + 0.27 * DOWN,
                center + 0.25 * LEFT + 0.12 * UP,
            )
            ends = (
                center + 0.42 * LEFT + 0.08 * UP,
                center + 0.18 * LEFT + 0.25 * DOWN,
                center + 0.28 * RIGHT + 0.12 * DOWN,
                center + 0.62 * RIGHT + 0.18 * UP,
            )
            continuations = VGroup()
            hits = VGroup()
            outcome = VGroup(
                Text("MFP sampled inside volume", font_size=17, weight=BOLD, color=YELLOW_A),
                Text("particle histories updated", font_size=15, color=GREY_A),
            ).arrange(DOWN, buff=0.10)
        else:
            boundary_x = center[0] + 0.28
            starts = (
                np.array((center[0] - 1.10, center[1] + 0.18, center[2])),
                np.array((center[0] - 0.92, center[1] - 0.20, center[2])),
            )
            ends = tuple(
                np.array((boundary_x, start[1], center[2]))
                for start in starts
            )
            domain.add(Line(
                np.array((boundary_x, center[1] - 0.40, center[2])),
                np.array((boundary_x, center[1] + 0.34, center[2])),
                color=YELLOW,
                stroke_width=4,
            ))
            continuations = VGroup(*(
                Line(hit, hit + 0.42 * RIGHT, color=TEAL_A, stroke_width=3)
                for hit in ends
            ))
            hits = VGroup(*(
                Dot(hit, radius=0.045, fill_color=YELLOW)
                for hit in ends
            ))
            outcome = VGroup(
                Text(
                    "Surface Boundary Condition: TRANSMISSION",
                    font_size=15, weight=BOLD, color=TEAL_A,
                ),
                Text("particle histories updated", font_size=15, color=GREY_A),
            ).arrange(DOWN, buff=0.10)

        particles = VGroup(*(
            Dot(start, radius=0.045, fill_color=self.particle_color)
            for start in starts
        ))
        paths = VGroup(*(
            Line(start, end, color=YELLOW, stroke_width=3)
            for start, end in zip(starts, ends)
        ))
        arrows = VGroup(*(
            self.attached_direction_arrow(
                particle,
                end - start,
                length=0.21,
                gap=0.05,
                stroke_width=2.6,
            )
            for particle, start, end in zip(particles, starts, ends)
        ))
        outcome.move_to(4.05 * RIGHT + 0.28 * UP + self.event_depth)
        group = Group(
            domain, label, particles, arrows, paths, continuations, hits, outcome
        ).deactivate_depth_test()
        group.fix_in_frame()
        return ProcessDiagram(
            domain,
            label,
            particles,
            arrows,
            paths,
            continuations,
            hits,
            outcome,
            group,
        )
