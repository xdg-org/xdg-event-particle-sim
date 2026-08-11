# Event Simulation Animation

The animation is split by responsibility so most changes only require opening one file:

- `model.py` defines the grid, particle count, deterministic particle paths, event queues, and volume lookup.
- `components.py` builds reusable diagrams: BVHs, traversal trees, event outcomes, and the death-event panel.
- `animation.py` owns scene state and sequences the five narrative phases. Shared collision and surface-crossing behavior lives in `play_showcase()`.
- `../event-sim.py` is the small ManimGL entry point and contains the discoverable `EventSimDemo` scene class.

## Narrative Order

`EventSimulation.play()` is the table of contents:

1. Initial particle launch and first queue advance.
2. Collision-event explanation and synchronized replays.
3. Surface-crossing explanation and synchronized replays.
4. Further advances and the volume/BVH incoherency breakdown.
5. Vacuum-boundary death event and replays.

## Common Edits

- Change particle positions or spread in `build_transport_model()`.
- Change BVH geometry or labels in `VisualFactory`.
- Change event timing in `play_showcase()` or the relevant `play_*_phase()` method.
- Change shared particle size, color, or split-screen geometry on `EventSimulation`.

Render from the repository root:

```bash
manimgl demonstrations/event-sim.py EventSimDemo -w
```

When already inside `demonstrations/`, omit the directory prefix:

```bash
manimgl event-sim.py EventSimDemo -w
```
