# xdg-event-particle-sim

Pseudo particle transport applications for mocking neutral-particle transport
against CAD models with the XDG library. The primary application uses a GPU
event-based algorithm and acts as a mock interface for XDG's GPU API. A second
history-based CPU application provides an Embree reference for performance and
cell track-length comparisons. GPU sorting of event queues by volume, direction
octant, or compound combinations of the two is provided through CUDA or HIP
Thrust.

## Building

This application is built against an installed XDG package. Build and install
XDG with cuBQL and Embree support first, then point this project at that
installation.
The XDG build uses XDG's `cubql_llvm_ada` preset, while this standalone
application uses its own `llvm_ada` preset.

Example XDG install:

```bash
cd ~/xdg
cmake --preset cubql_llvm_ada \
  -DCMAKE_INSTALL_PREFIX=$HOME/xdg-install/cubql_llvm_ada
cmake --build --preset cubql_llvm_ada --target xdg -j
cmake --install build/cubql_llvm_ada \
  --prefix $HOME/xdg-install/cubql_llvm_ada
```

Then build this application with the standalone LLVM/Ada preset:

```bash
cd ~/xdg-event-particle-sim
git submodule update --init --recursive

export XDG_INSTALL_PREFIX=$HOME/xdg-install/cubql_llvm_ada

cmake --preset llvm_ada
cmake --build --preset llvm_ada
```

The standalone `llvm_ada` preset enables CUDA Thrust sorting and uses LLVM
OpenMP offload flags for an NVIDIA RTX 2000 Ada GPU.

Run example:

```bash
./build/llvm_ada/xdg-particle-sim-event <mesh.h5m> \
  --mesh-library MOAB \
  --rt-library CUBQL \
  --sort-mode volume-direction
```

`--sort-mode` accepts `disabled`, `volume`, `direction`, `volume-direction`, or
`direction-volume`. Compound mode names list the primary key first.

Run the comparable history-based CPU application with Embree:

```bash
OMP_NUM_THREADS=28 \
./build/llvm_ada/xdg-particle-sim-history <mesh.h5m> \
  --mesh-library MOAB \
  --rt-library EMBREE
```

Both executables use the same particle state, random-number generator, source,
unbounded surface-query ordering, collision sampling, and cell track-length
CSV format. Their `--format csv` summaries also share one schema so results can
be combined directly. Use the same `--seed`, `--n-particles`, `--max-events`,
and `--mfp` values for a verification run.

For performance comparisons, use `profile_transport_throughput_rays_per_s` for
both algorithms. The GPU-only `profile_total_ray_throughput_rays_per_s` field
measures the batched ray-trace phase rather than the complete transport loop.

## Plotting Ray Launch Data

The `plot_ray_launches.py` script reads the per-launch CSV written by
`--enable-profiling-ray-launch` and produces basic ray-tracing performance
plots.

Generate ray launch data:

```bash
./build/llvm_ada/xdg-particle-sim-event <mesh.h5m> \
  --mesh-library MOAB \
  --rt-library CUBQL \
  --enable-profiling-ray-launch \
  --ray-launch-profile-output ray-launches.csv
```

Then run:

```bash
python3 plot_ray_launches.py ray-launches.csv
```

## Mean Free Path Sweeps

The `sweep_mean_free_path.py` script runs the particle simulation over a list
of mean free paths, writes one consolidated CSV row per run, and automatically
plots particle outcomes, ray throughput, runtime, and transport workload.

For example:

```bash
python3 sweep_mean_free_path.py run atr.h5m \
  --mean-free-paths 0.1 0.25 0.5 1 2 5 10 \
  --output-directory mfp_sweep \
  --sort-mode volume \
  --profile-ray-launches
```

The main outputs are:

```text
mfp_sweep/mean_free_path_sweep.csv
mfp_sweep/mean_free_path_sweep_summary.png
mfp_sweep/mean_free_path_sweep_particle_states.png
mfp_sweep/mean_free_path_sweep_ray_performance.png
mfp_sweep/mean_free_path_sweep_workload.png
```

With `--profile-ray-launches`, the script also retains a ray-launch CSV and
generates the existing ray-launch summary plot for every mean free path.

Plots can be regenerated without rerunning transport:

```bash
python3 sweep_mean_free_path.py plot mfp_sweep/mean_free_path_sweep.csv
```
