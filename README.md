# xdg-event-particle-sim

A basic GPU event-based pseudo particle transport simulation application for
mocking neutral-particle transport against CAD models with the XDG library. This
application acts as a mock interface for XDG's GPU API for future particle
transport codes wishing to use XDG for CAD transport. GPU sorting of volumes is 
provided via CUDA thrust or HIP thrust sorting. 

## Building

This application is built against an installed XDG package. Build and install
XDG with cuBQL support first, then point this project at that installation.
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
  --sort-by-volume
```

The `--sort-by-volume` option enables the Thrust queue sort at runtime.

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
  --sort-by-volume \
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
