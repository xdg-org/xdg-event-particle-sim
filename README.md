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
  --sort-rays-by-volume
```

The `--sort-rays-by-volume` option enables the Thrust queue sort at runtime.

## Plotting Ray Batch Data

The `plot_ray_batches.py` script reads the per-batch CSV written by
`--profile-rays` and produces basic ray-tracing performance plots.

Generate batch data:

```bash
./build/llvm_ada/xdg-particle-sim-event <mesh.h5m> \
  --mesh-library MOAB \
  --rt-library CUBQL \
  --profile-rays \
  --ray-profile-output ray-batches.csv
```

Then run:

```bash
python3 plot_ray_batches.py ray-batches.csv
```