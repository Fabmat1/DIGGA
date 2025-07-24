# Specfit

Modern C++20 framework to fit observed stellar spectra using synthetic model grids.

Build:
```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j
ctest

# Run:

./specfit_cli --global globals.json --fit run.json [--threads N]
