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


────────────────────────────────────────────────────────────────────────────
Platform specific installation notes
────────────────────────────────────────────────────────────────────────────

1. Ubuntu (22.04 LTS, 24.04 LTS or newer)

# Make sure you have an up-to-date system
sudo apt update && sudo apt upgrade

# Mandatory build & runtime dependencies
sudo apt install \
    build-essential gfortran cmake git \
    libopenblas-dev \
    libboost-dev libboost-filesystem-dev libboost-system-dev \
    libcfitsio-dev libccfits-dev \
    libtbb-dev \
    libnlohmann-json3-dev libcxxopts-dev \
    python3-dev python3-numpy \
    libomp-dev                 # OpenMP runtime (clang) – gcc already ships it

# Optional GPU back-end
sudo apt install nvidia-cuda-toolkit        # or the official NVIDIA .run installer

# Eigen ≥ 3.4
Ubuntu 24.04 already ships 3.4.  
If your release still has 3.3.x do this once:

git clone https://gitlab.com/libeigen/eigen.git --branch 3.4.0
cd eigen && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
sudo make install          # installs to /usr/local

# ankerl::unordered_dense (header-only, not in the official repos yet)
git clone https://github.com/martinus/unordered_dense.git
cd unordered_dense && mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local
sudo make install          # again, this is header-only

2. Arch Linux

sudo pacman -Syu                       # full system upgrade
sudo pacman -S \
     base-devel git cmake gcc-fortran \
     openblas \
     boost \
     eigen \
     cfitsio ccfits \
     tbb \
     nlohmann-json \
     cxxopts \
     python python-numpy

# Optional GPU back-end
sudo pacman -S cuda                    # installs CUDA toolkit + drivers

# ankerl::unordered_dense (AUR)
# pick your favourite AUR helper, e.g. yay or paru
yay -S unordered_dense-git             # header-only, installs instantly

3. macOS

# if not done yet: xcode-select --install
# install homebrew, then
brew install cmake git llvm eigen boost openblas cfitsio tbb nlohmann-json cxxopts
# use recent clang
export CC=/opt/homebrew/opt/llvm/bin/clang
export CXX=/opt/homebrew/opt/llvm/bin/clang++
# you may have to add  -DCMAKE_PREFIX_PATH="$(brew --prefix)" as an option for cmake

4. Build Specfit (same on every distro)

git clone https://github.com/<your-user>/specfit.git
cd specfit
mkdir build && cd build
#       GPU on:  -DSPECFIT_ENABLE_CUDA=ON (default)
#       GPU off: -DSPECFIT_ENABLE_CUDA=OFF
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
ctest                                  # optional, runs the unit tests
sudo make install                      # optional, installs lib + CLI

Run:

specfit_cli --global globals.json --fit run.json [--threads N]

────────────────────────────────────────────────────────────────────────────
Troubleshooting

•  “Could NOT find Eigen3 (found version … 3.3.x)”  
   →  Follow the Eigen-from-source instructions above.

•  “Could NOT find unordered_dense”  
   →  Make sure the header is in a location CMake can find, e.g. /usr/local/include/ankerl/unordered_dense.

•  “CUDA toolkit not found”  
   →  Either install CUDA or rebuild with -DSPECFIT_ENABLE_CUDA=OFF.

That’s it—happy fitting!
