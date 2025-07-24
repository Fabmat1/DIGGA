cmake_minimum_required(VERSION 3.16)
project(specfit LANGUAGES C CXX Fortran)

# -------------------------------------------------------------------------
# Build options
# -------------------------------------------------------------------------
option(SPECFIT_ENABLE_CUDA "Enable CUDA support" ON)

# -------------------------------------------------------------------------
# C++ settings
# -------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_options(
    -Wno-deprecated-declarations
    -Wno-dev
    -march=native
    -funroll-loops
    -ftree-vectorize
)

# -------------------------------------------------------------------------
# Optional CUDA support
# -------------------------------------------------------------------------
if (SPECFIT_ENABLE_CUDA)
    enable_language(CUDA OPTIONAL)
endif()
find_package(CUDAToolkit QUIET)

# -------------------------------------------------------------------------
# Dependencies
# -------------------------------------------------------------------------
find_package(Eigen3          3.4 REQUIRED)
find_package(OpenMP          REQUIRED COMPONENTS C CXX Fortran)
find_package(OpenBLAS        0.3 REQUIRED)
find_package(Boost           1.72 REQUIRED COMPONENTS system filesystem)
find_package(nlohmann_json   REQUIRED)
find_package(cxxopts         CONFIG REQUIRED)
find_package(Python3         COMPONENTS Interpreter Development NumPy REQUIRED)
find_package(TBB             REQUIRED)
find_package(unordered_dense CONFIG REQUIRED)

find_library(CCFITS_LIB  CCfits)
find_library(CFITSIO_LIB cfitsio)

# -------------------------------------------------------------------------
# Main library
# -------------------------------------------------------------------------
add_library(specfit
    src/Spectrum.cpp
    src/SpectrumCache.cpp
    src/ThreadPool.cpp
    src/NyquistGrid.cpp
    src/Rebin.cpp
    src/AkimaSpline.cpp
    src/RotationalConvolution.cpp
    src/ContinuumModel.cpp
    src/ModelGrid.cpp
    src/SyntheticModel.cpp
    src/FitParameters.cpp
    src/Observation.cpp
    src/JsonUtils.cpp
    src/MultiDatasetCost.cpp
    src/ContinuumUtils.cpp
    src/Resolution.cpp          # CPU part (always built)
    src/Chi2Utils.cpp
    src/UnifiedFitWorkflow.cpp
    src/ReportUtils.cpp
    src/SNRHelpers.cpp
    src/SpectrumLoaders.cpp
)

target_include_directories(specfit PUBLIC include)
target_include_directories(specfit PRIVATE extern)

# Tell Eigen to use external BLAS/LAPACK
target_compile_definitions(specfit PUBLIC
    EIGEN_USE_BLAS
    EIGEN_USE_LAPACKE
)

target_link_libraries(specfit PUBLIC
    Eigen3::Eigen
    OpenBLAS::OpenBLAS
    ${CCFITS_LIB}
    ${CFITSIO_LIB}
    nlohmann_json::nlohmann_json
    Boost::boost
    Boost::system
    Boost::filesystem
    Python3::Python
    Python3::NumPy
    TBB::tbb
    unordered_dense::unordered_dense
)

# -------------------------------------------------------------------------
# Optional CUDA source & flags
# -------------------------------------------------------------------------
if (SPECFIT_ENABLE_CUDA AND CUDAToolkit_FOUND AND CMAKE_CUDA_COMPILER)
    message(STATUS "CUDA detected – enabling GPU back-end for degrade_resolution()")

    target_sources(specfit PRIVATE src/ResolutionCUDA.cu)

    set_source_files_properties(src/ResolutionCUDA.cu
        PROPERTIES
            LANGUAGE CUDA
            CUDA_STANDARD 20
            CUDA_STANDARD_REQUIRED ON
            CUDA_SEPARABLE_COMPILATION ON
    )

    target_compile_options(specfit PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr
                                    --diag-suppress=20014
                                    --diag-suppress=20015
                                    -Wno-deprecated-gpu-targets>
    )

    target_compile_definitions(specfit PRIVATE SPECFIT_USE_CUDA)
    target_link_libraries(specfit PRIVATE CUDA::cudart)
else()
    message(STATUS "CUDA disabled – building CPU-only version.")
endif()

# -------------------------------------------------------------------------
# OpenMP (CPU path)
# -------------------------------------------------------------------------
if (OpenMP_CXX_FOUND)
    target_link_libraries(specfit PUBLIC OpenMP::OpenMP_CXX)
endif()

# -------------------------------------------------------------------------
# Command-line executable
# -------------------------------------------------------------------------
add_executable(specfit_cli src/main.cpp)
target_include_directories(specfit_cli PRIVATE extern)
target_link_libraries(specfit_cli PRIVATE
    specfit
    cxxopts::cxxopts
    Python3::Python
)

# -------------------------------------------------------------------------
# Data paths and install
# -------------------------------------------------------------------------
include(GNUInstallDirs)

# Copy helper script to build directory
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/src/specfit_multiplot.py
               ${CMAKE_CURRENT_BINARY_DIR}/specfit_multiplot.py
               COPYONLY)

# Install helper script
install(PROGRAMS src/specfit_multiplot.py
        DESTINATION ${CMAKE_INSTALL_DATADIR}/specfit)

# Compile-time constant for install location
target_compile_definitions(specfit PRIVATE
    SPECFIT_INSTALL_DATADIR=\"${CMAKE_INSTALL_FULL_DATADIR}/specfit\")

# Install the library
install(TARGETS specfit
        EXPORT specfitTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# Install the CLI executable
install(TARGETS specfit_cli
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

# Install header files
install(DIRECTORY include/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        FILES_MATCHING PATTERN "*.h")

# Export CMake targets
install(EXPORT specfitTargets
        FILE specfitTargets.cmake
        NAMESPACE specfit::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/specfit)
