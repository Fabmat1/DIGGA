cmake_minimum_required(VERSION 3.16)
cmake_policy(SET CMP0074 NEW)       # honour *_ROOT / CMAKE_PREFIX_PATH

project(DIGGA LANGUAGES C CXX Fortran)

# ---------------------------------------------------------------------------
# 0. Tell CMake where Home-brew lives (needed only on macOS)
# ---------------------------------------------------------------------------
if(APPLE)
    foreach(_p "/opt/homebrew" "/usr/local")
        if(EXISTS "${_p}")
            list(APPEND CMAKE_PREFIX_PATH "${_p}")
        endif()
    endforeach()
endif()

# ---------------------------------------------------------------------------
# 1. Build options
# ---------------------------------------------------------------------------
option(DIGGA_ENABLE_CUDA "Build CUDA back-end" ON)

option(DIGGA_LEGACY_CONVOLUTION "Use legacy spectral-resolution algorithm" OFF)

if(DIGGA_LEGACY_CONVOLUTION)
    add_compile_definitions(DIGGA_LEGACY_CONVOLUTION)
endif()

# ---------------------------------------------------------------------------
# 2. C++ settings
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_options(
    -Wno-deprecated-declarations
    -Wno-dev
    -march=native
    -funroll-loops
    -ftree-vectorize
)

# ---------------------------------------------------------------------------
# 3. CUDA (optional)
# ---------------------------------------------------------------------------
if(DIGGA_ENABLE_CUDA)
    enable_language(CUDA OPTIONAL)
endif()
find_package(CUDAToolkit QUIET)

# ---------------------------------------------------------------------------
# 4. Dependencies
# ---------------------------------------------------------------------------
find_package(Eigen3        REQUIRED)
find_package(OpenMP        COMPONENTS C CXX Fortran)   # “REQUIRED” later
find_package(OpenBLAS      REQUIRED)
find_package(Boost         REQUIRED COMPONENTS filesystem)
find_package(nlohmann_json REQUIRED)
find_package(Python3       REQUIRED COMPONENTS Interpreter Development NumPy)
find_package(TBB           REQUIRED)

# ---- 4a. Header-only libraries that may lack a CMake package --------------
find_package(cxxopts QUIET CONFIG)
if(NOT cxxopts_FOUND)
    message(STATUS "cxxoptsConfig.cmake not found – creating IMPORTED target")
    find_path(CXXOPTS_INCLUDE_DIR cxxopts.hpp REQUIRED)
    add_library(cxxopts::cxxopts INTERFACE IMPORTED)
    set_target_properties(cxxopts::cxxopts
        PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${CXXOPTS_INCLUDE_DIR}")
endif()

find_package(unordered_dense QUIET CONFIG)
if(NOT unordered_dense_FOUND)
    message(STATUS "unordered_denseConfig.cmake not found – creating IMPORTED target")
    find_path(UNORD_DENSE_INCLUDE_DIR unordered_dense.h REQUIRED)
    add_library(unordered_dense::unordered_dense INTERFACE IMPORTED)
    set_target_properties(unordered_dense::unordered_dense
        PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${UNORD_DENSE_INCLUDE_DIR}")
endif()

find_library(CCFITS_LIB  CCfits   REQUIRED)
find_library(CFITSIO_LIB cfitsio  REQUIRED)

# ---------------------------------------------------------------------------
# 5. Main library  ----------------------------------------------------------
#     (renamed to DIGGAlib)
# ---------------------------------------------------------------------------
add_library(DIGGAlib
    src/Spectrum.cpp
    src/SpectrumCache.cpp
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
    src/Resolution.cpp            # always built (CPU)
    src/Chi2Utils.cpp
    src/UnifiedFitWorkflow.cpp
    src/ReportUtils.cpp
    src/SpectrumLoaders.cpp
)

target_include_directories(DIGGAlib
    PUBLIC  
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE 
        extern)


# Tell Eigen to delegate to external BLAS / LAPACK
target_compile_definitions(DIGGAlib PUBLIC
    EIGEN_USE_BLAS
    EIGEN_USE_LAPACKE)

target_link_libraries(DIGGAlib PUBLIC
    Eigen3::Eigen
    OpenBLAS::OpenBLAS
    ${CCFITS_LIB}
    ${CFITSIO_LIB}
    nlohmann_json::nlohmann_json
    Boost::boost Boost::filesystem
    Python3::Python Python3::NumPy
    TBB::tbb
    unordered_dense::unordered_dense)

# ---------------------------------------------------------------------------
# 6. CUDA bits (only if toolkit really found) -------------------------------
# ---------------------------------------------------------------------------
if(DIGGA_ENABLE_CUDA AND CUDAToolkit_FOUND AND CMAKE_CUDA_COMPILER)
    message(STATUS "CUDA detected – GPU path enabled")

    target_sources(DIGGAlib PRIVATE src/ResolutionCUDA.cu)

    set_source_files_properties(src/ResolutionCUDA.cu
        PROPERTIES
            LANGUAGE CUDA
            CUDA_STANDARD 20
            CUDA_STANDARD_REQUIRED ON
            CUDA_SEPARABLE_COMPILATION ON)

    target_compile_options(DIGGAlib PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr
                                    --diag-suppress=20014
                                    --diag-suppress=20015
                                    -Wno-deprecated-gpu-targets>)
    target_compile_definitions(DIGGAlib PRIVATE DIGGA_USE_CUDA)
    target_link_libraries(DIGGAlib  PRIVATE CUDA::cudart)
else()
    message(STATUS "CUDA not available – building CPU-only version")
endif()

# ---------------------------------------------------------------------------
# 7. OpenMP (only if compiler + libomp are present) -------------------------
# ---------------------------------------------------------------------------
if(OpenMP_CXX_FOUND)
    target_link_libraries(DIGGAlib PUBLIC OpenMP::OpenMP_CXX)
endif()

# ---------------------------------------------------------------------------
# 8. Command-line executable ------------------------------------------------
#     (keeps the short name DIGGA)
# ---------------------------------------------------------------------------
add_executable(DIGGA src/main.cpp)

# Put required runtime files next to the executable
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/global_settings.json
               ${CMAKE_CURRENT_BINARY_DIR}/global_settings.json COPYONLY)

target_include_directories(DIGGA PRIVATE extern)

target_link_libraries(DIGGA PRIVATE
    DIGGAlib
    cxxopts::cxxopts
    Python3::Python)

# ---------------------------------------------------------------------------
# 9. Install & export -------------------------------------------------------
# ---------------------------------------------------------------------------
include(GNUInstallDirs)

# Helper script
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/src/DIGGA_multiplot.py
               ${CMAKE_CURRENT_BINARY_DIR}/DIGGA_multiplot.py COPYONLY)
install(PROGRAMS src/DIGGA_multiplot.py
        DESTINATION ${CMAKE_INSTALL_DATADIR}/DIGGA)

# Compile-time constant with install location (on the library)
target_compile_definitions(DIGGAlib PRIVATE
    DIGGA_INSTALL_DATADIR=\"${CMAKE_INSTALL_FULL_DATADIR}/DIGGA\")

# Data file
install(FILES global_settings.json
        DESTINATION ${CMAKE_INSTALL_BINDIR})

# Library
install(TARGETS DIGGAlib
        EXPORT  DIGGATargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# Executable
install(TARGETS DIGGA
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

# Public headers
install(DIRECTORY include/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        FILES_MATCHING PATTERN "*.h")

# CMake export file
install(EXPORT DIGGATargets
        FILE       DIGGATargets.cmake
        NAMESPACE  specfit::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/DIGGA)

# ---------------------------------------------------------------------------
# 10. Mock Data Generator executable ----------------------------------------
# ---------------------------------------------------------------------------
add_executable(mock_data_generator src/mock_data_generator.cpp)

target_link_libraries(mock_data_generator PRIVATE
    DIGGAlib
    cxxopts::cxxopts
)

# Copy to build directory
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/global_settings.json
               ${CMAKE_CURRENT_BINARY_DIR}/global_settings.json COPYONLY)

# Installation
install(TARGETS mock_data_generator
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
