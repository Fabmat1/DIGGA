cmake_minimum_required(VERSION 3.16)
cmake_policy(SET CMP0074 NEW)       # honour *_ROOT / CMAKE_PREFIX_PATH

project(DIGGA LANGUAGES C CXX Fortran)

if(POLICY CMP0167)
    cmake_policy(SET CMP0167 OLD)
endif()

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
find_package(Boost         REQUIRED CONFIG COMPONENTS filesystem)
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
# 5. Core library (no Python, no matplotlib) – this is what ASTRA links
# ---------------------------------------------------------------------------
add_library(DIGGAcore
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
    src/Resolution.cpp
    src/Chi2Utils.cpp
    src/UnifiedFitWorkflow.cpp
    src/SpectrumLoaders.cpp
    src/DiggaAPI.cpp
    src/DiggaAPI_json.cpp       # JSON adapters (separate TU)
)
add_library(DIGGA::core ALIAS DIGGAcore)

target_include_directories(DIGGAcore
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE extern)

target_compile_definitions(DIGGAcore PUBLIC
    EIGEN_USE_BLAS EIGEN_USE_LAPACKE)

target_link_libraries(DIGGAcore PUBLIC
    Eigen3::Eigen
    OpenBLAS::OpenBLAS
    ${CCFITS_LIB}
    ${CFITSIO_LIB}
    nlohmann_json::nlohmann_json
    Boost::boost Boost::filesystem
    TBB::tbb
    unordered_dense::unordered_dense)

if(OpenMP_CXX_FOUND)
    target_link_libraries(DIGGAcore PUBLIC OpenMP::OpenMP_CXX)
endif()

# ---------------------------------------------------------------------------
# 6. Optional CUDA attach to core
# ---------------------------------------------------------------------------
if(DIGGA_ENABLE_CUDA AND CUDAToolkit_FOUND AND CMAKE_CUDA_COMPILER)
    target_sources(DIGGAcore PRIVATE src/ResolutionCUDA.cu)
    set_source_files_properties(src/ResolutionCUDA.cu PROPERTIES
        LANGUAGE CUDA CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON
        CUDA_SEPARABLE_COMPILATION ON)
    target_compile_options(DIGGAcore PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr
            --diag-suppress=20014 --diag-suppress=20015
            -Wno-deprecated-gpu-targets>)
    target_compile_definitions(DIGGAcore PRIVATE DIGGA_USE_CUDA)
    target_link_libraries(DIGGAcore PRIVATE CUDA::cudart)
endif()

# ---------------------------------------------------------------------------
# 7. Report library (LaTeX + matplotlib-cpp). Optional – ASTRA doesn't need it.
# ---------------------------------------------------------------------------
option(DIGGA_BUILD_REPORT "Build the PDF/plot reporting helpers" ON)
if(DIGGA_BUILD_REPORT)
    add_library(DIGGAreport src/ReportUtils.cpp)
    add_library(DIGGA::report ALIAS DIGGAreport)
    target_include_directories(DIGGAreport PRIVATE extern)
    target_link_libraries(DIGGAreport
        PUBLIC  DIGGAcore
        PRIVATE Python3::Python Python3::NumPy)
    target_compile_definitions(DIGGAreport PRIVATE
        DIGGA_INSTALL_DATADIR=\"${CMAKE_INSTALL_FULL_DATADIR}/DIGGA\")
endif()

# ---------------------------------------------------------------------------
# 8. CLI executable
# ---------------------------------------------------------------------------
add_executable(DIGGA src/main.cpp)
target_include_directories(DIGGA PRIVATE extern)
target_link_libraries(DIGGA PRIVATE DIGGAcore cxxopts::cxxopts)
if(DIGGA_BUILD_REPORT)
    target_link_libraries(DIGGA PRIVATE DIGGAreport Python3::Python)
    target_compile_definitions(DIGGA PRIVATE DIGGA_HAVE_REPORT)
endif()

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/global_settings.json
               ${CMAKE_CURRENT_BINARY_DIR}/global_settings.json COPYONLY)

add_executable(mock_data_generator src/mock_data_generator.cpp)
target_link_libraries(mock_data_generator PRIVATE DIGGAcore cxxopts::cxxopts)

# ---------------------------------------------------------------------------
# 9. Install & export
# ---------------------------------------------------------------------------
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(_digga_targets DIGGAcore)
if(DIGGA_BUILD_REPORT)
    list(APPEND _digga_targets DIGGAreport)
endif()

install(TARGETS ${_digga_targets} DIGGA mock_data_generator
        EXPORT  DIGGATargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# FIX: headers are .hpp, not .h
install(DIRECTORY include/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        FILES_MATCHING PATTERN "*.hpp")

install(FILES global_settings.json DESTINATION ${CMAKE_INSTALL_BINDIR})
if(DIGGA_BUILD_REPORT)
    install(PROGRAMS src/DIGGA_multiplot.py
            DESTINATION ${CMAKE_INSTALL_DATADIR}/DIGGA)
endif()

install(EXPORT DIGGATargets
        FILE       DIGGATargets.cmake
        NAMESPACE  DIGGA::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/DIGGA)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/DIGGAConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/DIGGAConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/DIGGA)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/DIGGAConfigVersion.cmake
    VERSION 0.1.0 COMPATIBILITY SameMajorVersion)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/DIGGAConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/DIGGAConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/DIGGA)
