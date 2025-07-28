cmake_minimum_required(VERSION 3.16)
cmake_policy(SET CMP0074 NEW)       # honour *_ROOT / CMAKE_PREFIX_PATH

project(specfit LANGUAGES C CXX Fortran)

# ---------------------------------------------------------------------------
# 0. Tell CMake where Home-brew lives (needed only on macOS)
# ---------------------------------------------------------------------------
if(APPLE)
    # Intel Macs → /usr/local ;  Apple-Silicon → /opt/homebrew
    foreach(_p "/opt/homebrew" "/usr/local")
        if(EXISTS "${_p}")
            list(APPEND CMAKE_PREFIX_PATH "${_p}")
        endif()
    endforeach()
endif()

# ---------------------------------------------------------------------------
# 1. Build options
# ---------------------------------------------------------------------------
option(SPECFIT_ENABLE_CUDA  "Build CUDA back-end" ON)

# ---------------------------------------------------------------------------
# 2. C++ settings
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    add_compile_options(
        -Wno-deprecated-declarations
        -Wno-dev
        -march=native
        -funroll-loops
        -ftree-vectorize
    )
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(
        -Wno-deprecated-declarations
        -march=native
        -funroll-loops
        -ftree-vectorize
        -fconstexpr-steps=2147483647  # Increase constexpr evaluation steps
    )
endif()

# ---------------------------------------------------------------------------
# 2a. macOS + GCC compatibility fix
# ---------------------------------------------------------------------------
if(APPLE)
    execute_process(COMMAND xcrun --show-sdk-path OUTPUT_VARIABLE MACOS_SDK_PATH OUTPUT_STRIP_TRAILING_WHITESPACE)
    
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        message(WARNING "Using GCC on macOS can cause header compatibility issues. Consider using Clang instead.")
        
        # Force GCC to use the macOS SDK properly
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -isysroot ${MACOS_SDK_PATH} -I${MACOS_SDK_PATH}/usr/include")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -isysroot ${MACOS_SDK_PATH} -I${MACOS_SDK_PATH}/usr/include")
        
        # Add fallback paths
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I/usr/local/include -I/opt/homebrew/include")
        
        # Disable some warnings that GCC triggers with macOS headers
        add_compile_options(-Wno-deprecated-declarations -Wno-unused-command-line-argument)
    else()
        # For Clang, just set the sysroot
        set(CMAKE_OSX_SYSROOT ${MACOS_SDK_PATH})
    endif()
endif()

# ---------------------------------------------------------------------------
# 3. CUDA (optional)
# ---------------------------------------------------------------------------
if(SPECFIT_ENABLE_CUDA)
    enable_language(CUDA OPTIONAL)
endif()
find_package(CUDAToolkit QUIET)

# ---------------------------------------------------------------------------
# 4. Dependencies
# ---------------------------------------------------------------------------
find_package(Eigen3          3.4 REQUIRED)
find_package(OpenMP          COMPONENTS C CXX Fortran)   # "REQUIRED" later

# ---- 4a. OpenBLAS with fallback handling ----------------------------------
find_package(OpenBLAS REQUIRED)

# Create the target if it doesn't exist
if(NOT TARGET OpenBLAS::OpenBLAS)
    if(TARGET openblas)
        # Some installations use lowercase
        add_library(OpenBLAS::OpenBLAS ALIAS openblas)
    elseif(OpenBLAS_LIBRARIES)
        # Fallback: create imported target manually
        add_library(OpenBLAS::OpenBLAS INTERFACE IMPORTED)
        set_target_properties(OpenBLAS::OpenBLAS PROPERTIES
            INTERFACE_LINK_LIBRARIES "${OpenBLAS_LIBRARIES}")
        if(OpenBLAS_INCLUDE_DIRS)
            set_target_properties(OpenBLAS::OpenBLAS PROPERTIES
                INTERFACE_INCLUDE_DIRECTORIES "${OpenBLAS_INCLUDE_DIRS}")
        endif()
    else()
        # Last resort: try pkg-config
        find_package(PkgConfig REQUIRED)
        pkg_check_modules(OpenBLAS REQUIRED openblas)
        add_library(OpenBLAS::OpenBLAS INTERFACE IMPORTED)
        set_target_properties(OpenBLAS::OpenBLAS PROPERTIES
            INTERFACE_LINK_LIBRARIES "${OpenBLAS_LIBRARIES}"
            INTERFACE_INCLUDE_DIRECTORIES "${OpenBLAS_INCLUDE_DIRS}"
            INTERFACE_LINK_DIRECTORIES "${OpenBLAS_LIBRARY_DIRS}")
    endif()
endif()

find_package(Boost           1.72 REQUIRED COMPONENTS system filesystem)
find_package(nlohmann_json   REQUIRED)
find_package(Python3         REQUIRED COMPONENTS Interpreter Development NumPy)
find_package(TBB             REQUIRED)

# ---- 4b. Header-only libraries that may lack a CMake package --------------
# cxxopts
find_package(cxxopts QUIET CONFIG)
if(NOT cxxopts_FOUND)
    message(STATUS "cxxoptsConfig.cmake not found – creating IMPORTED target")
    find_path(CXXOPTS_INCLUDE_DIR cxxopts.hpp REQUIRED)
    add_library(cxxopts::cxxopts INTERFACE IMPORTED)
    set_target_properties(cxxopts::cxxopts
        PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${CXXOPTS_INCLUDE_DIR}")
endif()

# unordered_dense
find_package(unordered_dense QUIET CONFIG)
if(NOT unordered_dense_FOUND)
    message(STATUS "unordered_denseConfig.cmake not found – creating IMPORTED target")
    find_path(UNORD_DENSE_INCLUDE_DIR unordered_dense.h REQUIRED)
    add_library(unordered_dense::unordered_dense INTERFACE IMPORTED)
    set_target_properties(unordered_dense::unordered_dense
        PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${UNORD_DENSE_INCLUDE_DIR}")
endif()

# CFITSIO / CCfits (plain libraries, no CMake package shipped)
find_library(CCFITS_LIB  CCfits    REQUIRED)
find_library(CFITSIO_LIB cfitsio   REQUIRED)

# ---------------------------------------------------------------------------
# 5. Main library
# ---------------------------------------------------------------------------
add_library(specfit
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
    src/SNRHelpers.cpp
    src/SpectrumLoaders.cpp
)

# Fix for include directories error
target_include_directories(specfit
    PUBLIC  
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE 
        ${CMAKE_CURRENT_SOURCE_DIR}/extern
)

# Tell Eigen to delegate to external BLAS / LAPACK
target_compile_definitions(specfit PUBLIC
    EIGEN_USE_BLAS
    EIGEN_USE_LAPACKE)

target_link_libraries(specfit PUBLIC
    Eigen3::Eigen
    OpenBLAS::OpenBLAS
    ${CCFITS_LIB}
    ${CFITSIO_LIB}
    nlohmann_json::nlohmann_json
    Boost::boost Boost::system Boost::filesystem
    Python3::Python Python3::NumPy
    TBB::tbb
    unordered_dense::unordered_dense)

# ---------------------------------------------------------------------------
# 6. CUDA source & flags (only if toolkit really found)
# ---------------------------------------------------------------------------
if(SPECFIT_ENABLE_CUDA AND CUDAToolkit_FOUND AND CMAKE_CUDA_COMPILER)
    message(STATUS "CUDA detected – GPU path enabled")

    target_sources(specfit PRIVATE src/ResolutionCUDA.cu)

    set_source_files_properties(src/ResolutionCUDA.cu
        PROPERTIES
            LANGUAGE CUDA
            CUDA_STANDARD 20
            CUDA_STANDARD_REQUIRED ON
            CUDA_SEPARABLE_COMPILATION ON)

    target_compile_options(specfit PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr
                                    --diag-suppress=20014
                                    --diag-suppress=20015
                                    -Wno-deprecated-gpu-targets>)

    target_compile_definitions(specfit PRIVATE SPECFIT_USE_CUDA)
    target_link_libraries(specfit PRIVATE CUDA::cudart)
else()
    message(STATUS "CUDA not available – building CPU-only version")
endif()

# ---------------------------------------------------------------------------
# 7. OpenMP (only if compiler + libomp are really present)
# ---------------------------------------------------------------------------
if(OpenMP_CXX_FOUND)
    target_link_libraries(specfit PUBLIC OpenMP::OpenMP_CXX)
endif()

# ---------------------------------------------------------------------------
# 8. Command-line executable
# ---------------------------------------------------------------------------
add_executable(specfit_cli src/main.cpp)

# Copy global_settings.json to build directory next to executable
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/global_settings.json
               ${CMAKE_CURRENT_BINARY_DIR}/global_settings.json COPYONLY)

target_include_directories(specfit_cli PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/extern)
target_link_libraries(specfit_cli PRIVATE
    specfit
    cxxopts::cxxopts
    Python3::Python)

# ---------------------------------------------------------------------------
# 9. Install & export
# ---------------------------------------------------------------------------
include(GNUInstallDirs)

# Helper script
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/src/specfit_multiplot.py
               ${CMAKE_CURRENT_BINARY_DIR}/specfit_multiplot.py COPYONLY)
install(PROGRAMS src/specfit_multiplot.py
        DESTINATION ${CMAKE_INSTALL_DATADIR}/specfit)

# Compile-time constant with install location
target_compile_definitions(specfit PRIVATE
    SPECFIT_INSTALL_DATADIR=\"${CMAKE_INSTALL_FULL_DATADIR}/specfit\")

# Add this to your install section (section 9)
install(FILES global_settings.json
        DESTINATION ${CMAKE_INSTALL_BINDIR})

install(TARGETS specfit
        EXPORT  specfitTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(TARGETS specfit_cli
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

install(DIRECTORY include/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        FILES_MATCHING PATTERN "*.h")

install(EXPORT specfitTargets
        FILE       specfitTargets.cmake
        NAMESPACE  specfit::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/specfit)
