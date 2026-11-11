cmake_minimum_required(VERSION 3.20)
project(lob_bench LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Benchmarks are meaningless without optimisation, so default to Release.
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-Wall -Wextra -Wpedantic)

# Header-only helpers (latency statistics, sample file loader).
add_library(lob INTERFACE)
target_include_directories(lob INTERFACE include)

# --- ONNX Runtime ------------------------------------------------------------
# Uses the official prebuilt release. Pass -DONNXRUNTIME_ROOT=<dir> to use an
# existing copy; otherwise the right archive is downloaded at configure time.
set(ORT_VERSION 1.31.1)

if(NOT ONNXRUNTIME_ROOT)
  if(APPLE AND CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
    set(ORT_PLATFORM osx-arm64)
  elseif(UNIX AND NOT APPLE AND CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
    set(ORT_PLATFORM linux-x64)
  elseif(UNIX AND NOT APPLE AND CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
    set(ORT_PLATFORM linux-aarch64)
  else()
    message(FATAL_ERROR "No prebuilt ONNX Runtime for this platform; set ONNXRUNTIME_ROOT.")
  endif()

  include(FetchContent)
  FetchContent_Declare(onnxruntime
    URL https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-${ORT_PLATFORM}-${ORT_VERSION}.tgz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
  FetchContent_MakeAvailable(onnxruntime)
  set(ONNXRUNTIME_ROOT ${onnxruntime_SOURCE_DIR})
endif()

find_library(ONNXRUNTIME_LIBRARY onnxruntime PATHS ${ONNXRUNTIME_ROOT}/lib NO_DEFAULT_PATH REQUIRED)

add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES
  IMPORTED_LOCATION ${ONNXRUNTIME_LIBRARY}
  INTERFACE_INCLUDE_DIRECTORIES ${ONNXRUNTIME_ROOT}/include)

# --- Targets -----------------------------------------------------------------
add_executable(lob_bench src/bench.cpp)
target_link_libraries(lob_bench PRIVATE lob onnxruntime)

# Integer-only CNN: needs no ONNX Runtime at all.
add_executable(lob_fixed src/fixed_bench.cpp)
target_link_libraries(lob_fixed PRIVATE lob)

enable_testing()
add_executable(lob_tests tests/test_main.cpp)
target_link_libraries(lob_tests PRIVATE lob)
add_test(NAME lob_tests COMMAND lob_tests)
