cmake_minimum_required(VERSION 3.0)
project(3DMinRegLineIntersect)

# Find requires packages
# This project requires Eigen
find_package(Eigen3 REQUIRED)

include_directories(include ${EIGEN3_INCLUDE_DIR})

## Libraries
add_library(UTILS src/utils.cpp)
add_library(SOLVERS src/solvers.cpp)
add_library(SOLVERSLINES src/solverLines.cpp)
add_library(SOLVERSPOINTS src/solverPoints.cpp)

## Executables
# single solvers runs
add_executable(run1L1Q1P examples/run1L1Q1P.cpp)
add_executable(run3L1Q examples/run3L1Q.cpp)
add_executable(run6L examples/run6L.cpp)
add_executable(run3Q examples/run3Q.cpp)
add_executable(run2M examples/run2M.cpp)

target_link_libraries(run1L1Q1P Eigen3::Eigen SOLVERS UTILS)
target_link_libraries(run3L1Q Eigen3::Eigen SOLVERS UTILS)
target_link_libraries(run6L Eigen3::Eigen SOLVERSLINES UTILS)
target_link_libraries(run3Q Eigen3::Eigen SOLVERSPOINTS UTILS)
target_link_libraries(run2M Eigen3::Eigen SOLVERS UTILS)