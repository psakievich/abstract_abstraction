cmake_minimum_required(VERSION 3.20.0)
project(proof_of_concept LANGUAGES CXX)

# proof of concept library definition
add_library(poc INTERFACE src/vs)
target_include_directories(
  poc
  INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
# not really required yet, but placing there so I don't forget
target_compile_features(poc INTERFACE cxx_std_17)
option(ENABLE_AMREX_TESTS "Build unit-tests against AMREX" OFF)
option(ENABLE_KOKKOS_TESTS "Build unit-tests against Kokkos" OFF)

# stuff for unit testing executables
find_package(GTest REQUIRED)

#stl unit test
add_executable(test_stl ${CMAKE_CURRENT_SOURCE_DIR}/unit_tests.cpp ${CMAKE_CURRENT_SOURCE_DIR}/test/test_vector_space.cpp)
target_link_libraries(test_stl GTest::gtest GTest::gtest_main)
target_include_directories(
  test_stl
  PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

# kokkos unit test
if(ENABLE_KOKKOS_TESTS)
  find_package(Kokkos REQUIRED)
  add_executable(test_kokkos ${CMAKE_CURRENT_SOURCE_DIR}/unit_tests.cpp ${CMAKE_CURRENT_SOURCE_DIR}/test/test_vector_space.cpp)
  target_link_libraries(test_kokkos GTest::gtest GTest::gtest_main Kokkos::kokkos)
  target_compile_definitions(test_kokkos PUBLIC USE_KOKKOS)
  target_include_directories(
    test_kokkos
    PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
endif()

if(ENABLE_AMREX_TESTS)
  find_package(AMReX REQUIRED)
  add_executable(test_amrex ${CMAKE_CURRENT_SOURCE_DIR}/unit_tests.cpp ${CMAKE_CURRENT_SOURCE_DIR}/test/test_vector_space.cpp)
  target_link_libraries(test_amrex GTest::gtest GTest::gtest_main AMReX::amrex)
  target_compile_definitions(test_amrex PUBLIC USE_AMREX)
  target_include_directories(
    test_amrex
    PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
endif()
