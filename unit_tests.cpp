#include "gtest/gtest.h"

#if defined(USE_KOKKOS)
#include "Kokkos_Core.hpp"
#define INIT_ARG(X, Y) Kokkos::initialize(X, Y)
#define END_ARG Kokkos::finalize()
#else
#define INIT_ARG(X, Y) [](int& /*unused*/, char** /*unused*/){}
#define END_ARG void()
#endif
int main(int argc, char **argv) {
  INIT_ARG(argc, argv);
  ::testing::InitGoogleTest(&argc, argv);
  auto result = RUN_ALL_TESTS();
  END_ARG;
  return result;
}
