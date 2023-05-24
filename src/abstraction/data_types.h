// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//
// Original implementation of this code by Shreyas Ananthan for AMR-Wind
// - (https://github.com/Exawind/amr-wind)
//
#ifndef DATA_TYPES_H_
#define DATA_TYPES_H_
#include <string>

// TODO this will need to get broken up to more manageable chunks at some point
#if defined(USE_KOKKOS)
//****************************************************
/// USE KOKKOS
//****************************************************
#include "Kokkos_core.hpp"
#define FUNCTION_DECORATOR KOKKOS_FORCEINLINE_FUNCTION
#define DEVICE_LAMBDA KOKKOS_LAMBDA

using defaultType = double;

namespace abstract {
template <typename T> class Scalar {
private:
  using VTYPE = Kokkos::View<T *>;
  VTYPE deviceValue_;
  VTYPE::HostMirror hostValue_;

public:
  Scalar(std::string name, T initValue)
      : deviceValue_(name, 1),
        hostValue_(Kokkos::create_mirror_view(deviceValue_)) {
    hostValue_[0] = initValue;
    copy_host_to_device();
  }

  void copy_host_to_device() { Kokkos::deep_copy(hostValue_, deviceValue_); }

  void copy_device_to_host() { Kokkos::deep_copy(deviceValue_, hostValue_); }

  T* device_data() { return deviceValue_.data(); }

  T host_value() { return hostValue_[0]; }
};

template <typename T> class Vector {
private:
  using VTYPE = Kokkos::View<T *>;
  VTYPE deviceValue_;
  VTYPE::HostMirror hostValue_;

public:
  Vector(std::string name, int size)
      : deviceValue_(name, size),
        hostValue_(Kokkos::create_mirror_view(deviceValue_)) {
  }

  void copy_host_to_device() { Kokkos::deep_copy(hostValue_, deviceValue_); }

  void copy_device_to_host() { Kokkos::deep_copy(deviceValue_, hostValue_); }

  T* device_data() { return deviceValue_.data(); }

  T host_value(int i) { return hostValue_[i]; }
};
using parallel_for = Kokkos::parallel_for;
} // namespace abstract

#elif defined(USE_AMREX)
//****************************************************
/// USE AMREX
//****************************************************
#include "AMReX_Gpu.H"
#define FUNCTION_DECORATOR AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE
#define DEVICE_LAMBDA [=] AMREX_GPU_DEVICE
using defaultType = amrex::Real;
namespace abstract {
using parallel_for = amrex::ParallelFor;
}
#else
//****************************************************
/// USE STL
//****************************************************
#include <vector>
#define FUNCTION_DECORATOR inline
#define DEVICE_LAMBDA [=]
using defaultType = double;
namespace abstract {
template <typename T> class Scalar {
private:
  using VTYPE = std::vector<T>;
  VTYPE deviceValue_;
  VTYPE &hostValue_;

public:
  Scalar(std::string /*name*/, T initValue)
      : deviceValue_(1, initValue), hostValue_(deviceValue_) {}

  void copy_host_to_device() {}

  void copy_device_to_host() {}

  T* device_data(){ return deviceValue_.data(); }

  T host_value() { return hostValue_[0]; }
};
template <typename T> class Vector {
private:
  using VTYPE = std::vector<T>;
  VTYPE deviceValue_;
  VTYPE &hostValue_;

public:
  Vector(std::string /*name*/, int size)
      : deviceValue_(size), hostValue_(deviceValue_) {}

  void copy_host_to_device() {}

  void copy_device_to_host() {}

  T* device_data(){ return deviceValue_.data(); }

  T host_value(int i) { return hostValue_[i]; }
};
// simple parallel for
template <typename T, typename F> inline void parallel_for(T n, F &&f) {
  for (T i = 0; i < n; ++i) {
    f(i);
  }
}

} // namespace abstract
#endif

#endif
