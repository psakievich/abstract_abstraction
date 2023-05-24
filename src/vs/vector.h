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
// Adapted to use Kokkos

#ifndef VS_VECTOR_H
#define VS_VECTOR_H

#include "abstraction/data_types.h"
#include "vs/vstraits.h"

namespace vs {

/** Vector in 3D space
 */
template <typename T>
struct VectorT
{
  T vv[3]{Traits::zero(), Traits::zero(), Traits::zero()};

  //! Number of components
  static constexpr int ncomp = 3;
  using size_type = int;
  using value_type = T;
  using reference = T&;
  using iterator = T*;
  using const_iterator = const T*;
  using Traits = DTraits<T>;
  using VType = VectorT<T>;

  //! Construct a default vector, all components set to zero
  VectorT() = default;

  /** New vector given the three components
   */
  FUNCTION_DECORATOR
  VectorT(const T& x, const T& y, const T& z) : vv{x, y, z} {}

  ~VectorT() = default;
  VectorT(const VectorT&) = default;
  VectorT(VectorT&&) = default;
  VectorT& operator=(const VectorT&) & = default;
  VectorT& operator=(const VectorT&) && = delete;
  VectorT& operator=(VectorT&&) & = default;
  VectorT& operator=(VectorT&&) && = delete;

  //! Zero vector
  FUNCTION_DECORATOR static constexpr VectorT<T> zero()
  {
    return VectorT<T>{Traits::zero(), Traits::zero(), Traits::zero()};
  }

  FUNCTION_DECORATOR static constexpr VectorT<T> one()
  {
    return VectorT<T>{Traits::one(), Traits::one(), Traits::one()};
  }

  /** Vector along x-axis
   *
   *  \param x Magnitude of vector
   */
  FUNCTION_DECORATOR static constexpr VectorT<T>
  ihat(const T& x = Traits::one())
  {
    return VectorT<T>{x, Traits::zero(), Traits::zero()};
  }

  /** Vector along y-axis
   *
   *  \param y Magnitude of vector
   */
  FUNCTION_DECORATOR static constexpr VectorT<T>
  jhat(const T& y = Traits::one())
  {
    return VectorT<T>{Traits::zero(), y, Traits::zero()};
  }

  /** Vector along z-axis
   *
   *  \param z Magnitude of vector
   */
  FUNCTION_DECORATOR static constexpr VectorT<T>
  khat(const T& z = Traits::one())
  {
    return VectorT<T>{Traits::zero(), Traits::zero(), z};
  }

  //! Normalize the vector to unit vector
  FUNCTION_DECORATOR VectorT<T>& normalize();

  //! Return the unit vector parallel to this vector
  FUNCTION_DECORATOR VectorT<T> unit() const
  {
    return VectorT<T>(*this).normalize();
  }

  FUNCTION_DECORATOR T& x() & noexcept { return vv[0]; }
  FUNCTION_DECORATOR T& y() & noexcept { return vv[1]; }
  FUNCTION_DECORATOR T& z() & noexcept { return vv[2]; }
  FUNCTION_DECORATOR const T& x() const& noexcept { return vv[0]; }
  FUNCTION_DECORATOR const T& y() const& noexcept { return vv[1]; }
  FUNCTION_DECORATOR const T& z() const& noexcept { return vv[2]; }

  FUNCTION_DECORATOR VectorT<T> operator-() const;

  FUNCTION_DECORATOR VectorT<T> operator*=(const T val);

  FUNCTION_DECORATOR VectorT<T> operator/=(const T val);

  FUNCTION_DECORATOR T& operator[](size_type pos) & { return vv[pos]; }
  FUNCTION_DECORATOR const T& operator[](size_type pos) const&
  {
    return vv[pos];
  }

  FUNCTION_DECORATOR T* data() noexcept { return vv; }
  FUNCTION_DECORATOR const T* data() const noexcept { return vv; }

  iterator begin() noexcept { return vv; }
  iterator end() noexcept { return vv + ncomp; }
  const_iterator cbegin() const noexcept { return vv; }
  const_iterator cend() const noexcept { return vv + ncomp; }
  size_type size() const noexcept { return ncomp; }
};

using Vector = VectorT<defaultType>;

} // namespace vs

#include "vs/vectorI.h"

#endif /* VS_VECTOR_H */
