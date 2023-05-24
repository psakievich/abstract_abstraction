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

#ifndef VS_TENSOR_H
#define VS_TENSOR_H

#include "data_types.h"
#include "vstraits.h"
#include "vector.h"

namespace vs {

/** Tensor in 3D vector space
 */
template <typename T>
struct TensorT
{
  T vv[9]{Traits::zero(), Traits::zero(), Traits::zero(),
          Traits::zero(), Traits::zero(), Traits::zero(),
          Traits::zero(), Traits::zero(), Traits::zero()};

  static constexpr int ncomp = 9;
  using size_type = int;
  using value_type = T;
  using reference = T&;
  using iterator = T*;
  using const_iterator = const T*;
  using Traits = DTraits<T>;

  constexpr TensorT() = default;

  FUNCTION_DECORATOR constexpr TensorT(
    const T& xx,
    const T& xy,
    const T& xz,
    const T& yx,
    const T& yy,
    const T& yz,
    const T& zx,
    const T& zy,
    const T& zz)
    : vv{xx, xy, xz, yx, yy, yz, zx, zy, zz}
  {
  }

  FUNCTION_DECORATOR TensorT(
    const VectorT<T>& x,
    const VectorT<T>& y,
    const VectorT<T>& z,
    const bool transpose = false);

  ~TensorT() = default;
  TensorT(const TensorT&) = default;
  TensorT(TensorT&&) = default;
  TensorT& operator=(const TensorT&) & = default;
  TensorT& operator=(const TensorT&) && = delete;
  TensorT& operator=(TensorT&&) & = default;
  TensorT& operator=(TensorT&&) && = delete;

  FUNCTION_DECORATOR static constexpr TensorT<T> zero() noexcept
  {
    return TensorT<T>{Traits::zero(), Traits::zero(), Traits::zero(),
                      Traits::zero(), Traits::zero(), Traits::zero(),
                      Traits::zero(), Traits::zero(), Traits::zero()};
  }

  FUNCTION_DECORATOR static constexpr TensorT<T> I() noexcept
  {
    return TensorT{Traits::one(),  Traits::zero(), Traits::zero(),
                   Traits::zero(), Traits::one(),  Traits::zero(),
                   Traits::zero(), Traits::zero(), Traits::one()};
  }

  FUNCTION_DECORATOR void
  rows(const VectorT<T>& x, const VectorT<T>& y, const VectorT<T>& z) noexcept;
  FUNCTION_DECORATOR void
  cols(const VectorT<T>& x, const VectorT<T>& y, const VectorT<T>& z) noexcept;

  FUNCTION_DECORATOR VectorT<T> x() const noexcept;
  FUNCTION_DECORATOR VectorT<T> y() const noexcept;
  FUNCTION_DECORATOR VectorT<T> z() const noexcept;

  FUNCTION_DECORATOR VectorT<T> cx() const noexcept;
  FUNCTION_DECORATOR VectorT<T> cy() const noexcept;
  FUNCTION_DECORATOR VectorT<T> cz() const noexcept;

  FUNCTION_DECORATOR T& xx() & noexcept { return vv[0]; }
  FUNCTION_DECORATOR T& xy() & noexcept { return vv[1]; }
  FUNCTION_DECORATOR T& xz() & noexcept { return vv[2]; }

  FUNCTION_DECORATOR T& yx() & noexcept { return vv[3]; }
  FUNCTION_DECORATOR T& yy() & noexcept { return vv[4]; }
  FUNCTION_DECORATOR T& yz() & noexcept { return vv[5]; }

  FUNCTION_DECORATOR T& zx() & noexcept { return vv[6]; }
  FUNCTION_DECORATOR T& zy() & noexcept { return vv[7]; }
  FUNCTION_DECORATOR T& zz() & noexcept { return vv[8]; }

  FUNCTION_DECORATOR const T& xx() const& noexcept { return vv[0]; }
  FUNCTION_DECORATOR const T& xy() const& noexcept { return vv[1]; }
  FUNCTION_DECORATOR const T& xz() const& noexcept { return vv[2]; }

  FUNCTION_DECORATOR const T& yx() const& noexcept { return vv[3]; }
  FUNCTION_DECORATOR const T& yy() const& noexcept { return vv[4]; }
  FUNCTION_DECORATOR const T& yz() const& noexcept { return vv[5]; }

  FUNCTION_DECORATOR const T& zx() const& noexcept { return vv[6]; }
  FUNCTION_DECORATOR const T& zy() const& noexcept { return vv[7]; }
  FUNCTION_DECORATOR const T& zz() const& noexcept { return vv[8]; }

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

using Tensor = TensorT<defaultType>;

} // namespace vs

#include "tensorI.h"

#endif /* VS_TENSOR_H */
