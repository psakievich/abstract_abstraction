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

#ifndef TRIG_OPS_H
#define TRIG_OPS_H

/** \file trig_ops.h
 *
 *  Trigonometric utilities
 */

#include <cmath>
#include "data_types.h"

namespace utils {

//! Return \f$\pi\f$ as an double
FUNCTION_DECORATOR constexpr double
pi()
{
  return M_PI;
}

//! Return \f$2 \pi\f$
FUNCTION_DECORATOR constexpr double
two_pi()
{
  return 2.0 * M_PI;
}

//! Return \f$\pi / 2\f$
FUNCTION_DECORATOR constexpr double
half_pi()
{
  return 0.5 * M_PI;
}

//! Convert from degrees to radians
FUNCTION_DECORATOR double
radians(const double deg_val)
{
  return pi() * deg_val / 180.0;
}

//! Convert from radians to degrees
FUNCTION_DECORATOR double
degrees(const double rad_val)
{
  return 180.0 * rad_val / pi();
}

} // namespace utils

#endif /* TRIG_OPS_H */
