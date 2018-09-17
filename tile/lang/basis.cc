
#include "tile/lang/basis.h"

#include <utility>

#include <boost/math/common_factor_rt.hpp>

namespace vertexai {
namespace tile {
namespace lang {

bool BasisBuilder::addEquation(const Polynomial& orig) {
  IVLOG(4, "In basis builder, adding poly " << orig);
  // Remove any constants
  Polynomial nc = orig - orig.constant();
  Polynomial p = nc;

  // Reduce the polynomial via existing polynomials
  for (size_t i = 0; i < reduced_.size(); i++) {
    // Get the 'to-be-reduced' ratio
    Rational ratio = p[vars_[i]] / reduced_[i][vars_[i]];
    // Subtract it out
    p -= ratio * reduced_[i];
  }
  IVLOG(4, "Reduced verion:" << p);
  // If the result is 0, equation is linearly dependant on existing basis
  if (p == Polynomial()) {
    return false;
  }

  // Add the original equation minus the constant + the reduced equation
  added_.push_back(nc);
  reduced_.push_back(p);
  // Add in new variables if any
  for (const auto& kvp : p.getMap()) {
    if (vars_set_.count(kvp.first) == 0) {
      vars_set_.insert(kvp.first);
      vars_.push_back(kvp.first);
    }
  }
  // Swap the variable names to make the nth variable nonzero for the nth polynomial
  size_t i = added_.size() - 1;
  for (size_t j = i; j < vars_.size(); j++) {
    if (p[vars_[j]] != 0) {
      std::swap(vars_[i], vars_[j]);
      break;
    }
  }
  // All good
  return true;
}

}  // namespace lang
}  // namespace tile
}  // namespace vertexai
