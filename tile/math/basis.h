#pragma once

#include <set>
#include <string>
#include <vector>

#include "tile/math/polynomial.h"

namespace vertexai {
namespace tile {
namespace lang {

class BasisBuilder {
 public:
  // Returns the number of variables in the polynomials
  size_t variables() { return vars_.size(); }
  // Returns the number of dimensions in the basis
  size_t dimensions() { return added_.size(); }
  // Add an equation, returns true if dims went up
  bool addEquation(const Polynomial& p);
  // Return the set of basis vectors used
  const std::vector<Polynomial> basis() const { return added_; }

 private:
  std::vector<Polynomial> added_;    // A set of linearly unique polynomials
  std::vector<Polynomial> reduced_;  // The same as above, but reduced
  std::vector<std::string> vars_;    // All the variables found, in reduction order
  std::set<std::string> vars_set_;   // All the variables found, as a set
};

}  // namespace lang
}  // namespace tile
}  // namespace vertexai
