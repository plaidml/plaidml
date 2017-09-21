#pragma once

#include <map>
#include <set>
#include <string>
#include <vector>

#include "tile/lang/ops.h"

namespace vertexai {
namespace tile {
namespace lang {

class UseDef {
 public:
  // Construct use-definition chains for a program
  explicit UseDef(const Program& prog);

  // Given a program op #, compute all ops connected by simple functions
  std::set<size_t> ConnectedComponents(const Program& prog, size_t start, const std::set<size_t>& previously_computed);

  const std::map<std::string, size_t>& op_defs() const { return op_defs_; }
  const std::map<std::string, std::vector<size_t>>& uses() const { return uses_; }

 private:
  // For each identifier, the input # where it is declared (if input)
  std::map<std::string, size_t> in_defs_;
  // For each identifier, the op # where it is declared (if an op)
  std::map<std::string, size_t> op_defs_;
  // Every op that uses the identifier as an input
  std::map<std::string, std::vector<size_t>> uses_;
};

}  // namespace lang
}  // namespace tile
}  // namespace vertexai
