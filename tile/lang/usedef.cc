#include "tile/lang/usedef.h"

#include <set>
#include <stack>

namespace vertexai {
namespace tile {
namespace lang {

UseDef::UseDef(const Program& prog) {
  for (size_t i = 0; i < prog.inputs.size(); i++) {
    const auto& in = prog.inputs[i];
    if (in_defs_.count(in.name)) {
      throw std::runtime_error("Duplicate input " + in.name);
    }
    in_defs_[in.name] = i;
  }
  for (size_t i = 0; i < prog.ops.size(); i++) {
    if (in_defs_.count(prog.ops[i].output) || op_defs_.count(prog.ops[i].output)) {
      throw std::runtime_error("Variable " + prog.ops[i].output + " redeclared");
    }
    op_defs_[prog.ops[i].output] = i;
    uses_[prog.ops[i].output];  // Initialize an empty set for the output
    if (prog.ops[i].tag == Op::CONSTANT) {
      continue;
    }
    for (const std::string& v : prog.ops[i].inputs) {
      uses_[v].push_back(i);
    }
  }
}

}  // namespace lang
}  // namespace tile
}  // namespace vertexai
