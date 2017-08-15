#pragma once

#include <map>
#include <string>
#include <vector>

#include "tile/lang/ops.h"

namespace vertexai {
namespace tile {
namespace lang {

// Replace variable names in an op
void ReplaceVariables(Op* op, const std::map<std::string, std::string>& repl);

// Replace variable names in a program
void ReplaceVariables(std::vector<Op>* prog, const std::map<std::string, std::string>& repl);

// Inline a group of functions into a program
void ApplyDefines(Program* prog, const std::map<std::string, Program>& defs);

}  // namespace lang
}  // namespace tile
}  // namespace vertexai
