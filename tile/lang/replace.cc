#include "tile/lang/replace.h"

#include "tile/lang/parser.h"

namespace vertexai {
namespace tile {
namespace lang {

void ReplaceVariables(Op* op, const std::map<std::string, std::string>& repl) {
  auto oit = repl.find(op->output);
  if (oit != repl.end()) {
    op->output = oit->second;
  }
  for (std::string& s : op->inputs) {
    auto it = repl.find(s);
    if (it != repl.end()) {
      s = it->second;
    }
  }
  for (TensorSpec& ts : op->c.specs) {
    auto it = repl.find(ts.id);
    if (it != repl.end()) {
      ts.id = it->second;
    }
  }
  for (std::string& s : op->c.output_size) {
    auto it = repl.find(s);
    if (it != repl.end()) {
      s = it->second;
    }
  }
}

void ReplaceVariables(std::vector<Op>* prog, const std::map<std::string, std::string>& repl) {
  for (Op& op : *prog) {
    ReplaceVariables(&op, repl);
  }
}

void ApplyDefines(Program* prog, const std::map<std::string, Program>& defs) {
  std::vector<Op> out;
  for (const Op& op : prog->ops) {
    if (op.tag != Op::FUNCTION) {
      out.push_back(op);
      continue;
    }
    auto it = defs.find(op.f.fn);
    if (it == defs.end()) {
      out.push_back(op);
      continue;
    }
    Program new_ops = it->second;
    std::map<std::string, std::string> repl;
    for (size_t i = 0; i < op.inputs.size(); i++) {
      repl["X" + std::to_string(i + 1)] = op.inputs[i];
    }
    for (Op& op2 : new_ops.ops) {
      repl[op2.output] = "_T" + std::to_string(prog->next_tmp++);
      ReplaceVariables(&op2, repl);
    }
    repl.clear();
    repl[new_ops.ops.back().output] = op.output;
    ReplaceVariables(&new_ops.ops.back(), repl);
    out.insert(out.end(), new_ops.ops.begin(), new_ops.ops.end());
  }
  prog->ops = out;
}

}  // namespace lang
}  // namespace tile
}  // namespace vertexai
