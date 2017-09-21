#include "tile/lang/ops.h"
#include "tile/lang/sym_poly.h"

#include <cinttypes>
#include <sstream>

#include <boost/math/common_factor_rt.hpp>

namespace vertexai {
namespace tile {
namespace lang {

std::string to_string(const AggregationOp& c) { return std::string({static_cast<char>(c)}); }

std::string to_string(const CombinationOp& c) {
  if (c == CombinationOp::EQ) {
    return "==";
  } else {
    return std::string({static_cast<char>(c)});
  }
}

std::string to_string(const SymbolicConstraint& c) {
  if (c.poly) {
    return "0 <= " + c.poly->ToString() + " < " + c.range;
  } else {
    return to_string(c.bound);
  }
}

std::string to_string(const TensorSpec& ts, bool as_out) {
  std::string r;
  r += ts.id + "[";
  for (size_t i = 0; i < ts.spec.size(); i++) {
    if (i != 0) {
      r += ", ";
    }
    r += to_string(ts.spec[i]);
  }
  for (size_t i = 0; i < ts.sspec.size(); i++) {
    if (i != 0) {
      r += ", ";
    }
    r += ts.sspec[i]->ToString();
  }
  if (!as_out) {
    r += "]";
  }
  return r;
}

std::string to_string(const Contraction& cnt) {
  std::string r = to_string(cnt.specs[0], true);
  if (cnt.specs[0].spec.size() != 0 || cnt.specs[0].sspec.size() != 0) {
    r += " : ";
  }
  for (size_t i = 0; i < cnt.output_size.size(); i++) {
    if (i != 0) {
      r += ", ";
    }
    r += cnt.output_size[i];
  }
  r += "] = ";
  r += to_string(cnt.agg_op) + "(";
  for (size_t i = 1; i < cnt.specs.size(); i++) {
    if (i != 1) {
      if (i == 2 && cnt.comb_op == CombinationOp::COND) {
        r += " == ";
      } else {
        r += " " + to_string(cnt.comb_op) + " ";
      }
    }
    r += to_string(cnt.specs[i]);
  }
  r += ")";
  for (size_t i = 0; i < cnt.constraints.size(); i++) {
    if (cnt.constraints[i].range == "") {
      r += ", " + to_string(cnt.constraints[i].bound.poly) + " < " + std::to_string(cnt.constraints[i].bound.range);
    } else {
      r += ", " + cnt.constraints[i].poly->ToString() + " < " + cnt.constraints[i].range;
    }
  }
  if (cnt.no_defract) {
    r += " no_defract";
  }
  return r;
}

std::string to_string(const Attribute& attr) {
  std::string r = attr.name;
  if (attr.params.size()) {
    r += '(';
    bool first = true;
    for (const auto& param : attr.params) {
      if (!first) {
        r += ", ";
      }
      r += param;
      first = false;
    }
    r += ')';
  }
  return r;
}

std::string to_string(const Op& op) {
  std::string r;

  for (const auto& attr : op.attributes) {
    r += "[[" + to_string(attr) + "]] ";
  }

  switch (op.tag) {
    case Op::CONTRACTION:
      r += to_string(op.c);
      break;

    case Op::CONSTANT:
      r += op.output + " = " + op.inputs[0];
      break;

    case Op::FUNCTION:
      r += op.output + " = " + op.f.fn + "(";
      for (size_t i = 0; i < op.inputs.size(); i++) {
        if (i != 0) {
          r += ", ";
        }
        r += op.inputs[i];
      }
      r += ")";
      break;
  }
  return r;
}

std::string to_string(const Input& in) {
  std::string r;

  switch (in.tag) {
    case Input::VARIABLE:
      r = in.name;
      break;

    case Input::FIXED:
      r = in.name + "[";
      for (size_t i = 0; i < in.dims.size(); i++) {
        if (i != 0) {
          r += ", ";
        }
        r += in.dims[i];
      }
      r += "]";
      break;
  }

  return r;
}

std::string to_string(const Program& prog) {
  std::string r;
  r = "function (\n";
  for (size_t i = 0; i < prog.inputs.size(); i++) {
    r += "  " + to_string(prog.inputs[i]);
    if (i != prog.inputs.size() - 1) {
      r += ",\n";
    }
  }
  r += "\n) -> (\n";
  for (size_t i = 0; i < prog.outputs.size(); i++) {
    r += "  " + prog.outputs[i];
    if (i != prog.outputs.size() - 1) {
      r += ",\n";
    }
  }
  r += "\n) {\n";
  for (size_t i = 0; i < prog.ops.size(); i++) {
    r += "  " + to_string(prog.ops[i]) + ";\n";
  }
  r += "}\n";

  return r;
}

std::set<std::string> Contraction::getIndexVariables() const { return std::get<0>(getIndexAndOutputVars()); }

std::tuple<std::set<std::string>, std::set<std::string>> Contraction::getIndexAndOutputVars() const {
  std::set<std::string> indexVars;
  std::set<std::string> outputVars;
  // Go over all specs
  bool is_first = true;
  for (const auto& tensor : specs) {
    for (const auto& poly : tensor.spec) {
      for (const auto& kvp : poly.getMap()) {
        indexVars.insert(kvp.first);
      }
    }
    if (is_first) {
      outputVars = indexVars;
      is_first = false;
    }
  }
  // Valdiate that no variables appear only in the constraints
  // Pre-add the 'constant' variable (which is always valid)
  indexVars.insert("");
  for (const auto& cons : constraints) {
    for (const auto& kvp : cons.bound.poly.getMap()) {
      // If if there is a variable used in a constraint and that variable doesn't
      // appear in the list of variables from the tensors, then it's a variable
      // that appears only in the constraints, and we need to throw.
      if (indexVars.find(kvp.first) == indexVars.end()) {
        std::ostringstream ErrorStr;
        ErrorStr << "Contraction::getIndexAndOutputVars: Variable '" << kvp.first
                 << "' appears only in constraints of contraction:\n"
                 << " Tensors:";
        for (const auto& tensor : specs) {
          ErrorStr << " {";
          for (const auto& poly : tensor.spec) {
            ErrorStr << poly.toString() << ", ";
          }
          ErrorStr.seekp(-2, ErrorStr.cur);
          ErrorStr << "},";
        }
        ErrorStr.seekp(-1, ErrorStr.cur);
        ErrorStr << "\nConstraints:";
        for (const auto& cons_error : constraints) {
          ErrorStr << " { Poly: " << cons_error.bound.poly.toString();
          ErrorStr << ", Range: " << std::to_string(cons_error.bound.range);
          ErrorStr << ", Var: " << cons_error.range << " }";
        }
        throw std::runtime_error(ErrorStr.str());
      }
    }
  }
  // Erase constant
  indexVars.erase("");
  outputVars.erase("");

  return std::tie(indexVars, outputVars);
}

const std::map<std::string, std::string>& BinaryOpMap() {
  static std::map<std::string, std::string> bin_ops = {
      {"add", "+"},     {"sub", "-"},    {"mul", "*"},     {"div", "/"},       {"cmp_eq", "=="},
      {"cmp_ne", "!="}, {"cmp_lt", "<"}, {"cmp_gt", ">"},  {"cmp_le", "<="},   {"cmp_ge", ">="},
      {"bit_and", "&"}, {"bit_or", "|"}, {"bit_xor", "^"}, {"bit_left", "<<"}, {"bit_right", ">>"},
  };
  return bin_ops;
}

}  // namespace lang
}  // namespace tile
}  // namespace vertexai
