#pragma once

#include <array>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include <boost/operators.hpp>

#include "base/util/logging.h"
#include "base/util/printstring.h"
#include "base/util/transfer_object.h"
#include "tile/lang/lang.pb.h"
#include "tile/lang/polynomial.h"

namespace vertexai {
namespace tile {
namespace lang {

enum class AggregationOp : char { SUM = '+', MAX = '>', MIN = '<', PROD = '*', ASSIGN = '=', NONE = 'N' };

enum class CombinationOp : char { MULTIPLY = '*', PLUS = '+', EQ = '=', COND = '?', NONE = 'N' };

std::string to_string(const AggregationOp& c);
std::string to_string(const CombinationOp& c);

typedef std::vector<Polynomial> IndexSpec;

class SymbolicPolynomial;
typedef std::shared_ptr<SymbolicPolynomial> SymbolicPolynomialPtr;

struct SymbolicConstraint {
  SymbolicConstraint(const SymbolicPolynomialPtr& _poly, const std::string& _range) : poly(_poly), range(_range) {}
  explicit SymbolicConstraint(const RangeConstraint& _bound) : bound(_bound) {}
  SymbolicPolynomialPtr poly;  // A symbolic representation of the polynominal
  std::string range;           // A symbolic representation of the range
  RangeConstraint bound;       // A concrete final version of the bound
};

std::string to_string(const SymbolicConstraint& c);

inline MAKE_LOGGABLE(SymbolicConstraint, c, os) {
  os << to_string(c);
  return os;
}

typedef std::vector<SymbolicPolynomialPtr> SymbolicSpec;

struct TensorSpec {
  std::string id;      // The name of the tensor
  SymbolicSpec sspec;  // The symbolic polynomial specfications
  IndexSpec spec;      // The concrete final specifications
};

std::string to_string(const TensorSpec& ts, bool as_out = false);

inline MAKE_LOGGABLE(IndexSpec, t, os) {
  os << "[";
  for (size_t i = 0; i < t.size(); i++) {
    os << t[i];
    if (i != t.size() - 1) {
      os << ", ";
    }
  }
  os << "]";
  return os;
}

struct Contraction {
  Contraction() {}
  explicit Contraction(int inputs) : specs(inputs + 1) {}
  CombinationOp comb_op = CombinationOp::MULTIPLY;
  AggregationOp agg_op = AggregationOp::SUM;
  bool no_defract = false;
  std::string use_default;
  std::vector<std::string> output_size;
  // By convention the output of a contraction is always the first spec.
  std::vector<TensorSpec> specs;
  std::vector<SymbolicConstraint> constraints;
  std::set<std::string> getIndexVariables() const;
  std::tuple<std::set<std::string>, std::set<std::string>> getIndexAndOutputVars() const;
};

std::string to_string(const Contraction& cnt);

struct Function {
  std::string fn;
  std::vector<std::string> params;
  bool is_special() const {
    return fn == "gather" || fn == "scatter" || fn == "shape" || (fn.size() > 5 && fn.substr(0, 5) == "prng_");
  }
};

std::string to_string(const proto::Attribute& attr);

struct Op {
  enum { CONTRACTION, FUNCTION, CONSTANT } tag;
  std::string output;
  std::vector<std::string> inputs;
  Contraction c;
  Function f;
  std::vector<proto::Attribute> attributes;
};

std::string to_string(const Op& op);

inline MAKE_LOGGABLE(Op, o, os) {
  os << to_string(o);
  return os;
}

struct Input {
  enum { FIXED, VARIABLE } tag;  // If variable, spec.spec is ignored
  std::string name;
  std::vector<std::string> dims;
};

std::string to_string(const Input& in);

struct Program {
  uint64_t next_tmp = 0;
  std::vector<Input> inputs;
  std::vector<std::string> outputs;
  std::vector<Op> ops;
};

std::string to_string(const Program& prog);

const std::map<std::string, std::string>& BinaryOpMap();

}  // namespace lang
}  // namespace tile
}  // namespace vertexai

TRANSFER_ENUM(vertexai::tile::lang::AggregationOp);
TRANSFER_ENUM(vertexai::tile::lang::CombinationOp);
