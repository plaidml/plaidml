#pragma once

#include <map>
#include <set>
#include <string>
#include <vector>

#include "tile/base/shape.h"
#include "tile/lang/ops.h"

namespace vertexai {
namespace tile {
namespace lang {

struct Binding {
  explicit Binding(const TensorShape& shape) : tag(TENSOR), shape(shape) {}
  explicit Binding(int64_t iconst) : tag(ICONST), iconst(iconst) { shape.type = DataType::INT32; }
  explicit Binding(double fconst, DataType dtype) : tag(FCONST), fconst(fconst) { shape.type = dtype; }
  explicit Binding(const std::vector<Binding>& tuple) : tag(TUPLE), tuple(tuple) {}

  enum { TENSOR, ICONST, FCONST, TUPLE } tag;
  TensorShape shape;
  int64_t iconst;
  double fconst;
  std::vector<Binding> tuple;

  std::string key() const;
  bool operator==(const Binding& rhs) const;
  bool operator!=(const Binding& rhs) const;
};

inline MAKE_LOGGABLE(Binding, t, os) {
  switch (t.tag) {
    case Binding::TENSOR:
      os << "T:" << t.shape;
      break;
    case Binding::ICONST:
      os << "I:" << t.iconst;
      break;
    case Binding::FCONST:
      os << "F:" << t.fconst;
      break;
    case Binding::TUPLE:
      os << "T:" << t.tuple.size();
  }
  return os;
}

typedef std::map<std::string, Binding> Bindings;

// Compute output from input shapes
// TODO: T383
// ShapeMap ComputeShapes(const std::string& code, const Bindings& inputs)

// Update the initial bindings in var to contain all variables.  Also, inline constant values into program
void TypeCheck(Program* prog, Bindings* vars);

// Do the whole ball of wax
Bindings BindProgram(Program* p, const ShapeMap& inputs, const ShapeMap& outputs);

// Set the default data type for floating-point computations
void SetFloatX(DataType dtype);

// Compute result type by 'upcasting' to the highest type in the hierarchy
DataType CommonSupertype(DataType left, DataType right);

}  // namespace lang
}  // namespace tile
}  // namespace vertexai
