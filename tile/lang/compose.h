#pragma once

#include <list>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "base/util/intern.h"
#include "tile/lang/ops.h"
#include "tile/lang/parser.h"
#include "tile/lang/shape.h"
#include "tile/lang/type.h"

namespace vertexai {
namespace tile {
namespace lang {

// A Value represents something which can produce a tensor (IE an actual tensor, a
// computation on tensors, a constant, etc).
class Value : public el::Loggable {
 public:
  enum Type { TENSOR, PLACEHOLDER, FCONST, ICONST, FUNCTION, CONTRACTION };

  Value() {}
  virtual ~Value() {}

  // Get the derived type (without needing to cast)
  virtual Type type() const = 0;
  // How many dimensions does this tensor have (0 = scalar)
  virtual size_t num_dims() const = 0;
  // For each dimension, how big is this tensor (may be a computation)
  virtual std::shared_ptr<Value> dim_value(size_t i) const = 0;

  virtual void log(el::base::type::ostream_t& os) const;  // NOLINT(runtime/references)
};

// This is an abstract base class for whatever underlying buffer concept the user of the system wished to use, however,
// we presume pointer equality on buffers is equivalence on buffers
class BufferBase {
 public:
  virtual ~BufferBase() {}
};

// FConst values represent a floating point constant
class FConstValue final : public Value {
 public:
  static std::shared_ptr<FConstValue> make(const double& value) { return Interned<FConstValue>::make(value); }

  explicit FConstValue(const double& value) : value_{value} {}

  double value() const { return value_; }
  Value::Type type() const final { return Value::Type::FCONST; }
  size_t num_dims() const final { return 0; }
  std::shared_ptr<Value> dim_value(size_t i) const final { throw std::runtime_error("Invalid access of FConst"); }

 private:
  double value_;
};

// IConst values represent an integer constant
class IConstValue final : public Value {
 public:
  static std::shared_ptr<IConstValue> make(const int64_t& val);

  explicit IConstValue(const int64_t& value) : value_{value} {}

  int64_t value() const { return value_; }
  Value::Type type() const final { return Value::Type::ICONST; }
  size_t num_dims() const final { return 0; }
  std::shared_ptr<Value> dim_value(size_t i) const final { throw std::runtime_error("Invalid access of FConst"); }

 private:
  int64_t value_;
};

// Tensor values represent mutable tensors
class TensorValue final : public Value {
 public:
  static std::shared_ptr<TensorValue> make(const std::shared_ptr<BufferBase>& _buffer, const TensorShape& _shape) {
    return Interned<TensorValue>::make(_buffer, _shape);
  }

  TensorValue(const std::shared_ptr<BufferBase>& buffer, const TensorShape& shape) : buffer_{buffer}, shape_{shape} {}

  const std::shared_ptr<BufferBase>& buffer() const { return buffer_; }
  const TensorShape& shape() const { return shape_; }
  Value::Type type() const final { return Value::Type::TENSOR; }
  size_t num_dims() const final { return shape_.dims.size(); }
  std::shared_ptr<Value> dim_value(size_t i) const final {
    return IConstValue::make(static_cast<int64_t>(shape_.dims[i].size));
  }

 private:
  std::shared_ptr<BufferBase> buffer_;
  TensorShape shape_;
};

// Placeholders represent 'variables' to be filled in latter
// NOTE: Placeholders are *NOT* interned
class PlaceholderValue final : public Value {
 public:
  explicit PlaceholderValue(size_t ndims) {
    for (size_t i = 0; i < ndims; i++) {
      dims_.emplace_back(new PlaceholderValue(0));
    }
  }

  Value::Type type() const final { return Value::Type::PLACEHOLDER; }
  size_t num_dims() const final { return dims_.size(); }
  std::shared_ptr<Value> dim_value(size_t i) const final { return dims_[i]; }

 private:
  std::vector<std::shared_ptr<Value>> dims_;
};

// FunctionValues represent a function call that takes multiple tensors
// and returns an output tensor
struct FunctionValue final : public Value {
 public:
  static std::shared_ptr<Value> make(std::string fn, std::vector<std::shared_ptr<Value>> inputs);
  FunctionValue(std::string fn, std::vector<std::shared_ptr<Value>> inputs);

  const std::string& fn() const { return fn_; }
  const std::vector<std::shared_ptr<Value>> inputs() const { return inputs_; }
  Value::Type type() const final { return Value::Type::FUNCTION; }
  size_t num_dims() const final { return dims_.size(); }
  std::shared_ptr<Value> dim_value(size_t i) const final { return dims_[i]; }

 private:
  std::string fn_;
  std::vector<std::shared_ptr<Value>> inputs_;
  std::vector<std::shared_ptr<Value>> dims_;
};

// Special value version of a constraint
struct ValueConstraint {
  bool operator<(const ValueConstraint& rhs) const { return std::tie(poly, range) < std::tie(rhs.poly, rhs.range); }
  SymbolicPolynomialPtr poly;
  std::shared_ptr<Value> range;
};

class ContractionValue final : public Value {
 public:
  static std::shared_ptr<Value> make(CombinationOp comb_op, AggregationOp agg_op,
                                     const std::vector<SymbolicSpec>& specs,
                                     const std::vector<ValueConstraint>& constraints,
                                     const std::vector<std::shared_ptr<Value>>& inputs,
                                     const std::vector<std::shared_ptr<Value>>& dims,
                                     bool use_default,
                                     bool no_defract);

  ContractionValue(CombinationOp comb_op, AggregationOp agg_op, const std::vector<SymbolicSpec>& specs,
                   const std::vector<ValueConstraint>& constraints, const std::vector<std::shared_ptr<Value>>& inputs,
                   const std::vector<std::shared_ptr<Value>>& dims, bool use_default,
                   bool no_defract)
      : comb_op_{comb_op},
        agg_op_{agg_op},
        specs_{specs},
        constraints_{constraints},
        inputs_{inputs},
        dims_{dims},
        use_default_{use_default},
        no_defract_{no_defract} {}

  const CombinationOp& comb_op() const { return comb_op_; }
  const AggregationOp& agg_op() const { return agg_op_; }
  const std::vector<SymbolicSpec>& specs() const { return specs_; }
  const std::vector<ValueConstraint>& constraints() const { return constraints_; }
  const std::vector<std::shared_ptr<Value>>& inputs() const { return inputs_; }
  bool use_default() const { return use_default_; }
  bool no_defract() { return no_defract_; }
  Value::Type type() const final { return Value::Type::CONTRACTION; }
  size_t num_dims() const final { return dims_.size(); }
  std::shared_ptr<Value> dim_value(size_t i) const final { return dims_[i]; }
  size_t logical_input_size() const { return (use_default_ ? inputs_.size() - 1 : inputs_.size()); }

 private:
  CombinationOp comb_op_;
  AggregationOp agg_op_;
  std::vector<SymbolicSpec> specs_;
  std::vector<ValueConstraint> constraints_;
  std::vector<std::shared_ptr<Value>> inputs_;
  std::vector<std::shared_ptr<Value>> dims_;
  bool use_default_;
  bool no_defract_;
};

template <typename T>
class ValueVisitor {
 public:
  virtual ~ValueVisitor() {}

  virtual T Visit(const std::shared_ptr<TensorValue>&) = 0;
  virtual T Visit(const std::shared_ptr<PlaceholderValue>&) = 0;
  virtual T Visit(const std::shared_ptr<FConstValue>&) = 0;
  virtual T Visit(const std::shared_ptr<IConstValue>&) = 0;
  virtual T Visit(const std::shared_ptr<FunctionValue>&) = 0;
  virtual T Visit(const std::shared_ptr<ContractionValue>&) = 0;

  virtual T Apply(const std::shared_ptr<Value>& val) {
    switch (val->type()) {
      case Value::Type::TENSOR:
        return Visit(std::static_pointer_cast<TensorValue>(val));
      case Value::Type::PLACEHOLDER:
        return Visit(std::static_pointer_cast<PlaceholderValue>(val));
      case Value::Type::FCONST:
        return Visit(std::static_pointer_cast<FConstValue>(val));
      case Value::Type::ICONST:
        return Visit(std::static_pointer_cast<IConstValue>(val));
      case Value::Type::FUNCTION:
        return Visit(std::static_pointer_cast<FunctionValue>(val));
      case Value::Type::CONTRACTION:
        return Visit(std::static_pointer_cast<ContractionValue>(val));
    }
    throw std::runtime_error("Unknown type in Visit");
  }
};

struct RunInfo {
  std::string code;
  ShapeMap input_shapes;
  ShapeMap output_shapes;
  std::map<std::string, std::shared_ptr<BufferBase>> input_buffers;
  std::map<std::string, std::shared_ptr<BufferBase>> output_buffers;
};

class FunctionApplication;

class BoundFunction final : public ValueVisitor<std::string> {
  friend class ValuePolynomial;

 public:
  // Make a bound function from a snip
  explicit BoundFunction(const std::string& code, const std::string& id = "");

  // Make a function as a composite
  // Each method below must be called (if needed) in order as many times as needed.
  // IE:  First add all inputs with AddInput, then add any outputs with AddOutput, etc
  BoundFunction() {}
  void AddInput(const std::string& name, const std::shared_ptr<PlaceholderValue>& val);
  void AddOutput(const std::string& name, const std::shared_ptr<Value>& val);
  void AddDependency(const FunctionApplication& prev);
  void AddUpdate(const std::shared_ptr<TensorValue>& lhs, const std::shared_ptr<Value>& rhs);
  void Done();

  // Make a bound function from stored data
  BoundFunction(const Program& code, const std::vector<std::shared_ptr<TensorValue>>& bound_inputs);

  // Accessors
  const Program& prog() const { return prog_; }
  const std::map<std::string, size_t> in_pos() const { return in_pos_; }
  const std::map<std::string, size_t> out_pos() const { return out_pos_; }
  const std::map<std::string, std::shared_ptr<TensorValue>> in_bound() const { return in_bound_; }
  const std::map<std::string, std::shared_ptr<TensorValue>> out_bound() const { return out_bound_; }

  // Get some info about the function
  size_t num_inputs() const { return prog_.inputs.size() - in_bound_.size(); }
  size_t num_outputs() const { return prog_.outputs.size() - out_bound_.size(); }
  const std::string& input_name(size_t i) const { return prog_.inputs[i].name; }
  const std::string& output_name(size_t i) const { return prog_.outputs[i]; }

  // Prepare to run a function, this is only valid if num_inputs() == 0 and num_outputs() == 0
  RunInfo PrepareToRun() const;

 private:
  // Called during construction
  std::string NewTmp() { return std::string("_T") + std::to_string(prog_.next_tmp++); }
  std::string Apply(const std::shared_ptr<Value>& val) final;
  std::string Visit(const std::shared_ptr<TensorValue>& val) final;
  std::string Visit(const std::shared_ptr<PlaceholderValue>& val) final;
  std::string Visit(const std::shared_ptr<FConstValue>& val) final;
  std::string Visit(const std::shared_ptr<IConstValue>& val) final;
  std::string Visit(const std::shared_ptr<FunctionValue>& val) final;
  std::string Visit(const std::shared_ptr<ContractionValue>& val) final;

  // Used during construction
  std::set<std::shared_ptr<TensorValue>> updated_;
  std::map<std::shared_ptr<Value>, std::string> bindings_;

  Program prog_;
  std::map<std::string, size_t> in_pos_;
  std::map<std::string, size_t> out_pos_;
  std::map<std::string, std::shared_ptr<TensorValue>> in_bound_;
  std::map<std::string, std::shared_ptr<TensorValue>> out_bound_;
};

class FunctionApplication {
  friend class LookupPolynomial;

 public:
  explicit FunctionApplication(const std::shared_ptr<BoundFunction>& func);

  const std::list<std::pair<std::shared_ptr<TensorValue>, std::shared_ptr<Value>>> updates() const { return updates_; }
  void AddDependency(const FunctionApplication& prev);
  void SetInput(const std::string& name, const std::shared_ptr<Value>& val);
  void SetDone();
  std::shared_ptr<Value> GetOutput(const std::string& name);
  TensorShape GetOutputShape(const std::string& name);
  bool is_done() const { return is_done_; }

 private:
  bool is_done_ = false;
  std::shared_ptr<BoundFunction> func_;
  std::map<std::string, std::shared_ptr<Value>> bindings_;
  std::list<std::pair<std::shared_ptr<TensorValue>, std::shared_ptr<Value>>> updates_;
  size_t attached_;
  bool is_typechecked_ = false;
  Bindings typecheck_bindings_;
};

// Add X's to bring _ vars back into valid identifiers
Program Xify(const Program& orig);
// Undo an Xify
Program DeXify(const Program& orig);

}  // namespace lang
}  // namespace tile
}  // namespace vertexai
