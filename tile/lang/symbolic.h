#pragma once

#include <map>
#include <memory>
#include <set>
#include <vector>

#include "tile/lang/compose.h"

namespace vertexai {
namespace tile {
namespace lang {

typedef std::shared_ptr<Value> ValuePtr;
struct UseInfo {
  ValuePtr use;
  size_t idx;
};

extern std::map<ValuePtr, std::set<ValuePtr>> g_deriv_source;

class ComputeUses final : public ValueVisitor<void> {
 public:
  ComputeUses() {}                                           // Initial empty top
  explicit ComputeUses(const ValuePtr& top) { Apply(top); }  // Single top
  void AddTop(const ValuePtr& top) { Apply(top); }
  const std::vector<UseInfo>& uses(const ValuePtr& val) { return uses_[val]; }

 private:
  void Apply(const ValuePtr& val) final;
  void Visit(const std::shared_ptr<TensorValue>&) final {}
  void Visit(const std::shared_ptr<PlaceholderValue>&) final {}
  void Visit(const std::shared_ptr<FConstValue>&) final {}
  void Visit(const std::shared_ptr<IConstValue>&) final {}
  void Visit(const std::shared_ptr<FunctionValue>& val) final;
  void Visit(const std::shared_ptr<ContractionValue>& val) final;

  std::map<ValuePtr, std::vector<UseInfo>> uses_;
  std::set<ValuePtr> done_;
};

class Gradient {
 public:
  Gradient();                                                // No initial source, must be added
  explicit Gradient(const ValuePtr& err);                    // Default initial scalar source
  void AddSource(const ValuePtr& wrt, const ValuePtr& val);  // Add a source
  ValuePtr operator()(const ValuePtr& val);                  // Compute gradients

 private:
  ValuePtr OpGrad(const ValuePtr& dout, const ValuePtr& op, size_t idx);
  ValuePtr FuncOp(const ValuePtr& dout, const std::shared_ptr<FunctionValue>& op, size_t idx);
  ValuePtr SumOp(const ValuePtr& dout, const std::shared_ptr<ContractionValue>& op, size_t idx);
  ValuePtr ExtremeOp(const ValuePtr& dout, const std::shared_ptr<ContractionValue>& op, size_t idx);
  ComputeUses uses_;
  std::map<ValuePtr, ValuePtr> done_;
};

Program ProgGrad(const Program& p);

}  // namespace lang
}  // namespace tile
}  // namespace vertexai
