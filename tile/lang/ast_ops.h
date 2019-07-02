#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "tile/base/shape.h"
#include "tile/lang/ast.h"
#include "tile/lang/type.h"

namespace vertexai {
namespace tile {
namespace lang {

struct PrimitiveOp {
  virtual ~PrimitiveOp() = default;
  virtual LogicalShape ComputeShape(const std::vector<std::shared_ptr<Expr>>& args) const = 0;
};

class PrimitiveOpRegistry {
 public:
  static PrimitiveOpRegistry* Instance() {
    static PrimitiveOpRegistry registry;
    return &registry;
  }

  void Register(const std::string& name, std::unique_ptr<PrimitiveOp> op) {  //
    registry_[name] = std::move(op);
  }

  const PrimitiveOp* Resolve(const std::string& name) {
    auto it = registry_.find(name);
    if (it == registry_.end()) {
      return nullptr;
    }
    return it->second.get();
  }

 private:
  std::unordered_map<std::string, std::unique_ptr<PrimitiveOp>> registry_;
};

LogicalShape ComputeOutputShape(const std::vector<std::shared_ptr<Expr>>& args);

}  // namespace lang
}  // namespace tile
}  // namespace vertexai
