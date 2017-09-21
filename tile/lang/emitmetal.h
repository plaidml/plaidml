
#pragma once

#include "tile/lang/emitc.h"
#include <sstream>
#include <string>

namespace vertexai {
namespace tile {
namespace lang {

class EmitMetal : public EmitC {
 public:
  void Visit(const sem::CallExpr &) override;
  void Visit(const sem::IndexExpr &) override;
  void Visit(const sem::BarrierStmt &) override;
  void Visit(const sem::Function &) override;

 private:
  void emitType(const sem::Type &t) override;
};

}  // namespace lang
}  // namespace tile
}  // namespace vertexai
