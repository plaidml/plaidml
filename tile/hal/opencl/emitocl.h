// Copyright 2017, Vertex.AI. CONFIDENTIAL

#pragma once

#include <sstream>
#include <string>

#include "tile/lang/emitc.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace opencl {

class Emit : public lang::EmitC {
 public:
  void Visit(const sem::CallExpr &) final;
  void Visit(const sem::IndexExpr &) final;
  void Visit(const sem::BarrierStmt &) final;
  void Visit(const sem::Function &) final;

 private:
  void emitType(const sem::Type &t) final;
};

}  // namespace opencl
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
