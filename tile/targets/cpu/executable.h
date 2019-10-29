// Copyright 2019, Intel Corp.

#pragma once

#include <llvm/ExecutionEngine/ExecutionEngine.h>

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "tile/stripe/stripe.h"
#include "tile/targets/cpu/programmodule.h"

// For now disable TBB parallelizm in odrer to investigate potential
// concurency (corectness) problems in XSMM.
#define PLAID_USE_TBB_PARALLELISM 0

namespace vertexai {
namespace tile {
namespace targets {
namespace cpu {

enum ParallelCallType {
  NONE = 0,
  XSMM,
};

class Executable {
 public:
  explicit Executable(const ProgramModule& module);
  void Run(const std::map<std::string, void*>& buffers);
  void Save(const std::string& filename);
  void SetPerfAttrs(stripe::Block* block);

 private:
  std::unique_ptr<llvm::ExecutionEngine> engine_;
  std::vector<std::string> parameters_;
};

}  // namespace cpu
}  // namespace targets
}  // namespace tile
}  // namespace vertexai
