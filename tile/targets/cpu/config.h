// Copyright 2018, Intel Corp.

#pragma once

#include <functional>
#include <map>
#include <string>
#include <vector>

#include "tile/stripe/stripe.h"

namespace vertexai {
namespace tile {
namespace targets {
namespace cpu {

// The JIT provides builtin implementations for a variety of intrinsics, but
// additional, external handlers may also be provided for context-specific
// services. The JIT will provide the actual datatypes of the input scalars;
//
typedef std::function<void*(std::vector<DataType>*, DataType*)> External;

struct Config {
  bool profile_block_execution = false;
  bool profile_loop_body = false;
  bool print_llvm_ir_simple = VLOG_IS_ON(3);
  bool print_llvm_ir_optimized = VLOG_IS_ON(4);
  bool print_assembly = VLOG_IS_ON(4);
  std::map<std::string, External> externals;
};

}  // namespace cpu
}  // namespace targets
}  // namespace tile
}  // namespace vertexai
