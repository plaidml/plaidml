
// Copyright 2018, Intel Corporation

#pragma once

#include <map>
#include <string>
#include <vector>

#include "tile/stripe/stripe.h"

namespace vertexai {
namespace tile {
namespace codegen {

struct GidxInfo {
  size_t gid_base = 0;  // Which gid
  size_t pre_mod = 0;   // 0 means don't mod
  size_t pre_div = 1;
};

class KernelInfo {
 public:
  explicit KernelInfo(const std::shared_ptr<stripe::Block>& block_ptr);
  std::shared_ptr<stripe::Block> block;
  std::vector<size_t> local_dims;
  std::vector<size_t> group_dims;
  std::map<std::string, GidxInfo> gidx_extract;
  std::string kernel_name;

 private:
  void compute_gidx_packing();
};

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
