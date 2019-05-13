// Copyright 2018, Intel Corporation

#pragma once

#include <string>

#include "tile/codegen/alias.h"
#include "tile/codegen/codegen.pb.h"
#include "tile/codegen/compile_pass.h"
#include "tile/stripe/stripe.h"

namespace vertexai {
namespace tile {
namespace codegen {

void ApplyCache(const AliasMap& map,                                       //
                stripe::Block* block,                                      //
                const std::string& var_name,                               //
                const stripe::Location& mem_loc,                           //
                const stripe::Location& xfer_loc,                          //
                const stripe::Tags load_tags = {"cache", "cache_load"},    //
                const stripe::Tags store_tags = {"cache", "cache_store"},  //
                bool add_constraints = true);

class CachePass final : public CompilePass {
 public:
  explicit CachePass(const proto::CachePass& options) : options_{options} {}
  void Apply(stripe::Block* root) const final;

 private:
  proto::CachePass options_;
};

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
