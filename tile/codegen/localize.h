// Copyright 2018, Intel Corporation

#pragma once

#include <set>
#include <string>

#include "tile/codegen/alias.h"
#include "tile/codegen/codegen.pb.h"
#include "tile/codegen/compile_pass.h"
#include "tile/stripe/stripe.h"

namespace vertexai {
namespace tile {
namespace codegen {

// Fixup all downstream reference to var_name (the into name inside of block)
// Specifically, move the location down, as well as the strides
void FixupRefs(stripe::Block* block, const std::string& var_name);

// Make var_name a local and restride to match size
// Also, propagate this on down
void LocalizeRef(stripe::Block* block, const std::string& var_name);

// Localize everything I can, don't update location (for now)
void LocalizeBlockPass(const AliasMap& scope, stripe::Block* block, const std::set<std::string>& ref_reqs = {});

// Localize starting from root for things that match reqs
class LocalizePass final : public CompilePass {
 public:
  explicit LocalizePass(const proto::LocalizePass& options) : options_{options} {}
  void Apply(CompilerState* state) const final;

 private:
  proto::LocalizePass options_;
};

class LocateMemoryPass final : public CompilePass {
 public:
  explicit LocateMemoryPass(const proto::LocateMemoryPass& options) : options_{options} {}
  void Apply(CompilerState* state) const final;

 private:
  proto::LocateMemoryPass options_;
};

class LocateBlockPass final : public CompilePass {
 public:
  explicit LocateBlockPass(const proto::LocateBlockPass& options) : options_{options} {}
  void Apply(CompilerState* state) const final;

 private:
  proto::LocateBlockPass options_;
};

class LocateInnerBlockPass final : public CompilePass {
 public:
  explicit LocateInnerBlockPass(const proto::LocateInnerBlockPass& options) : options_{options} {}
  void Apply(CompilerState* state) const final;

 private:
  proto::LocateInnerBlockPass options_;
};

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
