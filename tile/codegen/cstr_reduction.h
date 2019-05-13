// Copyright 2018, Intel Corporation

#pragma once

#include <map>
#include <set>
#include <vector>

#include "tile/codegen/alias.h"
#include "tile/codegen/codegen.pb.h"
#include "tile/codegen/compile_pass.h"
#include "tile/stripe/stripe.h"

namespace vertexai {
namespace tile {
namespace codegen {

void LightCstrReduction(const AliasMap& alias_map, stripe::Block* block,
                        const proto::LightConstraintReductionPass& options);

void IlpCstrReduction(const AliasMap& alias_map, stripe::Block* block,
                      const proto::IlpConstraintReductionPass& options);

class LightCstrReductionPass final : public CompilePass {
 public:
  explicit LightCstrReductionPass(const proto::LightConstraintReductionPass& options) : options_{options} {}
  void Apply(stripe::Block* root) const final;

 private:
  proto::LightConstraintReductionPass options_;
};

void IlpCstrReduction(const AliasMap& alias_map, stripe::Block* block,
                      const proto::IlpConstraintReductionPass& options);

class IlpCstrReductionPass final : public CompilePass {
 public:
  explicit IlpCstrReductionPass(const proto::IlpConstraintReductionPass& options) : options_{options} {}
  void Apply(stripe::Block* root) const final;

 private:
  proto::IlpConstraintReductionPass options_;
};

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
