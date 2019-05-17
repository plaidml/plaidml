// Copyright 2018, Intel Corporation

#pragma once

#include <map>
#include <string>

#include "tile/codegen/alias.h"
#include "tile/codegen/codegen.pb.h"
#include "tile/codegen/compile_pass.h"
#include "tile/stripe/stripe.h"

namespace vertexai {
namespace tile {
namespace codegen {

struct RefDefine {
  std::string ref_name;
  stripe::Block* block;
  stripe::StatementIt stmt_iter;
};
typedef std::map<std::string, RefDefine> RefDefineMap;

void Pad(stripe::Block* block, const AliasMap& map, const RefDefineMap& ref_def_map);
void CollectRefDefine(stripe::Block* block, RefDefineMap* ref_def_map);

class PadPass final : public CompilePass {
 public:
  explicit PadPass(const proto::PadPass& options) : options_{options} {}
  void Apply(CompilerState* state) const final;

 private:
  proto::PadPass options_;
};

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
