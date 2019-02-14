// Copyright 2018, Intel Corporation

#pragma once

#include <string>
#include <vector>

#include "tile/codegen/alias.h"
#include "tile/lang/generate.h"
#include "tile/lang/semtree.h"
#include "tile/stripe/stripe.h"

namespace vertexai {
namespace tile {
namespace codegen {

class SemtreeEmitter : public stripe::ConstStmtVisitor {
 public:
  explicit SemtreeEmitter(const AliasMap& am);
  void Visit(const stripe::Load&);
  void Visit(const stripe::Store&);
  void Visit(const stripe::Constant&);
  void Visit(const stripe::Special&);
  void Visit(const stripe::Intrinsic&);
  void Visit(const stripe::Block&);

  std::string vname(const std::string in, size_t depth);
  sem::ExprPtr convert_affine(const stripe::Affine& aff, size_t depth);
  sem::StmtPtr add_loops(const stripe::Block&);
  void do_gids(const stripe::Block&);
  void do_lids(const stripe::Block&);

  size_t depth_;
  std::shared_ptr<sem::Block> cur_;
  std::vector<AliasMap> scopes_;
  const AliasMap* scope_;
  size_t in_kernel_ = 0;
  lang::KernelList kernels_;
};

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
