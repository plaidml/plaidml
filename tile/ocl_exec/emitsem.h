// Copyright 2018, Intel Corporation

#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "tile/codegen/alias.h"
#include "tile/lang/generate.h"
#include "tile/lang/semtree.h"
#include "tile/stripe/stripe.h"

namespace vertexai {
namespace tile {
namespace codegen {

// For (index = init; index < range; index += step)
struct LoopInfo {
  std::string index;
  int init;
  int range;
  int step;
  // stmts to be inserted before the loop
  std::vector<sem::StmtPtr> init_stmts;
  // stmts to be inserted at the end of the loop
  std::vector<sem::StmtPtr> step_stmts;
};

class SemtreeEmitter : public stripe::ConstStmtVisitor {
 public:
  explicit SemtreeEmitter(const AliasMap& am, size_t threads);
  void Visit(const stripe::Load&);
  void Visit(const stripe::Store&);
  void Visit(const stripe::LoadIndex&);
  void Visit(const stripe::Constant&);
  void Visit(const stripe::Special&);
  void Visit(const stripe::Intrinsic&);
  void Visit(const stripe::Block&);

  std::string generate_name(const std::string& prefix);
  std::string safe_name(const std::string& in) const;
  std::string ref_name(const std::string& in) const;
  std::string scalar_name(const std::string& in) const;
  std::string idx_name(const std::string& in) const;
  sem::ExprPtr convert_affine(const stripe::Affine& aff) const;
  void process_affine(const std::string idx, const stripe::Affine& aff);
  sem::StmtPtr add_loops(const stripe::Block&);
  void do_gids(const stripe::Block&);
  sem::StmtPtr do_lids(const stripe::Block&);

  size_t threads_;
  size_t loop_mul_;
  size_t tot_ops_;
  size_t tot_loads_;
  size_t tot_stores_;
  std::vector<size_t> lid_limits_;
  size_t depth_ = 0;
  std::shared_ptr<sem::Block> cur_;
  std::vector<AliasMap> scopes_;
  const AliasMap* scope_;
  size_t in_kernel_ = 0;
  size_t in_threads_ = 0;
  // current outside loops
  std::vector<LoopInfo> loop_info_;
  // the prefix for generated names
  std::map<std::string, size_t> prefix_;
  lang::KernelList kernels_;
};

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
