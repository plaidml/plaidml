// Copyright 2018, Intel Corporation

#pragma once

#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "tile/codegen/alias.h"
#include "tile/lang/generate.h"
#include "tile/lang/intrinsic.h"
#include "tile/lang/semtree.h"
#include "tile/ocl_exec/intrinsic.h"
#include "tile/stripe/stripe.h"

namespace vertexai {
namespace tile {
namespace codegen {

class SemtreeEmitter : public stripe::ConstStmtVisitor {
 public:
  sem::ExprPtr default_intrinsic_emitter(const stripe::Intrinsic& in,
                                         const std::map<std::string, sem::ExprPtr>& exprs = {}) const;
  std::string safe_name(const std::string& in) const;
  std::string ref_buf(const std::string& in) const;
  std::string ref_idx(const std::string& in, int delta = 0) const;
  std::string idx_name(const std::string& in) const;
  std::string scalar_name(const std::string& in) const;
  sem::ExprPtr convert_affine(const stripe::Affine& aff) const;

  explicit SemtreeEmitter(const AliasMap& am, size_t threads);
  void Visit(const stripe::Load&);
  void Visit(const stripe::Store&);
  void Visit(const stripe::LoadIndex&);
  void Visit(const stripe::Constant&);
  void Visit(const stripe::Special&);
  void Visit(const stripe::Intrinsic&);
  void Visit(const stripe::Block&);

  size_t max_threads(const stripe::Block& block);
  sem::StmtPtr add_loops(const stripe::Block&);
  void do_gids(const stripe::Block&);
  sem::StmtPtr make_special(const std::string& name, const stripe::Block& block,
                            std::vector<const stripe::Refinement*> args);
  void compute_thread_count(const stripe::Block& block);
  sem::StmtPtr do_lids(const stripe::Block&);
  void init_loop_local(const std::string& buf, DataType type,  //
                       size_t size, const sem::ExprPtr& init);
  void init_loop_register(const std::string& buf, DataType type,  //
                          size_t size, const sem::ExprPtr& init);

  size_t hw_threads_;
  size_t threads_;
  size_t used_threads_;
  size_t outer_threads_;
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
  const stripe::Block* thread_condition_ = nullptr;
  bool local_var_;
  // Intrinsic list and emitters
  lang::IntrinsicList intrinsics_;
  lang::KernelList kernels_;
};

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
