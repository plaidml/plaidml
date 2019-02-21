// Copyright 2018, Intel Corporation

#include "tile/ocl_exec/emitsem.h"
#include "tile/lang/gid.h"
#include "tile/lang/sembuilder.h"

using namespace vertexai::tile::sem::builder;  // NOLINT

namespace vertexai {
namespace tile {
namespace codegen {

#define _C(x) _(vname(x, depth_))
#define _P(x) _(vname(x, depth_ - 1))

SemtreeEmitter::SemtreeEmitter(const AliasMap& am, size_t threads) : threads_(threads) {
  scopes_.emplace_back(am);
  scope_ = &scopes_.back();
}

std::string SemtreeEmitter::vname(const std::string in, size_t depth) {
  // Replace $ with _, add depth
  std::string out = "d" + std::to_string(depth) + "_" + in;
  for (size_t i = 0; i < out.size(); i++) {
    if (out[i] == '$') {
      out[i] = '_';
    }
  }
  return out;
}

sem::ExprPtr SemtreeEmitter::convert_affine(const stripe::Affine& aff, size_t depth) {
  sem::ExprPtr r;
  for (const auto& kvp : aff.getMap()) {
    sem::ExprPtr term = kvp.first.empty() ? sem::ExprPtr(_Const(kvp.second))
                                          : sem::ExprPtr(_Const(kvp.second) * _(vname(kvp.first, depth)));
    r = r ? r + term : term;
  }
  return r ? r : _Const(0);
}

void SemtreeEmitter::Visit(const stripe::Load& stmt) {
  auto type = scope_->at(stmt.from).shape.type;
  cur_->push_back(_Declare({sem::Type::VALUE, type}, vname(stmt.into, depth_), _C(stmt.from)[_Const(0)]));
}

void SemtreeEmitter::Visit(const stripe::Store& stmt) {
  std::string agg_op = scope_->at(stmt.into).base_ref->agg_op;
  if (agg_op == "") {
    cur_->push_back(_C(stmt.into)[_Const(0)] = _C(stmt.from));
    return;
  }
  if (agg_op == "add") {
    cur_->push_back(_C(stmt.into)[_Const(0)] = _C(stmt.into)[_Const(0)] + _C(stmt.from));
    return;
  }
  throw std::runtime_error("Unknown agg-op");
}

void SemtreeEmitter::Visit(const stripe::Constant& stmt) {
  sem::Type type(sem::Type::VALUE);
  sem::ExprPtr expr;
  if (stmt.type == stripe::ConstType::Integer) {
    type.dtype = DataType::INT64;
    expr = std::make_shared<sem::IntConst>(stmt.iconst);
  } else {
    type.dtype = DataType::FLOAT64;
    expr = std::make_shared<sem::FloatConst>(stmt.fconst);
  }
  cur_->push_back(_Declare(type, vname(stmt.name, depth_), expr));
}

void SemtreeEmitter::Visit(const stripe::Special&) {
  // TODO: Something here
}

static std::map<std::string, std::string> bin_ops = {
    {"add", "+"},
    {"mul", "*"},
};

static std::map<std::string, std::string> simple_ops = {
    {"zelu", "relu"},
};

void SemtreeEmitter::Visit(const stripe::Intrinsic& in) {
  auto in_cast = [this, &in](size_t i) { return _Cast({sem::Type::VALUE, in.type}, _(vname(in.inputs[i], depth_))); };
  if (in.inputs.size() == 2 && in.outputs.size() == 1 && bin_ops.count(in.name)) {
    std::string bname = bin_ops.at(in.name);
    cur_->push_back(_Declare({sem::Type::VALUE, in.type}, vname(in.outputs[0], 0),
                             std::make_shared<sem::BinaryExpr>(bname, in_cast(0), in_cast(1))));
    return;
  }
  if (in.inputs.size() == 1 && in.outputs.size() == 1 && simple_ops.count(in.name)) {
    std::string sname = simple_ops.at(in.name);
    return;
  }
  throw std::runtime_error("Unknown intrinsic: " + in.name);
}

sem::StmtPtr SemtreeEmitter::add_loops(const stripe::Block& block) {
  sem::StmtPtr top = cur_;
  for (const auto& idx : block.idxs) {
    if (idx.range != 1) {
      top = _For(vname(idx.name, depth_), idx.range, 1, top);
    } else if (idx.affine != stripe::Affine()) {
      cur_->push_front(_Declare({sem::Type::INDEX}, vname(idx.name, depth_), convert_affine(idx.affine, depth_ - 1)));
    }
  }
  return top;
}

void SemtreeEmitter::do_gids(const stripe::Block& block) {
  std::vector<size_t> logical;
  for (const auto& idx : block.idxs) {
    logical.push_back(idx.range);
  }
  size_t max_dim = 1024 * 1024;
  auto map = lang::gid::MakeMap({max_dim, max_dim, max_dim}, logical);
  std::vector<std::shared_ptr<sem::Expression>> gids;
  for (size_t i = 0; i < 3; i++) {
    gids.push_back(_Index(sem::IndexExpr::GROUP, i));
  }
  for (size_t i = 0; i < block.idxs.size(); i++) {
    const auto& idx = block.idxs[i];
    cur_->push_front(_Declare({sem::Type::INDEX}, vname(idx.name, depth_), LogicalIndex(gids, map.dims[i])));
  }
  kernels_.kernels.emplace_back();
  lang::KernelInfo& ki = kernels_.kernels.back();
  ki.kname = "kernel_" + std::to_string(kernels_.kernels.size());
  ki.comments = block.name + "\n" + block.comments;
  std::vector<sem::Function::param_t> params;
  for (const auto& ref : block.ref_outs()) {
    sem::Type type = {sem::Type::POINTER_MUT, ref->interior_shape.type, 1, 0, sem::Type::GLOBAL};
    params.push_back(std::make_pair(type, vname(ref->from, depth_ - 1)));
    ki.outputs.push_back(ref->from);
  }
  for (const auto& ref : block.ref_ins()) {
    sem::Type type = {sem::Type::POINTER_CONST, ref->interior_shape.type, 1, 0, sem::Type::GLOBAL};
    params.push_back(std::make_pair(type, vname(ref->from, depth_ - 1)));
    ki.inputs.push_back(ref->from);
  }
  ki.kfunc = std::make_shared<sem::Function>(ki.kname, sem::Type(), params, cur_);
  ki.gwork = {map.gid_sizes[0] * threads_, map.gid_sizes[1], map.gid_sizes[2]};
  ki.lwork = {threads_, 1, 1};
  IVLOG(1, "gwork = " << ki.gwork);
  IVLOG(1, "lwork = " << ki.lwork);
}

void SemtreeEmitter::do_lids(const stripe::Block& block) {
  size_t prev_threads = 1;
  for (size_t i = block.idxs.size(); i > 0; i--) {
    const auto& idx = block.idxs[i - 1];
    auto expr = _Index(sem::IndexExpr::LOCAL, 0) / prev_threads;
    if (i != 1) {
      expr = expr % idx.range;
    }
    cur_->push_front(_Declare({sem::Type::INDEX}, vname(idx.name, depth_), expr));
    prev_threads *= idx.range;
  }
}

void SemtreeEmitter::Visit(const stripe::Block& block) {
  std::shared_ptr<sem::Block> outer = cur_;
  if (!in_kernel_ && block.has_tag("kernel")) {
    in_kernel_ = depth_;
  }
  scopes_.emplace_back(*scope_, const_cast<stripe::Block*>(&block));
  scope_ = &scopes_.back();
  // New inner block
  cur_ = std::make_shared<sem::Block>();
  depth_++;
  for (const auto& stmt : block.stmts) {
    stmt->Accept(this);
  }
  depth_--;
  scopes_.pop_back();
  scope_ = &scopes_.back();
  // Now, add the refinement bumps to cur_
  for (const auto& ref : block.refs) {
    if (ref.dir == stripe::RefDir::None) {
      sem::Type ptype = {sem::Type::VALUE, ref.interior_shape.type, 1, ref.interior_shape.elem_size(),
                         sem::Type::LOCAL};
      cur_->push_front(_Declare(ptype, vname(ref.into, depth_), sem::ExprPtr()));
    } else {
      sem::Type ptype = {ref.dir == stripe::RefDir::In ? sem::Type::POINTER_CONST : sem::Type::POINTER_MUT,
                         ref.interior_shape.type};
      cur_->push_front(_Declare(ptype, vname(ref.into, depth_),
                                _(vname(ref.from, depth_ - 1)) + convert_affine(ref.FlatAccess(), depth_)));
    }
  }
  // Add block loops
  sem::StmtPtr top;
  if (in_kernel_ && in_kernel_ == depth_) {
    do_gids(block);
    in_kernel_ = false;
  } else if (block.has_tag("main") || block.has_tag("program")) {
    // No-op
  } else if (block.has_tag("gpu_thread")) {
    do_lids(block);
    outer->push_back(cur_);
  } else {
    outer->push_back(add_loops(block));
  }
  cur_ = outer;
}

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
