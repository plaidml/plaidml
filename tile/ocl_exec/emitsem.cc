// Copyright 2018, Intel Corporation

#include "tile/ocl_exec/emitsem.h"
#include "tile/lang/gen_special.h"
#include "tile/lang/gen_trivial.h"
#include "tile/lang/gid.h"
#include "tile/lang/ops.h"
#include "tile/lang/sembuilder.h"

using namespace vertexai::tile::sem::builder;  // NOLINT

namespace vertexai {
namespace tile {
namespace codegen {

SemtreeEmitter::SemtreeEmitter(const AliasMap& am, size_t threads)
    : threads_(threads), lid_limits_({1024 * 1024, 1024 * 1024, 1024 * 1024}) {
  scopes_.emplace_back(am);
  scope_ = &scopes_.back();
}

std::string SemtreeEmitter::safe_name(const std::string& in) const {
  std::string out = in;
  for (size_t i = 0; i < out.size(); i++) {
    if (out[i] == ':') {
      out[i] = '_';
    }
  }
  return out;
}

std::string SemtreeEmitter::ref_name(const std::string& in) const { return safe_name(scope_->at(in).base_name); }

std::string SemtreeEmitter::scalar_name(const std::string& in) const {
  if (in.size() < 1 || in[0] != '$') {
    throw std::runtime_error("SemtreeEmitter, invalid scalar name");
  }
  return "s_" + std::to_string(depth_) + "_" + in.substr(1, in.size() - 1);
}

std::string SemtreeEmitter::idx_name(const std::string& in) const {
  return safe_name(scope_->idx_sources().at(in).getMap().begin()->first);
}

sem::ExprPtr SemtreeEmitter::convert_affine(const stripe::Affine& aff) const {
  sem::ExprPtr r;
  for (const auto& kvp : aff.getMap()) {
    sem::ExprPtr term = kvp.first.empty() ? sem::ExprPtr(_Const(kvp.second))
                                          : sem::ExprPtr(_Const(kvp.second) * _(safe_name(kvp.first)));
    r = r ? r + term : term;
  }
  return r ? r : _Const(0);
}

void SemtreeEmitter::Visit(const stripe::Load& stmt) {
  auto ai = scope_->at(stmt.from);
  auto lval = _(ref_name(stmt.from))[convert_affine(ai.flat())];
  auto type = ai.shape.type;
  cur_->push_back(_Declare({sem::Type::VALUE, type}, scalar_name(stmt.into), lval));
}

static sem::ExprPtr AggInit(DataType type, const std::string agg_op) {
  if (agg_op == "" || agg_op == "assign") {
    return sem::ExprPtr();
  } else if (agg_op == "add") {
    return _LimitConst(sem::LimitConst::ZERO, type);
  } else if (agg_op == "max") {
    return _LimitConst(sem::LimitConst::MIN, type);
  } else if (agg_op == "min") {
    return _LimitConst(sem::LimitConst::MAX, type);
  } else if (agg_op == "mul") {
    return _LimitConst(sem::LimitConst::ONE, type);
  }
  throw std::runtime_error("Unknown agg-op:" + agg_op);
}

void SemtreeEmitter::Visit(const stripe::Store& stmt) {
  auto ai = scope_->at(stmt.into);
  auto lval = _(ref_name(stmt.into))[convert_affine(ai.flat())];
  std::string agg_op = ai.base_ref->agg_op;
  auto rval = _(scalar_name(stmt.from));
  sem::ExprPtr agg;
  if (agg_op == "" || agg_op == "assign") {
    agg = rval;
  } else if (agg_op == "add") {
    agg = lval + rval;
  } else if (agg_op == "max") {
    agg = _Cond(lval > rval, lval, rval);
  } else if (agg_op == "min") {
    agg = _Cond(lval < rval, lval, rval);
  } else if (agg_op == "mul") {
    agg = lval * rval;
  } else {
    throw std::runtime_error("Unknown agg-op:" + agg_op);
  }
  cur_->push_back(lval = agg);
}

void SemtreeEmitter::Visit(const stripe::LoadIndex& stmt) {
  auto lhs_name = scalar_name(stmt.into);
  auto rhs = convert_affine(stmt.from.sym_eval(scope_->idx_sources()));
  cur_->push_back(_Declare({sem::Type::VALUE, DataType::INT64}, lhs_name, rhs));
}

void SemtreeEmitter::Visit(const stripe::Constant& stmt) {
  sem::Type type(sem::Type::VALUE);
  sem::ExprPtr expr;
  if (stmt.type == stripe::ConstType::Integer) {
    type.dtype = DataType::INT64;
    expr = std::make_shared<sem::IntConst>(stmt.iconst);
  } else {
    type.dtype = DataType::FLOAT32;
    expr = std::make_shared<sem::FloatConst>(stmt.fconst);
  }
  cur_->push_back(_Declare(type, scalar_name(stmt.name), expr));
}

void SemtreeEmitter::Visit(const stripe::Special& spec) {
  std::string kname = "kernel_" + std::to_string(kernels_.kernels.size() + 1);
  if (spec.name == "reshape") {
    kernels_.kernels.push_back(lang::GenCopy(scope_->at(spec.inputs[0]).shape, spec.outputs[0], spec.inputs[0], kname));
    return;
  }
  // Try to call GenSpecial as a fallback by making a fake 'op'
  lang::Op op;
  op.tag = lang::Op::FUNCTION;
  op.f.fn = spec.name;
  op.inputs = spec.inputs;
  if (spec.outputs.size() > 1) {
    op.f.params = spec.outputs;
  } else {
    op.output = spec.outputs[0];
  }
  lang::HardwareSettings hw;
  hw.threads = threads_;
  hw.goal_dimension_sizes = lid_limits_;
  lang::Bindings vars;
  for (const auto& s : spec.inputs) {
    vars.emplace(s, lang::Binding(scope_->at(s).shape));
  }
  for (const auto& s : spec.outputs) {
    vars.emplace(s, lang::Binding(scope_->at(s).shape));
  }
  lang::GenSpecial(kernels_, op, vars, kname, hw);
}

void SemtreeEmitter::Visit(const stripe::Intrinsic& in) {
  static auto bin_ops = lang::BinaryOpMap();
  auto in_val = [this, &in](size_t i) { return _(scalar_name(in.inputs[i])); };
  auto in_cast = [this, &in](size_t i) { return _Cast({sem::Type::VALUE, in.type}, _(scalar_name(in.inputs[i]))); };
  sem::ExprPtr opexpr;
  if (in.inputs.size() == 2 && in.outputs.size() == 1 && bin_ops.count(in.name)) {
    opexpr = std::make_shared<sem::BinaryExpr>(bin_ops.at(in.name), in_cast(0), in_cast(1));
  } else if (in.name == "assign" || in.name == "ident" || in.name == "reshape" || in.name == "as_float" ||
             in.name == "as_int" || in.name == "as_uint") {
    opexpr = in_cast(0);
  } else if (in.name == "cond") {
    opexpr = _Cond(in_val(0), in_cast(1), in_cast(2));
  } else if (in.name == "neg") {
    opexpr = -in_cast(0);
  } else if (in.name == "bit_not") {
    opexpr = std::make_shared<sem::UnaryExpr>("~", in_val(0));
  } else if (in.name == "index") {
    throw std::runtime_error("index intrinsic goo is hard");
  } else {
    std::vector<sem::ExprPtr> inputs;
    for (const auto& str : in.inputs) {
      inputs.push_back(_(scalar_name(str)));
    }
    opexpr = std::make_shared<sem::CallExpr>(_(in.name), inputs);
  }
  IVLOG(1, "Pushing back on " << in.outputs[0]);
  cur_->push_back(_Declare({sem::Type::VALUE, in.type}, scalar_name(in.outputs[0]), opexpr));
}

sem::StmtPtr SemtreeEmitter::add_loops(const stripe::Block& block) {
  sem::StmtPtr top = cur_;
  for (const auto& idx : block.idxs) {
    if (idx.affine == stripe::Affine()) {
      top = _For(idx_name(idx.name), idx.range, 1, top);
    }
  }
  return top;
}

void SemtreeEmitter::do_gids(const stripe::Block& block) {
  std::vector<size_t> logical;
  for (const auto& idx : block.idxs) {
    logical.push_back(idx.range);
  }
  auto map = lang::gid::MakeMap(lid_limits_, logical, false);
  std::vector<std::shared_ptr<sem::Expression>> gids;
  for (size_t i = 0; i < 3; i++) {
    gids.push_back(_Index(sem::IndexExpr::GROUP, i));
  }
  for (size_t i = 0; i < block.idxs.size(); i++) {
    const auto& idx = block.idxs[i];
    cur_->push_front(_Declare({sem::Type::INDEX}, idx_name(idx.name), LogicalIndex(gids, map.dims[i])));
  }
  kernels_.kernels.emplace_back();
  lang::KernelInfo& ki = kernels_.kernels.back();
  ki.kname = "kernel_" + std::to_string(kernels_.kernels.size());
  ki.comments = "//" + block.name + "\n//" + block.comments + "\n";
  std::vector<sem::Function::param_t> params;
  for (const auto& ref : block.ref_outs()) {
    sem::Type type = {sem::Type::POINTER_MUT, ref->interior_shape.type, 1, 0, sem::Type::GLOBAL};
    params.push_back(std::make_pair(type, ref_name(ref->into)));
    ki.outputs.push_back(ref->from);
  }
  std::set<std::string> dups;
  for (const auto& ref : block.ref_ins()) {
    if (dups.count(ref->from)) {
      continue;
    }
    dups.insert(ref->from);
    sem::Type type = {sem::Type::POINTER_CONST, ref->interior_shape.type, 1, 0, sem::Type::GLOBAL};
    params.push_back(std::make_pair(type, ref_name(ref->into)));
    ki.inputs.push_back(ref->from);
  }
  ki.kfunc = std::make_shared<sem::Function>(ki.kname, sem::Type(), params, cur_);
  ki.gwork = {{map.gid_sizes[0] * threads_, map.gid_sizes[1], map.gid_sizes[2]}};
  ki.lwork = {{threads_, 1, 1}};
}

sem::StmtPtr SemtreeEmitter::do_lids(const stripe::Block& block) {
  sem::StmtPtr top = cur_;
  size_t prev_threads = 1;
  sem::ExprPtr tid = _Index(sem::IndexExpr::LOCAL, 0);
  for (size_t i = block.idxs.size(); i > 0; i--) {
    const auto& idx = block.idxs[i - 1];
    if (idx.affine != stripe::Affine()) {
      continue;
    }
    auto expr = tid / prev_threads;
    if (i != 1) {
      expr = expr % idx.range;
    }
    cur_->push_front(_Declare({sem::Type::INDEX}, idx_name(idx.name), expr));
    prev_threads *= idx.range;
  }
  if (block.idxs_product() < threads_) {
    top = sem::builder::_If(tid < sem::ExprPtr(_Const(block.idxs_product())), top);
  }
  return top;
}

void SemtreeEmitter::Visit(const stripe::Block& block) {
  std::shared_ptr<sem::Block> outer = cur_;
  if (block.has_tag("kernel")) {
    if (block.has_tag("zero")) {
      if (block.refs.size() != 1) {
        throw std::runtime_error("Zero kernels must have a single output");
      }
      std::string bname = block.refs[0].from;
      std::string kname = "kernel_" + std::to_string(kernels_.kernels.size() + 1);
      kernels_.kernels.push_back(lang::GenZero(scope_->at(bname).shape, bname, kname));
      return;
    }
    if (block.has_tag("copy")) {
      if (block.ref_outs().size() != 1 || block.ref_ins().size() != 1) {
        throw std::runtime_error("Copy kernels must have one output + one input");
      }
      std::string src_name = block.ref_ins()[0]->from;
      std::string dst_name = block.ref_outs()[0]->from;
      std::string kname = "kernel_" + std::to_string(kernels_.kernels.size() + 1);
      kernels_.kernels.push_back(lang::GenCopy(scope_->at(src_name).shape, dst_name, src_name, kname));
      return;
    }
    in_kernel_++;
  }
  if (block.has_tag("gpu_thread")) {
    in_threads_++;
  }
  scopes_.emplace_back(*scope_, const_cast<stripe::Block*>(&block));
  scope_ = &scopes_.back();
  // New inner block
  cur_ = std::make_shared<sem::Block>();
  depth_++;
  for (const auto& stmt : block.stmts) {
    stmt->Accept(this);
  }
  // Now, add any new locals
  for (const auto& ref : block.refs) {
    if (ref.dir == stripe::RefDir::None) {
      size_t size = ref.interior_shape.elem_size();
      sem::Type ptype = {sem::Type::VALUE, ref.interior_shape.type, 1, size,
                         (in_threads_ ? sem::Type::NORMAL : sem::Type::LOCAL)};
      sem::ExprPtr init = AggInit(ref.interior_shape.type, ref.agg_op);
      if (in_threads_) {
        cur_->push_front(_Declare(ptype, ref_name(ref.into), init));
      } else {
        if (init) {
          // Threaded initialization of buffers
          auto while_loop = _Block({});
          while_loop->push_back(_Declare({sem::Type::INDEX}, "_init_", sem::ExprPtr(_Index(sem::IndexExpr::LOCAL, 0))));
          while_loop->push_back(_While(_("_init_") < _Const(size), _Block({_(ref_name(ref.into))[_("_init_")] = init,
                                                                           _("_init_") = _("_init_") + threads_})));
          cur_->push_front(while_loop);
        }
        cur_->push_front(_Declare(ptype, ref_name(ref.into), sem::ExprPtr()));
      }
    }
  }
  if (block.constraints.size()) {
    sem::ExprPtr bexpr = _Const(1);
    for (const auto& con : block.constraints) {
      auto gcon = con.sym_eval(scope_->idx_sources());
      bexpr = bexpr & (convert_affine(gcon) >= sem::ExprPtr(_Const(0)));
    }
    cur_ = _Block({sem::builder::_If(bexpr, cur_)});
  }
  // Add block loops
  sem::StmtPtr top;
  if (block.has_tag("kernel") && in_kernel_ == 1) {
    do_gids(block);
  } else if (block.has_tag("main") || block.has_tag("program")) {
    // No-op
  } else if (block.has_tag("gpu_thread") && in_threads_ == 1) {
    outer->push_back(do_lids(block));
    outer->push_back(_Barrier());
  } else {
    outer->push_back(add_loops(block));
  }
  // Unwind depth
  if (block.has_tag("kernel")) {
    in_kernel_--;
  }
  if (block.has_tag("gpu_thread")) {
    in_threads_--;
  }
  depth_--;
  scopes_.pop_back();
  scope_ = &scopes_.back();
  cur_ = outer;
}

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
