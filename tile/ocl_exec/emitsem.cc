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

sem::ExprPtr SemtreeEmitter::default_intrinsic_emitter(const stripe::Intrinsic& in,
                                                       const std::map<std::string, sem::ExprPtr>& exprs) const {
  static auto bin_ops = lang::BinaryOpMap();
  auto in_val = [&](size_t i, bool cast = true) -> sem::ExprPtr {
    auto it = exprs.find(in.inputs[i]);
    sem::ExprPtr r;
    if (it == exprs.end()) {
      r = _(scalar_name(in.inputs[i]));
    } else {
      r = it->second;
    }
    if (cast) {
      r = _Cast({sem::Type::VALUE, in.type}, r);
    }
    return r;
  };
  sem::ExprPtr opexpr;
  if (in.inputs.size() == 2 && in.outputs.size() == 1 && bin_ops.count(in.name)) {
    opexpr = std::make_shared<sem::BinaryExpr>(bin_ops.at(in.name), in_val(0), in_val(1));
  } else if (in.name == "assign" ||    //
             in.name == "ident" ||     //
             in.name == "reshape" ||   //
             in.name == "as_float" ||  //
             in.name == "as_int" ||    //
             in.name == "as_uint") {
    opexpr = in_val(0);
  } else if (in.name == "cond") {
    opexpr = _Cond(in_val(0, false), in_val(1), in_val(2));
  } else if (in.name == "neg") {
    opexpr = -in_val(0);
  } else if (in.name == "bit_not") {
    opexpr = std::make_shared<sem::UnaryExpr>("~", in_val(0, false));
  } else {
    std::vector<sem::ExprPtr> inputs;
    for (const auto& str : in.inputs) {
      inputs.push_back(_(scalar_name(str)));
    }
    opexpr = std::make_shared<sem::CallExpr>(_(in.name), inputs);
  }
  return opexpr;
}

SemtreeEmitter::SemtreeEmitter(const AliasMap& am, size_t threads)  //
    : hw_threads_(threads), threads_(threads), lid_limits_({1024 * 1024, 1024 * 1024, 1024 * 1024}) {
  loop_mul_ = 1;
  scopes_.emplace_back(am);
  scope_ = &scopes_.back();
  for (const auto& spec : hal::opencl::ocl_intrinsics) {
    intrinsics_.add(spec);
  }
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

std::string SemtreeEmitter::ref_buf(const std::string& in) const {  //
  return safe_name("rb_" + scope_->at(in).base_name);
}

std::string SemtreeEmitter::ref_idx(const std::string& in, int delta) const {  //
  return "ri_" + std::to_string(depth_ + delta) + "_" + in;
}

std::string SemtreeEmitter::scalar_name(const std::string& in) const {
  if (in.size() < 1 || in[0] != '$') {
    throw std::runtime_error("SemtreeEmitter, invalid scalar name");
  }
  return "s_" + std::to_string(depth_) + "_" + in.substr(1, in.size() - 1);
}

std::string SemtreeEmitter::idx_name(const std::string& in) const {  //
  return "d" + std::to_string(depth_) + "_" + in;
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
  auto lval = _(ref_buf(stmt.from))[_(ref_idx(stmt.from))];
  auto type = scope_->at(stmt.from).shape.type;
  cur_->push_back(_Declare({sem::Type::VALUE, type}, scalar_name(stmt.into), lval));
  tot_loads_ += loop_mul_;
}

static sem::ExprPtr DoAgg(const std::string& agg_op, sem::ExprPtr lval, sem::ExprPtr rval) {
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
  return agg;
}

void SemtreeEmitter::Visit(const stripe::Store& stmt) {
  auto lval = _(ref_buf(stmt.into))[_(ref_idx(stmt.into))];
  std::string agg_op = scope_->at(stmt.into).base_ref->agg_op;
  auto rval = _(scalar_name(stmt.from));
  sem::ExprPtr agg = DoAgg(agg_op, lval, rval);
  cur_->push_back(lval = agg);
  if (agg_op != "" && agg_op != "assign") {
    tot_ops_ += loop_mul_;
  }
  tot_stores_ += loop_mul_;
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
  if (spec.name == "zero") {
    kernels_.kernels.push_back(lang::GenZero(scope_->at(spec.outputs[0]).shape, spec.outputs[0], kname));
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
  hw.threads = hw_threads_;
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
  sem::ExprPtr opexpr;
  if (intrinsics_.exist(in.name)) {
    opexpr = intrinsics_.emit(in);
  } else {
    opexpr = default_intrinsic_emitter(in);
  }
  tot_ops_ += loop_mul_;
  cur_->push_back(_Declare({sem::Type::VALUE, in.type}, scalar_name(in.outputs[0]), opexpr));
}

sem::StmtPtr SemtreeEmitter::add_loops(const stripe::Block& block) {
  sem::StmtPtr top = cur_;
  std::map<std::string, stripe::Affine> ref_flats;
  for (const auto& ref : block.refs) {
    ref_flats[ref.into()] = ref.FlatAccess();
  }
  std::map<std::string, int64_t> last_tot;
  // Make a loop for every 'true' index, from outside in
  for (const auto& idx : block.idxs) {
    if (idx.affine != stripe::Affine()) {
      continue;  // Skip computed indexes
    }
    auto inner = _Block({});
    inner->push_back(top);
    for (auto& kvp : ref_flats) {
      int64_t diff = kvp.second[idx.name] - last_tot[kvp.first];
      if (diff) {
        inner->push_back(_(ref_idx(kvp.first)) = _(ref_idx(kvp.first)) + _Const(diff));
      }
      last_tot[kvp.first] = kvp.second[idx.name] * idx.range;
      kvp.second.mutateMap().erase(idx.name);
    }
    top = _For(idx_name(idx.name), idx.range, 1, inner);
  }
  auto init = _Block({});
  for (const auto& rkvp : ref_flats) {
    stripe::Affine tot;
    for (const auto& mkvp : rkvp.second.getMap()) {
      if (mkvp.first == "") {
        tot += mkvp.second;
      } else {
        tot += mkvp.second * scope_->idx_sources().at(mkvp.first);
      }
    }
    std::string from = block.ref_by_into(rkvp.first)->from;
    auto ival = convert_affine(tot);
    if (from != "") {
      ival = ival + _(ref_idx(from, -1));
    }
    init->push_back(_Declare({sem::Type::INDEX}, ref_idx(rkvp.first), ival));
  }
  init->push_back(top);
  top = init;
  return top;
}

void SemtreeEmitter::do_gids(const stripe::Block& block) {
  if (block.ref_outs().size() == 0) {
    return;
  }
  std::vector<size_t> logical;
  for (const auto& idx : block.idxs) {
    logical.push_back(idx.range);
  }
  auto map = lang::gid::MakeMap(lid_limits_, logical, false);
  std::vector<std::shared_ptr<sem::Expression>> gids;
  for (size_t i = 0; i < 3; i++) {
    gids.push_back(_Index(used_threads_ == 1 ? sem::IndexExpr::GLOBAL : sem::IndexExpr::GROUP, i));
  }
  for (const auto& ref : block.refs) {
    auto new_aff = ref.FlatAccess().sym_eval(scope_->idx_sources());
    cur_->push_front(_Declare({sem::Type::INDEX}, ref_idx(ref.into()), convert_affine(new_aff)));
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
    params.push_back(std::make_pair(type, ref_buf(ref->into())));
    ki.outputs.push_back(ref->from);
  }
  std::set<std::string> dups;
  for (const auto& ref : block.ref_ins()) {
    if (dups.count(ref->from)) {
      continue;
    }
    dups.insert(ref->from);
    sem::Type type = {sem::Type::POINTER_CONST, ref->interior_shape.type, 1, 0, sem::Type::GLOBAL};
    params.push_back(std::make_pair(type, ref_buf(ref->into())));
    ki.inputs.push_back(ref->from);
  }
  ki.kfunc = std::make_shared<sem::Function>(ki.kname, sem::Type(), params, cur_);
  if (block.has_tag("subgroup_outer")) {
    ki.kfunc->is_subgroup = true;
  }
  ki.gwork = {{map.gid_sizes[0] * used_threads_, map.gid_sizes[1], map.gid_sizes[2]}};
  if (used_threads_ == 1) {
    ki.lwork = {{0, 0, 0}};
  } else {
    ki.lwork = {{used_threads_, 1, 1}};
  }

  ki.tot_flops = tot_ops_;
  tot_ops_ = 0;
  ki.tot_bytes = 4 * (tot_loads_ + tot_stores_);
  tot_loads_ = 0;
  tot_stores_ = 0;
}

sem::StmtPtr SemtreeEmitter::make_special(const std::string& name, const stripe::Block& block,
                                          std::vector<const stripe::Refinement*> args) {
  throw std::runtime_error("TODO: Special");
  /*
  std::vector<sem::ExprPtr> params;
  for (const stripe::Refinement* ref : args) {
    int64_t lda = 0;
    for (const auto d : block.exterior_shape(ref->into()).dims) {
      if (d.size > 1 && d.stride > 1) {
        lda = d.stride / 4;
      }
    }
    params.push_back(_(ref_buf(ref->from)));
    std::string idx = generate_name(ref->from);
    process_affine(idx, scope_->at(ref->from).flat());
    params.push_back(_(idx));
    params.push_back(_Const(lda));
  }
  return std::make_shared<sem::SpecialStmt>(name, params);
  */
}

void SemtreeEmitter::compute_thread_count(const stripe::Block& block) {
  if (block.has_tag("gpu_thread")) {
    size_t prod = 1;
    for (const auto& idx : block.idxs) {
      prod *= idx.range;
    }
    threads_ = std::max(threads_, prod);
  } else {
    for (auto stmt : block.stmts) {
      auto inner = stripe::Block::Downcast(stmt);
      if (inner) {
        compute_thread_count(*inner);
      }
    }
  }
}

sem::StmtPtr SemtreeEmitter::do_lids(const stripe::Block& block) {
  auto top = _Block({cur_});
  sem::ExprPtr tid = _Index(sem::IndexExpr::LOCAL, 0);
  for (const auto& ref : block.refs) {
    auto idx_value = convert_affine(ref.FlatAccess().sym_eval(scope_->idx_sources()));
    if (ref.from != "") {
      idx_value = idx_value + _(ref_idx(ref.from, -1));
    }
    top->push_front(_Declare({sem::Type::INDEX}, ref_idx(ref.into()), idx_value));
  }
  for (size_t i = block.idxs.size(); i > 0; i--) {
    const auto& idx = block.idxs[i - 1];
    if (idx.affine != stripe::Affine()) {
      continue;
    }
    auto expr = block.has_tag("reg_cache") ? ((tid % block.idxs_product()) / used_threads_) : (tid / used_threads_);
    expr = expr % idx.range;
    top->push_front(_Declare({sem::Type::INDEX}, idx_name(idx.name), expr));
    used_threads_ *= idx.range;
  }
  /*
  if (!block.has_tag("reg_cache") && block.idxs_product() < threads_) {
    return sem::builder::_If(tid < sem::ExprPtr(_Const(block.idxs_product())),
  top);
  }
  */
  return top;
}

void SemtreeEmitter::init_loop_local(const std::string& buf, DataType type,  //
                                     size_t size, const sem::ExprPtr& init) {
  if (size < threads_ * 2) {
    auto while_loop = _Block({});
    while_loop->push_back(_Declare({sem::Type::INDEX}, "_init_", sem::ExprPtr(_Index(sem::IndexExpr::LOCAL, 0))));
    while_loop->push_back(_While(_("_init_") < _Const(size),
                                 _Block({_(ref_buf(buf))[_("_init_")] = init, _("_init_") = _("_init_") + threads_})));
    cur_->push_front(while_loop);
  } else {
    size_t vec_size = size / threads_;
    if (vec_size > 8) {
      vec_size = 16;
    } else if (vec_size > 4) {
      vec_size = 8;
    } else if (vec_size > 2) {
      vec_size = 4;
    }  // else vec_size is 2

    size_t align_size = (size / vec_size) * vec_size;
    // We process buf[0..align_size-1] using vectors first
    auto inner = _Block({});
    std::string fname = "vstore" + std::to_string(vec_size);
    auto outer = _Block({});
    sem::Type ptype = {sem::Type::POINTER_MUT, type, 1, 0, sem::Type::NORMAL};
    sem::Type vptype = {sem::Type::POINTER_MUT, type, vec_size, 0, sem::Type::NORMAL};
    sem::Type vtype = {sem::Type::VALUE, type, vec_size, 0, sem::Type::NORMAL};
    outer->push_back(_Declare({sem::Type::INDEX}, "_thread_", sem::ExprPtr(_Index(sem::IndexExpr::LOCAL, 0))));
    outer->push_back(_Declare(vtype, "_init_", init));
    for (size_t i = 0; i < align_size / vec_size / threads_; i++) {
      sem::StmtPtr call;
      if (i > 0) {
        call = _Special(fname, {_("_init_"), _("_thread_") + _Const(i * threads_), _(ref_buf(buf))});
      } else {
        call = _Special(fname, {_("_init_"), _("_thread_"), _(ref_buf(buf))});
      }
      outer->push_back(call);
    }

    size_t last = align_size / vec_size / threads_;
    if (align_size > last * vec_size * threads_) {
      if (last > 0) {
        outer->push_back(_Declare({sem::Type::INDEX}, "_next_", _("_thread_") + (last * threads_)));
        sem::StmtPtr call = _Special(fname, {_("_init_"), _("_next_"), _(ref_buf(buf))});
        auto iftrue = _Block({call});
        auto ifstmt = sem::builder::_If(_("_next_") < _Const(align_size / vec_size), iftrue);
        outer->push_back(ifstmt);
      } else {
        sem::StmtPtr call = _Special(fname, {_("_init_"), _("_thread_"), _(ref_buf(buf))});
        auto iftrue = _Block({call});
        auto ifstmt = sem::builder::_If(_("_thread_") < _Const(align_size / vec_size), iftrue);
        outer->push_back(ifstmt);
      }
    }

    // Then initialize buf[align_size..size-1]
    // size - align_size < threads, so we do it in element-wise
    // Use the last threads first because they may be not full
    if (align_size < size) {
      // for thread idx, it initializes buf[idx + size - threads_]
      auto ifstmt = _Block({_Declare({sem::Type::INDEX}, "_buf_idx_", _("_thread_") + (size - threads_))});
      auto iftrue = _Block({_(ref_buf(buf))[_("_buf_idx_")] = init});
      ifstmt->push_back(sem::builder::_If(_("_buf_idx_") >= _Const(align_size), iftrue));
      outer->push_back(ifstmt);
    }

    cur_->push_front(outer);
  }
}

class Unroller : public stripe::ConstStmtVisitor {
 public:
  Unroller(const SemtreeEmitter& emitter, const stripe::Block& block) : emitter_(emitter), block_(block) {}
  sem::StmtPtr run() {
    out_ = _Block({});
    unroll_rec(0);
    out_->push_back(_Barrier(true));
    return out_;
  }

  void unroll_rec(size_t idx_num) {
    if (idx_num == block_.idxs.size()) {
      for (const auto& stmt : block_.stmts) {
        stmt->Accept(this);
      }
      return;
    }
    if (block_.idxs[idx_num].affine != stripe::Affine()) {
      idxs_[block_.idxs[idx_num].name] = emitter_.scope_->idx_sources().at(block_.idxs[idx_num].name);
      unroll_rec(idx_num + 1);
      return;
    } else {
      for (size_t i = 0; i < block_.idxs[idx_num].range; i++) {
        idxs_[block_.idxs[idx_num].name] = i;
        unroll_rec(idx_num + 1);
      }
    }
  }

  void Visit(const stripe::Load& load) {
    auto ref = block_.ref_by_into(load.from);
    auto aff = ref->FlatAccess().sym_eval(idxs_);
    auto idx_base = _(emitter_.ref_idx(ref->from, -1));
    auto idx_expr = idx_base + emitter_.convert_affine(aff);
    sem::ExprPtr val = _(emitter_.ref_buf(load.from))[idx_expr];
    if (load.has_tags({"subgroup_broadcast"})) {
      std::string subgroup_idx = ref->access[ref->bank_dim->dim_pos].getMap().begin()->first;
      auto subgroup = _Const(idxs_[subgroup_idx].constant());
      val = _("sub_group_broadcast")(val, subgroup);
    }
    scalars_[load.into] = val;
  }

  void Visit(const stripe::Store& store) {
    auto ref = block_.ref_by_into(store.into);
    auto aff = ref->FlatAccess().sym_eval(idxs_);
    auto idx_base = _(emitter_.ref_idx(ref->from, -1));
    auto idx_expr = idx_base + emitter_.convert_affine(aff);
    auto rval = scalars_[store.from];
    auto lval = _(emitter_.ref_buf(store.into))[idx_expr];
    std::string agg_op = emitter_.scope_->at(store.into).base_ref->agg_op;
    auto agg = DoAgg(agg_op, lval, rval);
    out_->push_back(lval = agg);
  }

  void Visit(const stripe::Constant& stmt) {
    sem::Type type(sem::Type::VALUE);
    sem::ExprPtr expr;
    if (stmt.type == stripe::ConstType::Integer) {
      type.dtype = DataType::INT64;
      expr = std::make_shared<sem::IntConst>(stmt.iconst);
    } else {
      type.dtype = DataType::FLOAT32;
      expr = std::make_shared<sem::FloatConst>(stmt.fconst);
    }
    scalars_[stmt.name] = expr;
  }

  void Visit(const stripe::Intrinsic& in) {
    sem::ExprPtr opexpr;
    if (emitter_.intrinsics_.exist(in.name)) {
      opexpr = emitter_.intrinsics_.emit(in);
    } else {
      opexpr = emitter_.default_intrinsic_emitter(in, scalars_);
    }
    scalars_[in.outputs[0]] = opexpr;
  }

  void Visit(const stripe::LoadIndex&) { throw std::runtime_error("LoadIndex unimplmented in unrolling"); }
  void Visit(const stripe::Special&) { throw std::runtime_error("Special unimplemented in unrolling"); }
  void Visit(const stripe::Block&) { throw std::runtime_error("Block unimplemented in unrolling"); }

 private:
  const SemtreeEmitter& emitter_;
  const stripe::Block& block_;
  std::shared_ptr<sem::Block> out_;
  std::map<std::string, stripe::Affine> idxs_;
  std::map<std::string, sem::ExprPtr> scalars_;
};

void SemtreeEmitter::Visit(const stripe::Block& block) {
  std::shared_ptr<sem::Block> outer = cur_;
  if (block.has_tag("mac_inner")) {
    std::vector<const stripe::Refinement*> args = {block.ref_ins()[0], block.ref_ins()[1]};
    cur_->push_back(make_special("MAC_INNER", block, args));
    return;
  }
  if (block.has_tag("subgroup_inline")) {
    scopes_.emplace_back(*scope_, const_cast<stripe::Block*>(&block));
    scope_ = &scopes_.back();
    depth_++;
    Unroller unroll(*this, block);
    cur_->push_back(unroll.run());
    depth_--;
    scopes_.pop_back();
    scope_ = &scopes_.back();
    return;
  }
  if (block.has_tag("kernel")) {
    if (block.has_tag("zero")) {
      if (block.refs.size() != 1) {
        throw std::runtime_error("Zero kernels must have a single output");
      }
      std::string bname = block.ref_outs()[0]->from;
      std::string kname = "kernel_" + std::to_string(kernels_.kernels.size() + 1);
      auto ki = lang::GenZero(scope_->at(bname).shape, bname, kname);
      ki.comments = "ZERO";
      kernels_.kernels.push_back(ki);
      return;
    }
    if (block.has_tag("copy")) {
      if (block.ref_outs().size() != 1 || block.ref_ins().size() != 1) {
        throw std::runtime_error("Copy kernels must have one output + one input");
      }
      std::string src_name = block.ref_ins()[0]->from;
      std::string dst_name = block.ref_outs()[0]->from;
      std::string kname = "kernel_" + std::to_string(kernels_.kernels.size() + 1);
      auto ki = lang::GenCopy(scope_->at(src_name).shape, dst_name, src_name, kname);
      ki.comments = "COPY";
      kernels_.kernels.push_back(ki);
      return;
    }
    in_kernel_++;
    if (block.has_tag("no_threads")) {
      in_threads_++;
    }
    used_threads_ = 1;
  }
  if (block.has_tag("gpu_thread")) {
    in_threads_++;
  }
  scopes_.emplace_back(*scope_, const_cast<stripe::Block*>(&block));
  scope_ = &scopes_.back();

  // New inner block
  cur_ = std::make_shared<sem::Block>();
  depth_++;
  loop_mul_ *= block.idxs_product();
  for (const auto& stmt : block.stmts) {
    stmt->Accept(this);
  }

  // When two same refinements are used in the same stripe block,
  // do not emit duplicated local declarations.
  std::set<std::string> dup_ref;

  // Now, add any new locals
  for (const auto& ref : block.refs) {
    if (ref.dir == stripe::RefDir::None && dup_ref.find(ref.into()) == dup_ref.end()) {
      dup_ref.insert(ref.into());
      bool use_register = ref.location.devs.size() == 1 && ref.location.devs[0].name == "REGISTER";
      size_t size = ref.interior_shape.elem_size();
      sem::Type ptype = {sem::Type::VALUE, ref.interior_shape.type, 1, size,
                         ((in_threads_ || use_register) ? sem::Type::NORMAL : sem::Type::LOCAL)};
      sem::ExprPtr init = AggInit(ref.interior_shape.type, ref.agg_op);
      if (use_register || in_threads_) {
        cur_->push_front(_Declare(ptype, ref_buf(ref.into()), init));
      } else {
        if (init) {
          cur_->push_front(_Barrier());
          init_loop_local(ref.into(), ref.interior_shape.type, size, init);
        }
        cur_->push_front(_Declare(ptype, ref_buf(ref.into()), sem::ExprPtr()));
      }
    }
  }
  // Wrap in an if when required
  if (block.constraints.size()) {
    sem::ExprPtr bexpr = _Const(1);
    for (const auto& con : block.constraints) {
      auto gcon = con.sym_eval(scope_->idx_sources());
      bexpr = bexpr & (convert_affine(gcon) >= sem::ExprPtr(_Const(0)));
    }
    cur_ = _Block({sem::builder::_If(bexpr, cur_)});
  }
  // Add block loops
  if (block.has_tag("kernel") && in_kernel_ == 1) {
    do_gids(block);
    if (block.has_tag("no_threads")) {
      in_threads_--;
    }
  } else if (block.has_tag("main") || block.has_tag("program")) {
    // No-op
  } else if (block.has_tag("gpu_thread") && in_threads_ >= 1) {
    outer->push_back(do_lids(block));
    outer->push_back(_Barrier());
  } else {
    auto wrapped = add_loops(block);
    if (block.has_tag("mac_middle")) {
      std::shared_ptr<sem::Block> sblock;
      if (wrapped->isBlock()) {
        sblock = std::dynamic_pointer_cast<sem::Block>(wrapped);
      } else {
        sblock = std::make_shared<sem::Block>();
        sblock->push_back(wrapped);
      }
      sblock->push_front(make_special("MAC_INIT", block, {}));
      sblock->push_back(make_special("MAC_FINISH", block, {block.ref_outs()[0]}));
      wrapped = sblock;
    }
    outer->push_back(wrapped);
  }
  // Unwind depth
  if (block.has_tag("kernel")) {
    in_kernel_--;
  }
  if (block.has_tag("gpu_thread")) {
    in_threads_--;
  }
  loop_mul_ /= block.idxs_product();
  depth_--;
  scopes_.pop_back();
  scope_ = &scopes_.back();
  cur_ = outer;
}

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
