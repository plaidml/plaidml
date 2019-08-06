// Copyright 2019, Intel Corporation

#include "pmlc/dialect/stripe/transcode.h"

#include "base/util/lookup.h"
#include "pmlc/dialect/eltwise/ops.h"
#include "pmlc/dialect/stripe/analysis.h"

namespace pmlc {
namespace dialect {
namespace stripe {

using vertexai::safe_at;
using DataType = vertexai::tile::DataType;
using TensorShape = vertexai::tile::TensorShape;

namespace {

struct BlockInfo {
  explicit BlockInfo(stripe::Block* stripe) : stripe(stripe) {}
  stripe::Block* stripe;  // If null, skip (for constraints that merge with outer block)
  std::map<FlatTensorAccess, std::string> refs;
};

class StripeBuilder {
 public:
  explicit StripeBuilder(mlir::FuncOp func);
  std::shared_ptr<stripe::Block> getResult() { return cur_; }

  // Public purly to avoid annoyance with ForAllOps
  template <class ScalarOp>
  void apply();

 private:
  TensorShape get_shape(TensorLayoutAttr layout);
  void add_attributes(stripe::Taggable& out, DictionaryAttr in);  // NOLINT
  void add_refinements(Block* block, Value* tensor, stripe::RefDir dir, std::string* name_out);
  std::string get_idx(stripe::Block* block, mlir::BlockArgument* affine);
  stripe::Affine build_affine(stripe::Block* block, Value* affine);
  void visit(ParallelForOp op);
  void visit(ConstraintOp op, int count);
  void visit(LoadOp op);
  void visit(StoreOp op);
  void walk_interior(Block* inner);

  std::shared_ptr<stripe::Block> cur_;
  size_t next_scalar_ = 0;
  std::map<mlir::Block*, BlockInfo> blocks_;
  std::map<mlir::Value*, std::string> refs_;
  std::map<std::pair<stripe::Block*, mlir::BlockArgument*>, std::string> idxs_;
  std::map<mlir::Value*, std::string> scalars_;
  Operation* iop;
};

}  // End namespace

StripeBuilder::StripeBuilder(mlir::FuncOp func) {
  ParallelForOp op(&func.front().front());
  visit(op);
}

TensorShape StripeBuilder::get_shape(TensorLayoutAttr layout) {
  TensorShape r;
  r.type = layout.base().type();
  for (const auto& d : layout.dims()) {
    r.dims.emplace_back(d.stride, d.size);
  }
  return r;
}

void StripeBuilder::add_attributes(stripe::Taggable& out, DictionaryAttr in) {
  for (auto& kvp : in) {
    std::string name = kvp.first.str();
    if (name == "__name" || name == "__comments") {
      continue;
    }
    if (kvp.second.dyn_cast<UnitAttr>()) {
      out.set_attr(name);
    } else if (auto attr = kvp.second.dyn_cast<BoolAttr>()) {
      out.set_attr(name, attr.getValue());
    } else if (auto attr = kvp.second.dyn_cast<IntegerAttr>()) {
      out.set_attr(name, attr.getInt());
    } else if (auto attr = kvp.second.dyn_cast<FloatAttr>()) {
      out.set_attr(name, attr.getValueAsDouble());
    } else if (auto attr = kvp.second.dyn_cast<IntegerAttr>()) {
      out.set_attr(name, attr.getInt());
    } else if (auto attr = kvp.second.dyn_cast<StringAttr>()) {
      out.set_attr(name, attr.getValue().str());
    } else {
      throw std::runtime_error("Invalid attribute during conversion");
    }
  }
}

void StripeBuilder::add_refinements(Block* block, Value* tensor, stripe::RefDir dir, std::string* name_out) {
  // Compute all the info about the tensor
  auto ti = ComputeAccess(tensor);
  // Translate allocation shape
  TensorShape base_shape = get_shape(ti.base.layout());
  // Make a vector of 'inner' polynomials
  std::vector<AffinePolynomial> inner(base_shape.dims.size());
  // Move constants to inner
  for (size_t i = 0; i < inner.size(); i++) {
    std::swap(ti.access[i].constant, inner[i].constant);
  }
  // Get the source instruction
  auto op = tensor->getDefiningOp();
  // Go up block by block, adding refinements
  // Track the prior one added to allow correcting the 'from'
  // Stop when we've hit the 'allocating block'
  stripe::Refinement* ref = nullptr;
  while (true) {
    if (blocks_.at(block).stripe == NULL) {
      // This is a 'fake' block introduced by merged constraints
      // First, make sure to move op up if needed, and then go to next block
      while (op->getBlock() == block && mlir::isa<RefineOp>(op)) {
        op = mlir::cast<RefineOp>(op).in()->getDefiningOp();
      }
      block = block->getParentOp()->getBlock();
      continue;
    }
    stripe::Block* sblock = blocks_.at(block).stripe;
    // Begin by seeing if we've already made a ref for this block
    std::string& rname = (blocks_.at(block).refs[ti]);
    // If not, add the refinement
    if (rname == "") {
      if (auto rop = mlir::dyn_cast<RefineOp>(op)) {
        if (auto attr = rop.attrs().get("__name")) {
          if (auto sattr = attr.dyn_cast<StringAttr>()) {
            rname = sattr.getValue().str();
          }
        }
      }
      if (rname == "") {
        rname = "ref";
      }
      rname = sblock->unique_ref_name(rname);
      std::vector<stripe::Affine> access;
      for (size_t i = 0; i < ti.access.size(); i++) {
        stripe::Affine aff;
        for (const auto& kvp : ti.access[i].terms) {
          if (kvp.first->getOwner() == block) {
            aff += stripe::Affine(get_idx(sblock, kvp.first), kvp.second);
          }
        }
        access.push_back(aff);
      }
      TensorShape shape = base_shape;
      for (size_t i = 0; i < shape.dims.size(); i++) {
        auto range = AffineRange(inner[i]);
        access[i] += range.min;
        inner[i].constant -= range.min;
        shape.dims[i].size = range.max - range.min + 1;
        shape.dims[i].size = std::min(shape.dims[i].size, base_shape.dims[i].size);
      }
      sblock->refs.emplace(dir, "", rname, access, shape);
    }
    // Connect up previously added block
    if (ref) {
      ref->from = rname;
    } else {
      *name_out = rname;
    }
    // Get pointer to new/etc reference
    ref = &blocks_.at(block).stripe->ref_by_into(rname)->mut();
    // Add in directionality
    if (ref->dir != stripe::RefDir::InOut && ref->dir != dir) {
      ref->dir = stripe::RefDir::InOut;
    }
    // Remove indexes from ti for this block + add to inner polynomail
    for (size_t i = 0; i < ti.access.size(); i++) {
      auto& ap = ti.access[i];
      auto& ip = inner[i];
      for (auto it = ap.terms.begin(); it != ap.terms.end(); /* nothing */) {
        if (it->first->getOwner() == block) {
          ip.terms.emplace(*it);
          it = ap.terms.erase(it);
        } else {
          ++it;
        }
      }
    }
    // If op matches the block, move the op up, also add attributes
    while (op->getBlock() == block && mlir::isa<RefineOp>(op)) {
      add_attributes(*ref, mlir::cast<RefineOp>(op).attrs());
      op = mlir::cast<RefineOp>(op).in()->getDefiningOp();
    }
    // Must have hit the allocation, stop
    if (op->getBlock() == block) {
      break;
    }
    // Move one block up
    block = block->getParentOp()->getBlock();
  }
  // Special handing for allocation block
  ref->dir = stripe::RefDir::None;
}

std::string StripeBuilder::get_idx(stripe::Block* block, mlir::BlockArgument* affine) {
  auto key = std::make_pair(block, affine);
  auto it = idxs_.find(key);
  if (it == idxs_.end()) {
    throw std::runtime_error("Need to add passthurs");
  }
  return it->second;
}

stripe::Affine StripeBuilder::build_affine(stripe::Block* block, Value* base) {
  stripe::Affine r;
  AffinePolynomial poly(base);
  r += poly.constant;
  for (auto& kvp : poly.terms) {
    std::string name = get_idx(block, kvp.first);
    r += stripe::Affine(name, kvp.second);
  }
  return r;
}

void StripeBuilder::visit(ParallelForOp op) {
  // Construct the block and put it in the table
  cur_ = std::make_shared<stripe::Block>();
  Block& oblock = op.inner().front();
  // Move across the easy metadata
  if (auto attr = op.attrs().get("__name")) {
    if (auto sattr = attr.dyn_cast<StringAttr>()) {
      cur_->name = sattr.getValue().str();
    }
  }
  if (auto attr = op.attrs().get("__comments")) {
    if (auto sattr = attr.dyn_cast<StringAttr>()) {
      cur_->comments = sattr.getValue().str();
    }
  }
  // Add the 'true' indexes
  for (size_t i = 0; i < op.ranges().size(); i++) {
    int64_t range = op.ranges().getValue()[i].cast<IntegerAttr>().getInt();
    mlir::BlockArgument* ovpos = oblock.getArgument(i);
    Value* vpos = ovpos;
    std::string iname = "idx";
    if (vpos->hasOneUse()) {
      auto op = vpos->use_begin()->getOwner();
      if (auto meta = mlir::dyn_cast<AffineMeta>(op)) {
        vpos = meta.result();
        if (auto attr = meta.attrs().get("__name")) {
          if (auto sattr = attr.dyn_cast<StringAttr>()) {
            iname = sattr.getValue().str();
          }
        }
      }
    }
    iname = cur_->unique_idx_name(iname);
    idxs_.emplace(std::make_pair(cur_.get(), ovpos), iname);
    cur_->idxs.emplace_back(iname, range);
    if (auto meta = mlir::dyn_cast<AffineMeta>(vpos->getDefiningOp())) {
      add_attributes(cur_->idxs.back(), meta.attrs());
    }
  }
  // Add the attributes
  add_attributes(*cur_, op.attrs());
  blocks_.emplace(&oblock, BlockInfo(cur_.get()));
  walk_interior(&oblock);
}

void StripeBuilder::visit(ConstraintOp op, int count) {
  if (count == 1 && op.lt_case().empty()) {
    Block* inner = &op.ge_case().front();
    blocks_.emplace(inner, BlockInfo(nullptr));
    walk_interior(inner);
    // Find the stripe block to attach the contraint to
    Block* block = inner;
    while (blocks_.at(block).stripe == nullptr) {
      block = block->getParentOp()->getBlock();
    }
    stripe::Block* sblock = blocks_.at(block).stripe;
    sblock->constraints.push_back(build_affine(sblock, op.input()));
  } else {
    throw std::runtime_error("Complex contraints not supported right now");
  }
}

void StripeBuilder::visit(LoadOp op) {
  std::string ref_name;
  add_refinements(op.getOperation()->getBlock(), op.from(), stripe::RefDir::In, &ref_name);
  std::string into = std::string("$s") + std::to_string(next_scalar_++);
  scalars_.emplace(op.into(), into);
  cur_->stmts.push_back(std::make_shared<stripe::Load>(ref_name, into));
}

void StripeBuilder::visit(StoreOp op) {
  std::string ref_name;
  add_refinements(op.getOperation()->getBlock(), op.into(), stripe::RefDir::Out, &ref_name);
  std::string from = scalars_.at(op.from());
  cur_->stmts.push_back(std::make_shared<stripe::Store>(from, ref_name));
}

void StripeBuilder::walk_interior(Block* block) {
  // Count inner ops
  int count = 0;
  for (auto& op : *block) {
    if (op.hasNoSideEffect() || op.isKnownTerminator()) {
      continue;
    }
    count++;
  }
  // Go over all the inner ops
  for (auto& op_base : *block) {
    if (auto op = mlir::dyn_cast<ParallelForOp>(op_base)) {
      auto old_cur = cur_;
      visit(op);
      old_cur->stmts.push_back(cur_);
      cur_ = old_cur;
    } else if (auto op = mlir::dyn_cast<ConstraintOp>(op_base)) {
      visit(op, count);
    } else if (auto op = mlir::dyn_cast<LoadOp>(op_base)) {
      visit(op);
    } else if (auto op = mlir::dyn_cast<StoreOp>(op_base)) {
      visit(op);
    } else {
      // Try all the intrinsic ops
      iop = &op_base;
      eltwise::ForAllOps(*this);
    }
  }
}

template <class ScalarOp>
void StripeBuilder::apply() {
  if (auto op = mlir::dyn_cast<ScalarOp>(iop)) {
    std::string out_name = std::string("$s") + std::to_string(next_scalar_++);
    scalars_.emplace(op.result(), out_name);
    auto intr = std::make_shared<stripe::Intrinsic>();
    std::string dialect = op.getOperation()->getName().getDialect().str();
    std::string full_name = op.getOperation()->getName().getStringRef().str();
    intr->name = full_name.substr(dialect.size() + 1, full_name.size() - dialect.size() - 1);
    intr->outputs.push_back(out_name);
    for (size_t i = 0; i < ScalarOp::operands(); i++) {
      intr->inputs.push_back(scalars_.at(op.getOperation()->getOperand(i)));
    }
    cur_->stmts.push_back(intr);
  }
}

stripe::Program ToStripe(mlir::FuncOp func) {
  stripe::Program r;
  StripeBuilder builder(func);
  r.entry = builder.getResult();
  return r;
}

}  // namespace stripe
}  // namespace dialect
}  // namespace pmlc
