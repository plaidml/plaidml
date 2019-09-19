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

  // Public purely to avoid annoyance with ForAllOps
  template <class ScalarOp>
  void apply();

 private:
  TensorShape get_shape(TensorType type);
  void add_attributes(stripe::Taggable& out, ArrayRef<NamedAttribute> in);  // NOLINT
  void add_refinements(Block* block, Value* tensor, stripe::RefDir dir, std::string* name_out, std::string agg = "");
  stripe::Location build_location(stripe::Block* block, Value* device_path);
  std::string get_idx(stripe::Block* block, mlir::BlockArgument* affine);
  stripe::Affine build_affine(stripe::Block* block, Value* affine);
  void visit(ParallelForOp op);
  void visit(ConstraintOp op, int count);
  void visit(ExecutorOp op, int count);
  void visit(LoadOp op);
  void visit(StoreOp op);
  void visit(AggregateOp op);
  void walk_interior(Block* inner);

  std::shared_ptr<stripe::Block> cur_;
  size_t next_scalar_ = 0;
  std::map<mlir::Block*, BlockInfo> blocks_;
  std::map<std::pair<stripe::Block*, mlir::BlockArgument*>, std::string> idxs_;
  std::map<mlir::Value*, std::string> scalars_;
  Operation* iop;
  bool found_inst_;
};

}  // End namespace

StripeBuilder::StripeBuilder(mlir::FuncOp func) {
  // Construct the block and put it in the table
  cur_ = std::make_shared<stripe::Block>();
  cur_->name = func.getName();
  std::vector<NamedAttribute> attrs(func.getDialectAttrs().begin(), func.getDialectAttrs().end());
  add_attributes(*cur_, attrs);
  Block& oblock = func.front();
  BlockInfo blockInfo(cur_.get());
  for (size_t i = 0; i < func.getNumArguments(); i++) {
    // add refinement for each arg
    auto arg = func.getArgument(i);
    auto name = func.getArgAttr(i, "stripe.name").cast<StringAttr>().getValue();
    // Compute all the info about the tensor
    auto ti = ComputeAccess(arg);
    // Translate allocation shape
    TensorShape shape = get_shape(ti.base->getType().cast<TensorType>());
    std::vector<stripe::Affine> access(ti.access.size());
    stripe::Refinement ref{stripe::RefDir::None, "", name.str(), access, shape};
    ref.set_attr("user");
    if (arg->hasOneUse()) {
      if (auto op = mlir::dyn_cast<TensorRefOp>(*arg->user_begin())) {
        ref.location = build_location(cur_.get(), op.devicePath());
      }
    }
    cur_->refs.emplace(ref);
    blockInfo.refs[ti] = name.str();
  }
  blocks_.emplace(&oblock, blockInfo);
  walk_interior(&oblock);
}

TensorShape StripeBuilder::get_shape(TensorType type) {
  TensorShape ret;
  auto elementType = type.getElementType().cast<eltwise::ScalarType>();
  ret.type = elementType.type();
  for (const auto& dim : type.getShape()) {
    ret.dims.emplace_back(dim.stride, dim.size);
  }
  return ret;
}

void StripeBuilder::add_attributes(stripe::Taggable& out, ArrayRef<NamedAttribute> attrs) {
  for (auto kvp : attrs) {
    std::string name;
    if (kvp.first.strref().count('.')) {
      name = kvp.first.strref().split('.').second;
    } else {
      name = kvp.first.str();
    }
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
      IVLOG(1, "Attr: " << name);
      throw std::runtime_error("Invalid attribute during conversion");
    }
  }
}

void StripeBuilder::add_refinements(Block* block, Value* tensor, stripe::RefDir dir, std::string* name_out,
                                    std::string agg_name) {
  // Compute all the info about the tensor
  auto ti = ComputeAccess(tensor);
  // Translate allocation shape
  TensorShape base_shape = get_shape(ti.base->getType().cast<TensorType>());
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
      sblock->refs.emplace(dir, "", rname, access, shape, agg_name);
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
    // Remove indexes from ti for this block + add to inner polynomial
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
    // If op matches the block, move the op up, also add location and attributes
    while (op->getBlock() == block && mlir::isa<RefineOp>(op)) {
      ref->location = build_location(blocks_.at(block).stripe, mlir::cast<RefineOp>(op).devicePath());
      add_attributes(*ref, mlir::cast<RefineOp>(op).attrs().getValue());
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
  // Add location info to the allocation block
  if (auto alloc_op = mlir::dyn_cast<AllocateOp>(op)) {
    ref->location = build_location(blocks_.at(block).stripe, alloc_op.devicePath());
  }
}

stripe::Location StripeBuilder::build_location(stripe::Block* block, Value* device_path) {
  stripe::Location result;
  if (auto dev_path_op = mlir::cast<DevicePathOp>(device_path->getDefiningOp())) {
    for (const auto& dev_id : dev_path_op.dev_ids()) {
      if (auto dev_id_op = mlir::cast<DeviceIDOp>(dev_id->getDefiningOp())) {
        std::vector<stripe::Affine> units;
        for (auto* unit : dev_id_op.unit()) {
          units.emplace_back(build_affine(block, unit));
        }
        result.devs.emplace_back(stripe::Device{dev_id_op.name(), std::move(units)});
      }
    }
  }
  return result;
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
      add_attributes(cur_->idxs.back(), meta.attrs().getValue());
    }
  }
  // Add the attributes
  add_attributes(*cur_, op.attrs().getValue());
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

void StripeBuilder::visit(ExecutorOp op, int count) {
  Block* body = &op.body().front();
  blocks_.emplace(body, BlockInfo(nullptr));
  walk_interior(body);
  // Find the stripe block to attach the execution location to
  Block* block = body;
  while (blocks_.at(block).stripe == nullptr) {
    block = block->getParentOp()->getBlock();
  }
  stripe::Block* sblock = blocks_.at(block).stripe;
  if (sblock->location.devs.size() != 0) {
    throw std::runtime_error("Multiple executors not supported right now");
  }
  sblock->location = build_location(sblock, op.devicePath());
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

void StripeBuilder::visit(AggregateOp op) {
  std::string ref_name;
  AggTypeEnum agg_enum = static_cast<AggTypeEnum>(op.agg_type().getLimitedValue());
  std::string agg_name = stringifyAggTypeEnum(agg_enum);
  add_refinements(op.getOperation()->getBlock(), op.into(), stripe::RefDir::Out, &ref_name, agg_name);
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
    } else if (auto op = mlir::dyn_cast<ExecutorOp>(op_base)) {
      visit(op, count);
    } else if (auto op = mlir::dyn_cast<LoadOp>(op_base)) {
      visit(op);
    } else if (auto op = mlir::dyn_cast<StoreOp>(op_base)) {
      visit(op);
    } else if (auto op = mlir::dyn_cast<AggregateOp>(op_base)) {
      visit(op);
    } else {
      found_inst_ = false;
      // Try all the intrinsic ops
      iop = &op_base;
      eltwise::ForAllOps(*this);
      /*
      if (!found_inst_) {
        std::cout << "Unable to convert instruction: " << iop << "\n";
        throw std::runtime_error("Unable to find an instruction");
      }
      */
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
    found_inst_ = true;
  }
}

template <>
void StripeBuilder::apply<eltwise::ScalarConstantOp>() {
  if (auto op = mlir::dyn_cast<eltwise::ScalarConstantOp>(iop)) {
    std::string out_name = std::string("$c") + std::to_string(next_scalar_++);
    scalars_.emplace(op.result(), out_name);
    std::shared_ptr<stripe::Constant> cnst;
    auto val_attr = op.getValue();
    if (auto attr = val_attr.dyn_cast<IntegerAttr>()) {
      cnst = std::make_shared<stripe::Constant>(out_name, attr.getInt());
    } else if (auto attr = val_attr.dyn_cast<FloatAttr>()) {
      cnst = std::make_shared<stripe::Constant>(out_name, attr.getValueAsDouble());
    } else {
      throw std::runtime_error("Invalid attribute during conversion");
    }
    cur_->stmts.push_back(cnst);
  }
}

std::shared_ptr<stripe::Program> FromMLIR(mlir::ModuleOp module) {
  IVLOG(1, "FromMLIR");
  auto func = llvm::dyn_cast<mlir::FuncOp>(module.getBody()->front());
  StripeBuilder builder(func);
  auto ret = std::make_shared<stripe::Program>();
  ret->entry = builder.getResult();
  return ret;
}

}  // namespace stripe
}  // namespace dialect
}  // namespace pmlc
