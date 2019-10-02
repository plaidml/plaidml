// Copyright 2019, Intel Corporation

#include "pmlc/dialect/stripe/transcode.h"

#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Regex.h"

#include "base/util/lookup.h"
#include "pmlc/dialect/eltwise/ops.h"
#include "pmlc/dialect/stripe/analysis.h"
#include "pmlc/dialect/stripe/dialect.h"

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

struct ScalarInfo {
  std::set<std::string> names;
  size_t next = 0;
};

class StripeBuilder {
 public:
  explicit StripeBuilder(mlir::FuncOp func);
  std::shared_ptr<stripe::Block> getResult() { return cur_; }

  // Public purely to avoid annoyance with ForAllOps
  template <class ScalarOp>
  void apply();

 private:
  std::string scalar_name(Operation* op);
  TensorShape get_shape(TensorType type);
  void add_attributes(stripe::Taggable* out, ArrayRef<NamedAttribute> in);
  void add_refinements(Block* block, Value* tensor, stripe::RefDir dir, std::string* name_out, std::string agg = "",
                       bool is_spec = false);
  std::string get_idx(stripe::Block* block, mlir::BlockArgument* affine);
  stripe::Affine build_affine(stripe::Block* block, Value* affine);
  void visit(ParallelForOp op);
  void visit(ConstraintOp op, int count);
  void visit(LoadOp op);
  void visit(LoadIndexOp op);
  void visit(StoreOp op);
  void visit(AggregateOp op);
  void visit(SpecialOp op);
  void walk_interior(Block* inner);

  std::shared_ptr<stripe::Block> cur_;
  std::map<stripe::Block*, ScalarInfo> scalar_names_;
  std::map<mlir::Block*, BlockInfo> blocks_;
  std::map<std::pair<stripe::Block*, mlir::BlockArgument*>, std::string> idxs_;
  std::map<mlir::Value*, std::string> scalars_;
  Operation* iop;
  bool found_inst_;
};

struct StripeDevice {
  std::string name;
  std::vector<AffinePolynomial> units;
};

struct StripeLocation {
  std::vector<StripeDevice> devs;
};

std::pair<FlatTensorAccess, StripeLocation> ComputeAccessAndLoc(Value* tensor) {
  auto ret = ComputeAccess(tensor);
  auto loc = StripeLocation{};

  // At this point: ret.access contains the refinement accessors for every tensor dimension (including
  // non-address dimensions), and ret.base_type is a TensorType which gets us the underyling dimension
  // information.  For conversion to Stripe, we want to separate out the address dimensions (whose accessors
  // are preserved in the flat tensor access) from the non-address dimensions (which become part of the
  // location).

  std::vector<AffinePolynomial> access;

  auto combIt = ret.access.begin();
  auto matches = llvm::SmallVector<StringRef, 4>();
  for (const auto& dim : ret.base_type.getShape()) {
    if (combIt == ret.access.end()) {
      break;  // Should never happen; this is just to be careful.
    }
    if (dim.cls == "address") {
      access.emplace_back(*combIt);
    } else {
      static llvm::Regex re{R"(([[:alpha:]]+)_([[:digit:]]+)_([[:digit:]]+))"};
      if (re.match(dim.cls, &matches)) {
        const auto& dev_name = matches[1];
        std::size_t dev_idx, unit_idx;
        matches[2].getAsInteger(10, dev_idx);
        matches[3].getAsInteger(10, unit_idx);

        if (loc.devs.size() <= dev_idx) {
          loc.devs.resize(dev_idx + 1);
        }
        auto& dev = loc.devs.at(dev_idx);
        dev.name = dev_name;
        if (dev.units.size() <= unit_idx) {
          dev.units.resize(unit_idx + 1);
        }
        dev.units.at(unit_idx) = *combIt;
      }
    }
    ++combIt;
  }

  ret.access.swap(access);
  return std::make_pair(ret, loc);
}

}  // End namespace

StripeBuilder::StripeBuilder(mlir::FuncOp func) {
  // Construct the block and put it in the table
  cur_ = std::make_shared<stripe::Block>();
  cur_->name = func.getName();
  auto attrs = func.getAttrOfType<DictionaryAttr>(Dialect::getStripeAttrsName());
  add_attributes(cur_.get(), attrs.getValue());
  Block& oblock = func.front();
  BlockInfo blockInfo(cur_.get());
  for (size_t i = 0; i < func.getNumArguments(); i++) {
    // add refinement for each arg
    auto arg = func.getArgument(i);
    auto attrName = Dialect::getDialectAttrName(func.getContext(), "name");
    auto name = func.getArgAttr(i, attrName).cast<StringAttr>().getValue();
    // Compute all the info about the tensor
    auto ti = ComputeAccessAndLoc(arg).first;
    // Translate allocation shape
    TensorShape shape = get_shape(ti.base->getType().cast<TensorType>());
    std::vector<stripe::Affine> access(ti.access.size());
    stripe::Refinement ref{stripe::RefDir::None, "", name.str(), access, shape};
    ref.set_attr("user");
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
    if (dim.cls == "address") {
      ret.dims.emplace_back(dim.stride, dim.size);
    }
  }
  ret.is_const = type.is_const();
  return ret;
}

void StripeBuilder::add_attributes(stripe::Taggable* out, ArrayRef<NamedAttribute> attrs) {
  for (auto kvp : attrs) {
    llvm::StringRef name = kvp.first.strref();
    if (name.count('.')) {
      name = name.split('.').second;
    }
    if (name == "__name") {
      continue;
    }
    if (kvp.second.dyn_cast<UnitAttr>()) {
      out->set_attr(name);
    } else if (auto attr = kvp.second.dyn_cast<BoolAttr>()) {
      out->set_attr(name, attr.getValue());
    } else if (auto attr = kvp.second.dyn_cast<IntegerAttr>()) {
      out->set_attr(name, attr.getInt());
    } else if (auto attr = kvp.second.dyn_cast<FloatAttr>()) {
      out->set_attr(name, attr.getValueAsDouble());
    } else if (auto attr = kvp.second.dyn_cast<IntegerAttr>()) {
      out->set_attr(name, attr.getInt());
    } else if (auto attr = kvp.second.dyn_cast<StringAttr>()) {
      out->set_attr(name, attr.getValue().str());
    } else {
      IVLOG(1, "Attr: " << name.str());
      throw std::runtime_error("Invalid attribute during conversion");
    }
  }
}

void StripeBuilder::add_refinements(Block* block, Value* tensor, stripe::RefDir dir, std::string* name_out,
                                    std::string agg_name, bool is_spec) {
  // Compute all the info about the tensor
  auto ti = FlatTensorAccess{};
  auto sloc = StripeLocation{};
  std::tie(ti, sloc) = ComputeAccessAndLoc(tensor);
  size_t ndims = ti.access.size();
  // Translate allocation shape
  TensorShape base_shape = get_shape(ti.base_type);
  // Make a vector of 'inner' polynomials
  std::vector<AffinePolynomial> inner(ndims);
  std::vector<size_t> constants(ndims);
  for (size_t i = 0; i < ndims; i++) {
    constants[i] = ti.access[i].constant;
    ti.access[i].constant = 0;
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
    std::string& ref_name = blocks_.at(block).refs[ti];
    // If not, add the refinement
    if (ref_name == "") {
      if (auto rop = mlir::dyn_cast<RefineOp>(op)) {
        if (auto str_attr = rop.getAttrOfType<StringAttr>("name")) {
          ref_name = str_attr.getValue().str();
        }
      }
      if (ref_name == "") {
        ref_name = "ref";
      }
      ref_name = sblock->unique_ref_name(ref_name);
      std::vector<stripe::Affine> access;
      for (size_t i = 0; i < ndims; i++) {
        stripe::Affine aff;
        for (const auto& kvp : ti.access[i].terms) {
          if (kvp.first->getOwner() == block) {
            aff += stripe::Affine(get_idx(sblock, kvp.first), kvp.second);
          }
        }
        access.push_back(aff);
      }
      TensorShape shape = base_shape;
      for (size_t i = 0; i < ndims; i++) {
        auto range = AffineRange(inner[i]);
        if (!is_spec) {
          shape.dims[i].size = range.max - range.min + 1;
          shape.dims[i].size = std::min(shape.dims[i].size, base_shape.dims[i].size);
        }
        access[i] += constants[i];
        constants[i] = 0;
      }
      sblock->refs.emplace(dir, "", ref_name, access, shape, agg_name);
      // TODO: Only do this when we are 1-to-1
      agg_name = "";
    }
    // Connect up previously added block
    if (ref) {
      ref->from = ref_name;
    } else {
      *name_out = ref_name;
    }
    // Get pointer to new/etc reference
    ref = &blocks_.at(block).stripe->ref_by_into(ref_name)->mut();
    // Set the location
    for (const auto& dev : sloc.devs) {
      ref->location.devs.emplace_back(stripe::Device{dev.name});
      auto& sdev = ref->location.devs.back();
      for (const auto& unit : dev.units) {
        auto aff = stripe::Affine{unit.constant};
        for (const auto& kvp : unit.terms) {
          if (kvp.first->getOwner() == block) {
            aff += stripe::Affine(get_idx(sblock, kvp.first), kvp.second);
          }
        }
        sdev.units.emplace_back(std::move(aff));
      }
    }
    // Add in directionality
    if (ref->dir != stripe::RefDir::InOut && ref->dir != dir) {
      ref->dir = stripe::RefDir::InOut;
    }
    // Remove indexes from ti for this block + add to inner polynomial
    for (size_t i = 0; i < ndims; i++) {
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
    // If op matches the block, move the op up, also attributes
    while (op->getBlock() == block && mlir::isa<RefineOp>(op)) {
      if (auto attrs = op->getAttrOfType<DictionaryAttr>(Dialect::getStripeAttrsName())) {
        add_attributes(ref, attrs.getValue());
      }
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
  // Get full allocation shape
  ref->interior_shape = base_shape;
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
  // Move across the easy attrs
  if (auto attr = op.getAttrOfType<StringAttr>("name")) {
    cur_->name = attr.getValue().str();
  }
  if (auto attr = op.getAttrOfType<StringAttr>("comments")) {
    cur_->comments = attr.getValue().str();
  }
  // Add the 'true' indexes
  for (size_t i = 0; i < op.ranges().size(); i++) {
    int64_t range = op.ranges().getValue()[i].cast<IntegerAttr>().getInt();
    std::string idx_name = "idx";
    auto argName = llvm::formatv("arg{0}", i);
    if (auto attrs = op.getAttrOfType<DictionaryAttr>(argName.str())) {
      for (auto kvp : attrs.getValue()) {
        if (kvp.first.strref() == "__name") {
          idx_name = kvp.second.cast<StringAttr>().getValue().str();
        }
      }
    }
    idx_name = cur_->unique_idx_name(idx_name);
    idxs_.emplace(std::make_pair(cur_.get(), oblock.getArgument(i)), idx_name);
    cur_->idxs.emplace_back(idx_name, range);
    if (auto attrs = op.getAttrOfType<DictionaryAttr>(argName.str())) {
      add_attributes(&cur_->idxs.back(), attrs.getValue());
    }
  }
  // Add the attributes
  if (auto attrs = op.getAttrOfType<DictionaryAttr>(Dialect::getStripeAttrsName())) {
    add_attributes(cur_.get(), attrs.getValue());
  }
  // Add the location (if any) by checking the inputs to the block's closing return op.
  if (auto ret = mlir::dyn_cast<ReturnOp>(oblock.back())) {
    if (ret.getNumOperands() == 1) {  // Return ops may have zero or one argument/s.
      auto val = ret.getOperand(0);
      if (auto valType = val->getType().dyn_cast<TensorRefType>()) {
        auto offsets = std::vector<stripe::Affine>(valType.getRank());
        auto op = val->getDefiningOp();
        // Walk back refinements until we get to the TensorType.
        while (auto refOp = mlir::dyn_cast<RefineOp>(op)) {
          auto offIt = offsets.begin();
          for (auto offset : refOp.offsets()) {
            if (offIt == offsets.end()) {
              break;  // Should never happen; this is just to be careful.
            }
            *offIt++ += build_affine(cur_.get(), offset);
          }
          op = refOp.in()->getDefiningOp();
        }
        if (auto trefOp = mlir::dyn_cast<TensorRefOp>(op)) {
          if (auto tType = trefOp.in()->getType().dyn_cast<TensorType>()) {
            auto matches = llvm::SmallVector<StringRef, 4>();
            auto offIt = offsets.begin();
            for (const auto& dim : tType.getShape()) {
              static llvm::Regex re{R"(([[:alpha:]]+)_([[:digit:]]+)_([[:digit:]]+))"};
              if (re.match(dim.cls, &matches) && offIt != offsets.end()) {
                const auto& dev_name = matches[1];
                std::size_t dev_idx, unit_idx;
                matches[2].getAsInteger(10, dev_idx);
                matches[3].getAsInteger(10, unit_idx);
                if (cur_->location.devs.size() <= dev_idx) {
                  cur_->location.devs.resize(dev_idx + 1);
                }
                auto& dev = cur_->location.devs.at(dev_idx);
                dev.name = dev_name;
                if (dev.units.size() <= unit_idx) {
                  dev.units.resize(unit_idx + 1);
                }
                dev.units.at(unit_idx) = *offIt;
              }
              ++offIt;
            }
          }
        }
      }
    }
  }
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
    sblock->constraints.insert(sblock->constraints.begin(), build_affine(sblock, op.input()));
  } else {
    throw std::runtime_error("Complex contraints not supported right now");
  }
}

void StripeBuilder::visit(LoadOp op) {
  std::string ref_name;
  add_refinements(op.getOperation()->getBlock(), op.from(), stripe::RefDir::In, &ref_name);
  std::string into = scalar_name(op.getOperation());
  scalars_.emplace(op.into(), into);
  cur_->stmts.push_back(std::make_shared<stripe::Load>(ref_name, into));
}

void StripeBuilder::visit(LoadIndexOp op) {
  stripe::Affine from = build_affine(cur_.get(), op.from());
  std::string into = scalar_name(op.getOperation());
  scalars_.emplace(op.into(), into);
  cur_->stmts.push_back(std::make_shared<stripe::LoadIndex>(from, into));
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

void StripeBuilder::visit(SpecialOp op) {
  auto r = std::make_shared<stripe::Special>();
  size_t operand = 0;
  for (size_t i = 0; i < op.getNumOutputs(); i++) {
    std::string out_name;
    Operation* opr = op.getOperation();
    add_refinements(opr->getBlock(), opr->getOperand(operand++), stripe::RefDir::Out, &out_name, "", true);
    r->outputs.push_back(out_name);
  }
  for (size_t i = 0; i < op.getNumInputs(); i++) {
    std::string in_name;
    Operation* opr = op.getOperation();
    add_refinements(opr->getBlock(), opr->getOperand(operand++), stripe::RefDir::In, &in_name, "", true);
    r->inputs.push_back(in_name);
  }
  std::string dialect = op.getOperation()->getName().getDialect().str();
  std::string full_name = op.getOperation()->getName().getStringRef().str();
  r->name = full_name.substr(dialect.size() + 1, full_name.size() - dialect.size() - 1);
  cur_->stmts.push_back(r);
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
    } else if (auto op = mlir::dyn_cast<LoadIndexOp>(op_base)) {
      visit(op);
    } else if (auto op = mlir::dyn_cast<StoreOp>(op_base)) {
      visit(op);
    } else if (auto op = mlir::dyn_cast<AggregateOp>(op_base)) {
      visit(op);
    } else if (auto op = mlir::dyn_cast<SpecialOp>(op_base)) {
      visit(op);
    } else {
      found_inst_ = false;
      // Try all the intrinsic ops
      iop = &op_base;
      eltwise::ForAllOps(*this);
      // TODO: consider checking found_inst_.  However, since many instructions
      // (like affine computation) are *correct* to ignore, to do this we would
      // need some sort of whitelist of things to ignore... punting for now.
    }
  }
}

std::string StripeBuilder::scalar_name(Operation* op) {
  std::string out_name;
  auto attr = op->getAttr("scalar_name");
  if (attr) {
    auto name_attr = attr.template dyn_cast<StringAttr>();
    if (name_attr) {
      out_name = name_attr.getValue();
    }
  }
  auto& si = scalar_names_[cur_.get()];
  while (out_name == "" || si.names.count(out_name)) {
    out_name = "$s" + std::to_string(si.next++);
  }
  return out_name;
}

template <class ScalarOp>
void StripeBuilder::apply() {
  if (auto op = mlir::dyn_cast<ScalarOp>(iop)) {
    std::string out_name = scalar_name(op.getOperation());
    scalars_.emplace(op.result(), out_name);
    auto intr = std::make_shared<stripe::Intrinsic>();
    std::string dialect = op.getOperation()->getName().getDialect().str();
    std::string full_name = op.getOperation()->getName().getStringRef().str();
    intr->name = full_name.substr(dialect.size() + 1, full_name.size() - dialect.size() - 1);
    if (intr->name == "select") {
      intr->name = "cond";
    }
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
    std::string out_name = scalar_name(op.getOperation());
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
